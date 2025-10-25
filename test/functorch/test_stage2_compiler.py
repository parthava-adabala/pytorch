# Owner(s): ["oncall: pt2"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack

import torch
import torch.nn as nn
from torch._decomp import decomposition_table
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export, _restore_state_dict
from torch._functorch._aot_autograd.graph_compile import JointGraphCompiler
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._functorch.partitioners import default_partition
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStage2Compiler(TestCase):
    def test_simple_linear_stage_by_stage(self):
        """Test calling JointGraphCompiler methods individually."""

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleLinear()
        inputs = (torch.randn(4, 3, requires_grad=True),)

        # Step 1: Capture graph for export
        gm = _dynamo_graph_capture_for_export(model)(*inputs)
        _restore_state_dict(model, gm)

        with ExitStack() as stack:
            # Step 2: Export joint graph with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, gm, inputs, decompositions=decomposition_table
            )

            # Set up compilers in aot_config before using Stage2Compiler
            aot_state = joint_with_descriptors._aot_state
            aot_graph_capture = joint_with_descriptors._aot_graph_capture

            def nop_compiler(gm, args):
                return gm.forward

            aot_state.aot_config.partition_fn = default_partition
            aot_state.aot_config.fw_compiler = nop_compiler
            aot_state.aot_config.bw_compiler = nop_compiler

            # Step 3: Create JointGraphCompiler
            compiler = JointGraphCompiler(
                aot_state.aot_config,
                aot_state.fw_metadata,
                aot_graph_capture.maybe_subclass_meta,
            )

            # Step 4: Partition the joint graph
            partition_output = compiler.partition(
                aot_graph_capture.graph_module, aot_graph_capture.updated_flat_args
            )

            # Verify partition output
            self.assertIsNotNone(partition_output.fw_module)
            self.assertIsNotNone(partition_output.bw_module)
            self.assertGreater(partition_output.num_fw_outs_saved_for_bw, 0)
            self.assertIsInstance(partition_output.adjusted_flat_args, list)

            # Step 5: Compile forward
            fw_compile_output = compiler.fw_compile(
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)

            # Step 6: Compile backward
            bw_compile_output = compiler.bw_compile(
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Step 7: Create the final autograd function
            compiled_fn, fw_metadata = compiler.make_autograd_function(
                flat_args=aot_state.flat_args,
                wrappers=aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(compiled_fn)
            self.assertIsNotNone(fw_metadata)

            # Step 8: Wrap with cleaner calling convention
            model_fn = compiler.make_callable(
                compiled_fn,
                gm,
                joint_with_descriptors.params_spec,
                joint_with_descriptors.buffers_spec,
                joint_with_descriptors.in_spec,
                joint_with_descriptors.out_spec,
            )

        # Test functional correctness: model_fn should produce same results as original model

        # Test forward
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)
        torch.testing.assert_close(actual_output, expected_output)

        # Test backward - check that gradients match
        # Create fresh inputs for both eager and compiled
        inputs_eager = (torch.randn(4, 3, requires_grad=True),)
        inputs_compiled = (inputs_eager[0].detach().clone().requires_grad_(True),)

        # Run eager backward
        out_eager = model(*inputs_eager)
        out_eager.sum().backward()

        # Run compiled backward
        out_compiled = model_fn(*inputs_compiled)
        out_compiled.sum().backward()

        # Compare gradients for input
        torch.testing.assert_close(inputs_eager[0].grad, inputs_compiled[0].grad)

        # Compare gradients for parameters (note: gm has the parameters)
        for (name_eager, param_eager), (name_compiled, param_compiled) in zip(
            model.named_parameters(), gm.named_parameters()
        ):
            self.assertEqual(name_eager, name_compiled)
            torch.testing.assert_close(param_eager.grad, param_compiled.grad)

    def test_simple_linear_with_structured_io(self):
        """Test calling JointGraphCompiler with structured dict input and tuple output."""

        class SimpleLinearDict(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(4, 2)

            def forward(self, inputs):
                # Take a dict with two tensors and return a tuple
                x = self.linear1(inputs["x"])
                y = self.linear2(inputs["y"])
                return (x + y, x - y)

        model = SimpleLinearDict()
        inputs = ({"x": torch.randn(4, 3, requires_grad=True), "y": torch.randn(4, 4, requires_grad=True)},)

        # Step 1: Capture graph for export
        gm = _dynamo_graph_capture_for_export(model)(*inputs)
        _restore_state_dict(model, gm)

        with ExitStack() as stack:
            # Step 2: Export joint graph with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, gm, inputs, decompositions=decomposition_table
            )

            # Set up compilers in aot_config before using JointGraphCompiler
            aot_state = joint_with_descriptors._aot_state
            aot_graph_capture = joint_with_descriptors._aot_graph_capture

            def nop_compiler(gm, args):
                return gm.forward

            aot_state.aot_config.partition_fn = default_partition
            aot_state.aot_config.fw_compiler = nop_compiler
            aot_state.aot_config.bw_compiler = nop_compiler

            # Step 3: Create JointGraphCompiler
            compiler = JointGraphCompiler(
                aot_state.aot_config,
                aot_state.fw_metadata,
                aot_graph_capture.maybe_subclass_meta,
            )

            # Step 4: Partition the joint graph
            partition_output = compiler.partition(
                aot_graph_capture.graph_module, aot_graph_capture.updated_flat_args
            )

            # Verify partition output
            self.assertIsNotNone(partition_output.fw_module)
            self.assertIsNotNone(partition_output.bw_module)
            self.assertGreater(partition_output.num_fw_outs_saved_for_bw, 0)
            self.assertIsInstance(partition_output.adjusted_flat_args, list)

            # Step 5: Compile forward
            fw_compile_output = compiler.fw_compile(
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)

            # Step 6: Compile backward
            bw_compile_output = compiler.bw_compile(
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Step 7: Create the final autograd function
            compiled_fn, fw_metadata = compiler.make_autograd_function(
                flat_args=aot_state.flat_args,
                wrappers=aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(compiled_fn)
            self.assertIsNotNone(fw_metadata)

            # Step 8: Wrap with cleaner calling convention
            model_fn = compiler.make_callable(
                compiled_fn,
                gm,
                joint_with_descriptors.params_spec,
                joint_with_descriptors.buffers_spec,
                joint_with_descriptors.in_spec,
                joint_with_descriptors.out_spec,
            )

        # Test functional correctness: model_fn should preserve dict calling convention and tuple output

        # Test forward with dict input and tuple output
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)

        # Verify we got a tuple with 2 elements
        self.assertIsInstance(expected_output, tuple)
        self.assertIsInstance(actual_output, tuple)
        self.assertEqual(len(expected_output), 2)
        self.assertEqual(len(actual_output), 2)

        # Verify each element of the tuple matches
        torch.testing.assert_close(actual_output[0], expected_output[0])
        torch.testing.assert_close(actual_output[1], expected_output[1])

        # Test backward - check that gradients match
        # Create fresh inputs for both eager and compiled
        inputs_eager = ({"x": torch.randn(4, 3, requires_grad=True), "y": torch.randn(4, 4, requires_grad=True)},)
        inputs_compiled = ({"x": inputs_eager[0]["x"].detach().clone().requires_grad_(True),
                           "y": inputs_eager[0]["y"].detach().clone().requires_grad_(True)},)

        # Run eager backward (sum over both tuple outputs)
        out_eager = model(*inputs_eager)
        (out_eager[0].sum() + out_eager[1].sum()).backward()

        # Run compiled backward (sum over both tuple outputs)
        out_compiled = model_fn(*inputs_compiled)
        (out_compiled[0].sum() + out_compiled[1].sum()).backward()

        # Compare gradients for inputs
        torch.testing.assert_close(inputs_eager[0]["x"].grad, inputs_compiled[0]["x"].grad)
        torch.testing.assert_close(inputs_eager[0]["y"].grad, inputs_compiled[0]["y"].grad)

        # Compare gradients for parameters (note: gm has the parameters)
        for (name_eager, param_eager), (name_compiled, param_compiled) in zip(
            model.named_parameters(), gm.named_parameters()
        ):
            self.assertEqual(name_eager, name_compiled)
            torch.testing.assert_close(param_eager.grad, param_compiled.grad)

    def test_conv_bn_stage_by_stage(self):
        """Test JointGraphCompiler stage-by-stage with conv+batchnorm model."""

        class ConvBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 3, 3, padding=1)
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return torch.relu(x)

        model = ConvBN()
        model.eval()  # Use eval mode to avoid buffer mutations in functional correctness test
        inputs = (torch.randn(2, 1, 4, 4, requires_grad=True),)

        # Step 1: Capture graph for export
        gm = _dynamo_graph_capture_for_export(model)(*inputs)
        _restore_state_dict(model, gm)

        with ExitStack() as stack:
            # Step 2: Export joint graph with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, gm, inputs, decompositions=decomposition_table
            )

            # Set up state and graph capture
            aot_state = joint_with_descriptors._aot_state
            aot_graph_capture = joint_with_descriptors._aot_graph_capture

            def nop_compiler(gm, args):
                return gm.forward

            # Set up compilers
            aot_state.aot_config.partition_fn = default_partition
            aot_state.aot_config.fw_compiler = nop_compiler
            aot_state.aot_config.bw_compiler = nop_compiler

            # Step 3: Create JointGraphCompiler
            compiler = JointGraphCompiler(
                aot_state.aot_config,
                aot_state.fw_metadata,
                aot_graph_capture.maybe_subclass_meta,
            )

            # Step 4: Partition
            partition_output = compiler.partition(
                aot_graph_capture.graph_module, aot_graph_capture.updated_flat_args
            )

            self.assertIsNotNone(partition_output.fw_module)
            self.assertIsNotNone(partition_output.bw_module)

            # Step 5: Compile forward and backward
            fw_compile_output = compiler.fw_compile(
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            bw_compile_output = compiler.bw_compile(
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)
            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Step 6: Create the final autograd function
            compiled_fn, fw_metadata = compiler.make_autograd_function(
                flat_args=aot_state.flat_args,
                wrappers=aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(compiled_fn)
            self.assertIsNotNone(fw_metadata)

            # Step 7: Wrap with cleaner calling convention
            model_fn = compiler.make_callable(
                compiled_fn,
                gm,
                joint_with_descriptors.params_spec,
                joint_with_descriptors.buffers_spec,
                joint_with_descriptors.in_spec,
                joint_with_descriptors.out_spec,
            )

        # Test forward
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)
        torch.testing.assert_close(actual_output, expected_output)

        # Test backward - check that gradients match
        # Create fresh inputs for both eager and compiled
        inputs_eager = (torch.randn(2, 1, 4, 4, requires_grad=True),)
        inputs_compiled = (inputs_eager[0].detach().clone().requires_grad_(True),)

        # Run eager backward
        out_eager = model(*inputs_eager)
        out_eager.sum().backward()

        # Run compiled backward
        out_compiled = model_fn(*inputs_compiled)
        out_compiled.sum().backward()

        # Compare gradients for input
        torch.testing.assert_close(inputs_eager[0].grad, inputs_compiled[0].grad)

        # Compare gradients for parameters (note: gm has the parameters)
        for (name_eager, param_eager), (name_compiled, param_compiled) in zip(
            model.named_parameters(), gm.named_parameters()
        ):
            self.assertEqual(name_eager, name_compiled)
            torch.testing.assert_close(param_eager.grad, param_compiled.grad)

    def test_with_identity_passes(self):
        """Test applying identity passes between compilation stages."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModule()
        inputs = (torch.randn(4, 3, requires_grad=True),)

        # Identity pass that returns the graph unchanged
        def identity_pass(gm):
            """An identity pass that marks the graph as processed."""
            gm._identity_pass_applied = True
            return gm

        # Capture and export
        gm = _dynamo_graph_capture_for_export(model)(*inputs)
        _restore_state_dict(model, gm)

        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, gm, inputs, decompositions=decomposition_table
            )

            # Set up state and graph capture
            aot_state = joint_with_descriptors._aot_state
            aot_graph_capture = joint_with_descriptors._aot_graph_capture

            # Apply identity pass to joint graph
            aot_graph_capture.graph_module = identity_pass(aot_graph_capture.graph_module)
            self.assertTrue(hasattr(aot_graph_capture.graph_module, '_identity_pass_applied'))

            def nop_compiler(gm, args):
                return gm.forward

            # Set up compilers
            aot_state.aot_config.partition_fn = default_partition
            aot_state.aot_config.fw_compiler = nop_compiler
            aot_state.aot_config.bw_compiler = nop_compiler

            # Create compiler and partition
            compiler = JointGraphCompiler(
                aot_state.aot_config,
                aot_state.fw_metadata,
                aot_graph_capture.maybe_subclass_meta,
            )

            partition_output = compiler.partition(
                aot_graph_capture.graph_module, aot_graph_capture.updated_flat_args
            )

            # Apply identity pass to forward and backward modules
            partition_output.fw_module = identity_pass(partition_output.fw_module)
            partition_output.bw_module = identity_pass(partition_output.bw_module)

            self.assertTrue(hasattr(partition_output.fw_module, '_identity_pass_applied'))
            self.assertTrue(hasattr(partition_output.bw_module, '_identity_pass_applied'))

            # Continue compilation
            fw_compile_output = compiler.fw_compile(
                partition_output.fw_module,
                partition_output.adjusted_flat_args,
                partition_output.num_fw_outs_saved_for_bw,
            )

            bw_compile_output = compiler.bw_compile(
                partition_output.bw_module,
                fw_compile_output.fwd_output_strides,
                partition_output.num_symints_saved_for_bw,
            )

            self.assertIsNotNone(fw_compile_output.compiled_fw_func)
            self.assertIsNotNone(bw_compile_output.lazy_backward_info)

            # Create the final autograd function
            compiled_fn, fw_metadata = compiler.make_autograd_function(
                flat_args=aot_state.flat_args,
                wrappers=aot_graph_capture.wrappers,
                compiled_fw_func=fw_compile_output.compiled_fw_func,
                compiled_bw_func=bw_compile_output.compiled_bw_func,
                lazy_backward_info=bw_compile_output.lazy_backward_info,
                indices_of_inps_to_detach=partition_output.indices_of_inps_to_detach,
                num_symints_saved_for_bw=partition_output.num_symints_saved_for_bw,
            )
            self.assertIsNotNone(compiled_fn)
            self.assertIsNotNone(fw_metadata)

            # Wrap with cleaner calling convention
            model_fn = compiler.make_callable(
                compiled_fn,
                gm,
                joint_with_descriptors.params_spec,
                joint_with_descriptors.buffers_spec,
                joint_with_descriptors.in_spec,
                joint_with_descriptors.out_spec,
            )

        # Test functional correctness: model_fn should produce same results as original model
        expected_output = model(*inputs)
        actual_output = model_fn(*inputs)
        torch.testing.assert_close(actual_output, expected_output)


if __name__ == "__main__":
    run_tests()
