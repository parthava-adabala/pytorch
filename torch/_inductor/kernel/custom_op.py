# Owner(s): ["module: inductor"]

import functools
from typing import Any, Callable, Optional, Union

import torch
from torch._inductor import config
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import Buffer, FixedLayout, ir_node_to_tensor, TensorBox
from torch._inductor.lowering import lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.virtualized import V


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs.
    Separates tensors (for autotuning input_nodes) from non-tensor parameters.
    Non-tensor kwargs are later functools.partial'd into decomposition functions.

    Args:
        args: Positional arguments (mix of tensors and scalars)
        kwargs: Keyword arguments (mix of tensors and scalars)

    Returns:
        Tuple of (tensor_inputs_list, non_tensor_kwargs)
    """
    tensor_inputs = []
    non_tensor_kwargs = {}

    # Process args and kwargs: separate tensor inputs and non tensor args
    for i, arg in enumerate(args):
        if isinstance(arg, (TensorBox, Buffer)):
            tensor_inputs.append(arg)
        else:
            # Add non-tensor positional args to kwargs with generated names
            non_tensor_kwargs[f"arg_{i}"] = arg

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer)):
            tensor_inputs.append(value)
        else:
            non_tensor_kwargs[key] = value

    return tensor_inputs, non_tensor_kwargs


def _create_user_input_gen_fns(
    inputs: list[Any],
    arg_names: list[str],
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators from name-based to index-based format.
       Inductor autotune's input_gen_fns expects index of arg_names as key.

    Uses V.graph.sizevars.size_hints() to guess best for dynamic shapes.
    """

    name_to_index = {name: i for i, name in enumerate(arg_names)}
    index_based_fns = {}

    for name, gen_fn in user_input_gen_fns.items():
        if name in name_to_index:
            index_based_fns[name_to_index[name]] = gen_fn
        else:
            print(f"Warning: Unknown argument name '{name}' in input_gen_fns")

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor], arg_name: str
    ) -> Callable[[Any], torch.Tensor]:
        """Create internal input generator that converts IR buffer to user's fake tensor."""

        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            raw_shape = ir_buffer.get_size()
            concrete_shape = V.graph.sizevars.size_hints(
                raw_shape, fallback=config.unbacked_symint_fallback
            )

            fake_tensor = torch.empty(
                concrete_shape, dtype=ir_buffer.get_dtype(), device="meta"
            )
            return user_function(fake_tensor)

        return internal_input_gen_fn

    return {
        i: create_internal_input_gen_fn(
            user_gen_fn, arg_names[i] if i < len(arg_names) else f"arg_{i}"
        )
        for i, user_gen_fn in index_based_fns.items()
        if i < len(inputs)
    }


def _create_fallback_choice(
    name: str,
    default_impl: Callable[..., Any],
    fake_output: torch.Tensor,
    kwargs: dict[str, Any],
) -> ExternKernelChoice:
    """Create fallback choice for default implementation."""

    def fallback_wrapper(*args: Any) -> Any:
        return default_impl(*args, **kwargs)

    return ExternKernelChoice(
        kernel=fallback_wrapper,
        name=f"{name}_fallback_default",
        has_out_variant=False,
        op_overload=default_impl,
        use_fallback_kernel=True,
    )


def _create_parameter_variants(
    decompositions: list[Callable[..., Any]],
    tuning_knob: dict[str, list[Any]],
) -> list[Any]:  # Returns partial objects which are callable
    """Create parameter variants for decompositions using tuning knob.

    Args:
        decompositions: Base implementation functions
        tuning_knob: Parameter tuning dict with parameter names and value lists

    Returns:
        List of variant functions with all parameter combinations
    """
    # Validate parameter values
    for param_name, param_values in tuning_knob.items():
        if not param_values or not isinstance(param_values, (list, tuple)):
            raise TypeError(
                f"Parameter values for '{param_name}' must be a list or tuple, got {type(param_values)}"
            )

    # Generate all combinations of parameter values using Cartesian product
    import itertools

    param_names = list(tuning_knob.keys())
    param_values_lists = list(tuning_knob.values())
    param_combinations = list(itertools.product(*param_values_lists))

    # Create variants for each decomposition with each parameter combination
    variants = []
    for decomp_fn in decompositions:
        for param_combo in param_combinations:
            # Create kwargs dict for this combination
            param_kwargs = dict(zip(param_names, param_combo))

            # Create partial function with all parameters
            variant = functools.partial(decomp_fn, **param_kwargs)
            param_suffix = "_".join(
                f"{name}_{value}" for name, value in param_kwargs.items()
            )
            variant.__name__ = f"{decomp_fn.__name__}_{param_suffix}"  # type: ignore[attr-defined]
            variants.append(variant)

    return variants


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    kwargs: Optional[dict[str, Any]] = None,
    default_impl: Optional[Callable[..., Any]] = None,
    user_input_gen_fns: Optional[
        dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
    enable_epilogue_fusion: bool = False,
    enable_prologue_fusion: bool = False,
    disable_fallback: bool = False,
) -> Union[TensorBox, Any]:
    """Autotune custom operations by comparing multiple decomposition implementations.

    Currently supports SINGLE OUTPUT custom ops only.
    TODO: Add support for multiple output custom ops (tuple/list returns).

    This function generates multiple implementation choices for a custom operation and
    uses Inductor's autotuning system to select the best performing variant at runtime.
    After selecting the best choice, optionally applies inline epilogue fusion.

    Args:
        name: Unique identifier for the autotuning operation
        decompositions: List of alternative implementation functions to benchmark
        inputs: Input tensor IR nodes from compilation (TensorBox/Buffer objects)
        kwargs: Non-tensor parameters to pass to decomposition functions
        default_impl: Original custom op implementation used as fallback
        user_input_gen_fns: Optional custom input generators for benchmarking.
                           Maps input indices to functions that take fake tensors
                           and return real tensors for performance measurement.
        enable_epilogue_fusion: If True, apply inline epilogue fusion to the best choice

    Returns:
        IR node representing the optimized operation result

    Raises:
        TypeError: If decompositions is not a list/tuple
        RuntimeError: If no inputs or no valid choices generated
    """
    if kwargs is None:
        kwargs = {}

    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not inputs:
        raise RuntimeError(f"Custom op '{name}' requires tensor inputs for autotuning")

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=list(decompositions),
        input_nodes=list(inputs),
        kwargs=kwargs,
    )

    # Add default implementation as fallback (unless disabled)
    if default_impl and hasattr(default_impl, "_op") and not disable_fallback:
        fallback_name = f"{name}_fallback_default"
        from torch._inductor.select_algorithm import extern_kernels

        # Skip if extern_kernel already registered to avoid duplicate registration error
        if not hasattr(extern_kernels, fallback_name):
            with V.fake_mode:
                fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
                fake_output = default_impl(*fake_inputs, **kwargs)

            fallback_choice = _create_fallback_choice(
                name, default_impl, fake_output, kwargs
            )
            fallback_choice.maybe_append_choice(
                choices=choices,
                input_nodes=list(inputs),
                layout=FixedLayout(
                    device=fake_output.device,
                    dtype=fake_output.dtype,
                    size=fake_output.shape,
                    stride=fake_output.stride(),
                ),
            )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    # Convert user input generation functions to internal format
    input_gen_fns = {}
    if user_input_gen_fns:
        import inspect

        arg_names = (
            list(inspect.signature(decompositions[0]).parameters.keys())
            if decompositions
            else []
        )
        input_gen_fns = _create_user_input_gen_fns(
            inputs, arg_names, user_input_gen_fns
        )

    # Run autotuning to select the best choice
    selected_result = autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
    )

    # Apply inlining if epilogue fusion is enabled
    if enable_epilogue_fusion and isinstance(selected_result, TensorBox):
        # Find the winning choice that was selected during autotuning
        winning_choice = None

        # Debug: Let's understand the structure of selected_result
        print(f"🔍 Debugging selected_result: {type(selected_result)}")
        print(f"🔍 selected_result.data: {type(selected_result.data)}")
        if hasattr(selected_result.data, "__dict__"):
            print(
                f"🔍 selected_result.data attributes: {list(selected_result.data.__dict__.keys())}"
            )

        # Try different ways to find the winning choice
        if hasattr(selected_result, "data") and hasattr(
            selected_result.data, "subgraph_name"
        ):
            # SubgraphBuffer case - find matching choice by name
            subgraph_name = selected_result.data.subgraph_name
            print(f"🔍 Looking for subgraph_name: {subgraph_name}")
            for choice in choices:
                print(f"🔍 Choice name: {choice.name}")
                if choice.name == subgraph_name:
                    winning_choice = choice
                    break

        # Alternative: The first choice might be the winner if we can't find exact match
        if not winning_choice and choices:
            print(f"🔍 Using first choice as fallback: {choices[0].name}")
            winning_choice = choices[0]

        if winning_choice:
            print(f"🎯 Inlining winning choice: {winning_choice.name}")
            try:
                # Inline the winning choice operations into the main graph
                inlined_result = _inline_custom_op_choice(winning_choice, inputs, name)
                return inlined_result
            except Exception as e:
                print(f"❌ Inlining failed: {e}")
                print("⚠️  Falling back to marking approach")
        else:
            print(
                "⚠️  Could not find winning choice for inlining, falling back to marking"
            )

    # Mark result for custom op fusion if enabled (fallback path)
    if enable_epilogue_fusion and isinstance(selected_result, TensorBox):
        _mark_custom_op_for_epilogue_fusion(selected_result, name)

    if enable_prologue_fusion and isinstance(selected_result, TensorBox):
        _mark_custom_op_for_prologue_fusion(selected_result, name)

    return selected_result


def _inline_custom_op_choice(winning_choice, inputs: list[Any], name: str) -> TensorBox:
    """Inline the winning custom op choice by converting its FX operations to individual IR nodes.

    This converts the custom op from a single ExternKernel (unfusable) to multiple ComputedBuffer
    nodes (fusable), enabling epilogue fusion with subsequent operations.

    Args:
        winning_choice: The winning SubgraphChoiceCaller from autotuning
        inputs: Original input nodes
        name: Custom op name for debugging

    Returns:
        TensorBox containing the final operation result as individual IR nodes
    """
    from torch._inductor.lowering import lowerings

    # Get the GraphModule containing the operations
    gm = winning_choice.gm

    # Create a temporary graph lowering context to process the FX nodes
    # We'll extract the operations and add them to the current graph
    current_graph = V.graph

    # Create mapping from placeholder nodes to actual inputs
    node_to_value = {}
    placeholder_idx = 0

    # Process each node in the winning choice's graph
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Map placeholder to actual input
            if placeholder_idx < len(inputs):
                node_to_value[node] = inputs[placeholder_idx]
                placeholder_idx += 1
            else:
                raise RuntimeError(f"Not enough inputs for placeholder {node.name}")

        elif node.op == "call_function":
            # Convert FX operation to IR nodes using existing lowerings
            target = node.target
            args = [
                node_to_value[arg] if arg in node_to_value else arg for arg in node.args
            ]
            kwargs = {
                k: node_to_value[v] if v in node_to_value else v
                for k, v in node.kwargs.items()
            }

            # Call the appropriate lowering function
            if target in lowerings:
                result = lowerings[target](*args, **kwargs)
                node_to_value[node] = result
            else:
                # Fallback: try calling the target directly
                result = target(*args, **kwargs)
                node_to_value[node] = result

        elif node.op == "output":
            # Return the final result
            output_arg = node.args[0]
            if isinstance(output_arg, (list, tuple)):
                # Multi-output case (not yet supported)
                raise RuntimeError(
                    "Multi-output custom ops not yet supported for inlining"
                )
            else:
                # Single output case
                final_result = node_to_value[output_arg]
                return final_result

        else:
            raise RuntimeError(f"Unsupported node type: {node.op}")

    raise RuntimeError("No output node found in custom op graph")


def _mark_custom_op_for_epilogue_fusion(result: TensorBox, name: str) -> None:
    """Mark the result for custom op epilogue fusion by the scheduler.

    Args:
        result: The autotuning result to mark
        name: Operation name for identification
    """
    if hasattr(result, "data") and hasattr(result.data, "get_name"):
        # Mark this buffer as a custom op result eligible for epilogue fusion
        if not hasattr(result.data, "_custom_op_fusion_metadata"):
            result.data._custom_op_fusion_metadata = {}

        result.data._custom_op_fusion_metadata.update(
            {
                "epilogue_fusion_enabled": True,
                "custom_op_name": name,
            }
        )


def _mark_custom_op_for_prologue_fusion(result: TensorBox, name: str) -> None:
    """Mark the result for custom op prologue fusion by the scheduler.

    Args:
        result: The autotuning result to mark
        name: Operation name for identification
    """
    if hasattr(result, "data") and hasattr(result.data, "get_name"):
        # Mark this buffer as a custom op result eligible for prologue fusion
        if not hasattr(result.data, "_custom_op_fusion_metadata"):
            result.data._custom_op_fusion_metadata = {}

        result.data._custom_op_fusion_metadata.update(
            {
                "prologue_fusion_enabled": True,
                "custom_op_name": name,
            }
        )


def register_custom_op_autotuning(
    custom_op: torch._ops.OpOverload,
    decompositions: list[Callable[..., Any]],
    name: Optional[str] = None,
    input_gen_fns: Optional[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    tuning_knob: Optional[dict[str, list[Any]]] = None,
    enable_epilogue_fusion: bool = False,
    enable_prologue_fusion: bool = False,
    disable_fallback: bool = False,
) -> None:
    """Register custom operation for autotuning with multiple implementations.

    Supports multiple decompositions and optional parameter tuning.
    When tuning_knob is provided, creates variants of each decomposition
    with all combinations of parameter values (Cartesian product).

    Args:
        custom_op: Custom operation to register (e.g., torch.ops.mylib.myop.default)
        decompositions: Implementation functions to benchmark
        name: Operation name for identification (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking
        tuning_knob: Optional parameter tuning dict. Supports multiple parameters.
                    Creates all combinations of parameter values.

    Raises:
        TypeError: If decompositions is not a list/tuple
        ValueError: If no decompositions provided

    Example:
        import math

        def attention_variants(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, method: int = 0) -> torch.Tensor:
            if method == 0:   # Standard attention
                scale = 1.0 / math.sqrt(q.size(-1))
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(scores, dim=-1)
                return torch.matmul(attn_weights, v)

            elif method == 1: # Flash Attention
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                )

            elif method == 2: # Flex Attention
                def score_mod(score, b, h, m, n):
                    return score  # Identity - no modification
                return torch.nn.attention.flex_attention(q, k, v, score_mod=score_mod)

        # Register autotuning with parameter variants
        register_custom_op_autotuning(
            custom_op=torch.ops.mylib.attention.default,
            decompositions=[attention_variants],
            tuning_knob={"method": [0, 1, 2]},  # Standard vs Flash vs Flex
            input_gen_fns={
                0: lambda fake: torch.randn_like(fake, device='cuda') * 0.1,  # query
                1: lambda fake: torch.randn_like(fake, device='cuda') * 0.1,  # key
                2: lambda fake: torch.randn_like(fake, device='cuda') * 0.1,  # value
            }
        )
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple, got {type(decompositions)}"
        )

    if not decompositions:
        raise ValueError("At least one decomposition must be provided")

    if name is None:
        name = f"{custom_op._name}_autotuned"

    # Generate final decomposition list with optional parameter variants
    if tuning_knob is None:
        final_decompositions = list(decompositions)
    else:
        final_decompositions = _create_parameter_variants(decompositions, tuning_knob)

    @functools.wraps(custom_op)
    def autotuning_lowering(*args: Any, **kwargs: Any) -> Any:
        """Inductor lowering function that replaces custom op calls with autotuned versions."""
        # Extract tensor inputs and non-tensor parameters
        tensor_inputs, non_tensor_kwargs = _extract_tensor_inputs(args, kwargs)

        result = autotune_custom_op(
            name=name,
            decompositions=final_decompositions,
            inputs=tensor_inputs,
            kwargs=non_tensor_kwargs,
            default_impl=custom_op,
            user_input_gen_fns=input_gen_fns,
            enable_epilogue_fusion=enable_epilogue_fusion,
            enable_prologue_fusion=enable_prologue_fusion,
            disable_fallback=disable_fallback,
        )

        validate_ir(result)
        return result

    lowerings[custom_op] = autotuning_lowering


# Example of using inline epilogue fusion:
#
# # Method 1: Per-operation epilogue fusion
# register_custom_op_autotuning(
#     custom_op=torch.ops.mylib.myop.default,
#     decompositions=[decomp1, decomp2, decomp3],
#     enable_epilogue_fusion=True,  # Enable inline epilogue fusion
# )
#
# # Method 2: Enable globally via config
# import torch._inductor.config as config
# config.enable_custom_op_epilogue_fusion = True
#
# register_custom_op_autotuning(
#     custom_op=torch.ops.mylib.myop.default,
#     decompositions=[decomp1, decomp2, decomp3],
#     # Epilogue fusion enabled by global flag
# )
#
# The inline epilogue fusion system:
# 1. Run autotuning to select the fastest implementation choice
# 2. Apply inline epilogue fusion to the selected best choice if enabled
# 3. Mark the result for fusion-eligibility for the scheduler's existing fusion passes
# 4. Leverages today's fusion support without extending beyond current capabilities
