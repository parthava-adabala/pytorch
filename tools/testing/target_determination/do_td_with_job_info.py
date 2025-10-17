import json
import yaml
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def get_job_info_from_workflow_file(workflow_file: str) -> list[list[dict[str, Any]]]:
    """
    Returns groups of jobs that are similar based on the test configurations
    they run.

    This is pretty hardcoded, so it is fragile, but it returns a pretty accurate
    mapping

    TODO replace with better (automated?) system. Maybe a separate workflow that
    generates an artifact that says which jobs are similar according what tests
    they run, correlation etc, also looks at jobs on main branch or merge base
    to better determine what jobs exist.
    """
    workflow_file = workflow_file.split("@")[-1] # remove any reference to branch/commit
    regex = r"needs\.([a-zA-Z0-9_-]+)\.outputs\.test-matrix"

    with open(REPO_ROOT / workflow_file, "r") as f:
        yml = yaml.safe_load(f)
    jobs = yml.get("jobs", {})

    jobs_with_tests: list[dict[str, Any]] = []
    dependent_jobs = {}

    for job, job_info in jobs.items():
        if "test-matrix" not in job_info.get("with", {}):
            continue
        try:
            test_matrix = yaml.safe_load(job_info["with"]["test-matrix"])
            if "include" not in test_matrix:
                # ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm86-build.outputs.test-matrix }}
                match = re.search(regex, test_matrix)
                if match:
                    dep_job = match.group(1)
                    dependent_jobs[f"{job_info.get('name', job)}"] = {
                        "depends_on": f"{dep_job}",
                        "is_test": "test" in job_info.get("uses", ""),
                    }
                continue
        except yaml.YAMLError as e:
            print(f"Error parsing test-matrix for job {job}: {e}")
            continue
        jobs_with_tests.append(
            {
                "job_id": f"{job}",
                "job_name": f"{job_info.get('name', job)}",
                "test_matrix": sorted(set(entry["config"] for entry in test_matrix["include"])),
                "is_test": "test" in job_info.get("uses", ""),
            }
        )
    # Fill in dependent jobs
    for job, info in dependent_jobs.items():
        for j in jobs_with_tests:
            if j["job_id"] == info["depends_on"]:
                jobs_with_tests.append(
                    {
                        "job_id": job,
                        "job_name": job,
                        "test_matrix": j["test_matrix"],
                        "is_test": info["is_test"],
                    }
                )

    # Remove non test jobs
    jobs_with_tests = [j for j in jobs_with_tests if j.get("is_test", False)]

    # Dedup by name
    jobs_seen = set()
    jobs_with_tests = [j for j in jobs_with_tests if not (j["job_name"] in jobs_seen or jobs_seen.add(j["job_name"]))]

    # Remove job_id
    for j in jobs_with_tests:
        j.pop("job_id", None)

    individual_jobs = [
        {"job_name": j["job_name"], "config": config} for j in jobs_with_tests for config in j["test_matrix"]
    ]

    # Group the jobs together
    # generally same test config -> same group

    grouped_jobs = {}
    for job in individual_jobs:
        key = []
        if "onnx" in job["job_name"]:
            key.append("onnx")
        if "bazel" in job["job_name"]:
            key.append("bazel")
        if "cuda" in job["job_name"]:
            key.append("cuda")
        if "mac" in job["job_name"]:
            key.append("mac")
        if "windows" in job["job_name"]:
            key.append("windows")
        key.append(job["config"])
        key_str = "|".join(sorted(key))
        if key_str not in grouped_jobs:
            grouped_jobs[key_str] = []
        grouped_jobs[key_str].append(job)

    return list(grouped_jobs.values())


def get_all_workflow_files() -> list[Path]:
    """
    Get all GitHub workflow files in the .github/workflows directory.
    """
    workflow_dir = REPO_ROOT / ".github" / "workflows"
    return list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))


if __name__ == "__main__":
    all_jobs = []

    jobs = get_job_info_from_workflow_file(".github/workflows/pull.yml")
    print(json.dumps(jobs, indent=2))
