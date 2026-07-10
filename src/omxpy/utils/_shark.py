"""
_shark.py
---------
Helper module for checking Jupyter kernel / node info and Slurm job info
when running notebooks on the Shark/ALICE HPC cluster via VSCode Remote-SSH.

Usage (inside a notebook cell):

    import _shark
    _shark.report()               # prints everything (kernel + job info)

    # or individually
    _shark.print_kernel_info()
    _shark.print_job_info()

    # or get the info as dicts for programmatic use
    kernel = _shark.get_kernel_info()
    job    = _shark.get_job_info()
"""

import sys
import os
import re
import socket
import subprocess
from datetime import datetime


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _sh(cmd, timeout=10):
    """Run a shell command and return stdout, or '' on failure."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _grab(pattern, text, default="N/A"):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _parse_mem(m):
    """Convert Slurm memory strings (e.g. '64000M') into 'MB (GB)' format."""
    if m in ("N/A", None):
        return "N/A"
    num = re.sub(r"[^\d]", "", m)
    if not num:
        return m
    mb = int(num)
    return f"{mb} MB ({mb / 1024:.0f} GB)"


# -------------------------------------------------------------------------
# Kernel / node info
# -------------------------------------------------------------------------

def get_kernel_info():
    """Return a dict with hostname, python/ipython, and kernel details."""
    info = {
        "hostname": socket.gethostname(),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", "N/A"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "slurm_nodeid": os.environ.get("SLURM_NODEID", "N/A"),
        "python_executable": sys.executable,
        "python_version": sys.version.split("\n")[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "virtual_env": os.environ.get("VIRTUAL_ENV", "N/A"),
    }

    try:
        from IPython import __version__ as ipy_version
        info["ipython_version"] = ipy_version
    except ImportError:
        info["ipython_version"] = "N/A"

    try:
        import ipykernel
        info["ipykernel_path"] = ipykernel.__file__
        info["connection_file"] = ipykernel.get_connection_file()
    except Exception:
        info["ipykernel_path"] = "N/A"
        info["connection_file"] = "N/A"

    for cmd_name, cmd in [
        ("which_python", "which python"),
        ("which_python3", "which python3"),
        ("which_jupyter", "which jupyter"),
        ("which_ipython", "which ipython"),
    ]:
        info[cmd_name] = _sh(cmd)

    return info


def print_kernel_info():
    info = get_kernel_info()
    print("# -------------------- NODE / KERNEL INFO --------------------")
    print(f"Hostname              : {info['hostname']}")
    print(f"SLURM_JOB_NODELIST    : {info['slurm_job_nodelist']}")
    print(f"SLURM_JOB_ID          : {info['slurm_job_id']}")
    print(f"SLURM_NODEID          : {info['slurm_nodeid']}")
    print()
    print(f"Python Executable     : {info['python_executable']}")
    print(f"Python Version        : {info['python_version']}")
    print(f"IPython Version       : {info['ipython_version']}")
    print(f"Conda Env             : {info['conda_env']}")
    print(f"Virtual Env           : {info['virtual_env']}")
    print()
    print(f"ipykernel Path        : {info['ipykernel_path']}")
    print(f"Connection File       : {info['connection_file']}")
    print()
    print(f"which python          : {info['which_python']}")
    print(f"which python3         : {info['which_python3']}")
    print(f"which jupyter         : {info['which_jupyter']}")
    print(f"which ipython         : {info['which_ipython']}")


# -------------------------------------------------------------------------
# Slurm job info
# -------------------------------------------------------------------------

def get_job_info():
    """Return a dict with Slurm job allocation and runtime details (live job)."""
    job_id = os.environ.get(
        "SLURM_JOB_ID", _sh("squeue -h -u $USER -o '%A' | head -n1")
    )
    scontrol_out = _sh(f"scontrol show job {job_id}")

    # The node the kernel is *actually* executing on right now.
    # scontrol's NodeList can read "(null)" depending on when/where it's
    # queried, so fall back to values only the compute node itself knows.
    current_node = (
        os.environ.get("SLURMD_NODENAME")
        or os.environ.get("SLURM_NODELIST")
        or os.environ.get("SLURM_JOB_NODELIST")
        or socket.gethostname()
    )

    info = {
        "job_id": job_id,
        "job_name": _grab(r"JobName=(\S+)", scontrol_out),
        "job_user": _grab(r"UserId=(\S+?)\(", scontrol_out),
        "submit_host": _grab(
            r"SubmitHost=(\S+)", scontrol_out,
            default=os.environ.get("SLURM_SUBMIT_HOST", "N/A"),
        ),
        "run_node": _grab(
            r"NodeList=(\S+)", scontrol_out,
            default=os.environ.get("SLURM_JOB_NODELIST", "N/A"),
        ),
        "current_node": current_node,          # <-- new
        "current_hostname": socket.gethostname(),  # <-- new
        "submit_dir": _grab(
            r"WorkDir=(\S+)", scontrol_out,
            default=os.environ.get("SLURM_SUBMIT_DIR", "N/A"),
        ),
        "num_nodes": _grab(
            r"NumNodes=(\S+)", scontrol_out,
            default=os.environ.get("SLURM_JOB_NUM_NODES", "N/A"),
        ),
        "num_tasks": _grab(
            r"NumTasks=(\S+)", scontrol_out,
            default=os.environ.get("SLURM_NTASKS", "N/A"),
        ),
        "tasks_per_node": os.environ.get("SLURM_NTASKS_PER_NODE", "N/A"),
        "cpus_per_task": os.environ.get(
            "SLURM_CPUS_PER_TASK", _grab(r"CPUs/Task=(\S+)", scontrol_out)
        ),
        "num_cpus_total": _grab(r"NumCPUs=(\S+)", scontrol_out),
        "mem_per_node_raw": _grab(
            r"MinMemoryNode=(\S+)", scontrol_out,
            default=_grab(r"MinMemoryCPU=(\S+)", scontrol_out),
        ),
        "gpu_alloc": _grab(r"Gres=(\S+)", scontrol_out, default="N/A"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "start_time": _grab(r"StartTime=(\S+)", scontrol_out),
        "time_limit": _grab(r"TimeLimit=(\S+)", scontrol_out),
        "run_time": _grab(r"RunTime=(\S+)", scontrol_out),
        "now": datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
    }

    info["mem_per_node"] = _parse_mem(info["mem_per_node_raw"])
    info["gpu_hardware"] = (
        _sh("nvidia-smi --query-gpu=name --format=csv,noheader")
        or "nvidia-smi not found"
    )

    return info


def print_job_info():
    info = get_job_info()
    print("# -------------------- JOB INFORMATION --------------------")
    print(f"Job Name              : {info['job_name']}")
    print(f"Job ID                : {info['job_id']}")
    print(f"Job User              : {info['job_user']}")
    print(f"Submit Host           : {info['submit_host']}")
    print(f"Run Node              : {info['run_node']}")
    print(f"Current Exec Node     : {info['current_node']}")   # <-- new
    print(f"Current Hostname      : {info['current_hostname']}")  # <-- new
    print(f"Submit Dir            : {info['submit_dir']}")
    print(f"Nodes                 : {info['num_nodes']}")
    print(f"Tasks                 : {info['num_tasks']}")
    print(f"Tasks per Node        : {info['tasks_per_node']}")
    print(f"CPUs per Task         : {info['cpus_per_task']}")
    print(f"CPUs (Job Total)      : {info['num_cpus_total']}")
    print(f"Memory per Node       : {info['mem_per_node']}")
    print(f"GPUs Allocated        : {info['gpu_alloc']}")
    print(f"CUDA Visible Devs     : {info['cuda_visible_devices']}")
    print(f"GPU Hardware          : {info['gpu_hardware']}")
    print("# -------------------- RUNTIME INFORMATION ----------------")
    print(f"Date Start            : {info['start_time']}")
    print(f"Date Now              : {info['now']}")
    print(f"SLURM Run Time        : {info['run_time']}")
    print(f"SLURM Time Limit      : {info['time_limit']}")
    print("# ==================== JOB LIVE ============================")


# -------------------------------------------------------------------------
# Combined report + historical (post-job) lookup
# -------------------------------------------------------------------------

def report():
    """Print both kernel/node info and live job info."""
    print_kernel_info()
    print()
    print_job_info()


def print_job_history(job_id=None):
    """
    Print final job accounting info via sacct.
    Only accurate once the job has finished; use `job_id` to check a past job.
    """
    job_id = job_id or os.environ.get("SLURM_JOB_ID", "N/A")
    print(_sh(
        f"sacct -j {job_id} "
        "--format=JobID,JobName,Elapsed,Start,End,State,NNodes,NCPUS,ReqMem,NodeList -p"
    ))


if __name__ == "__main__":
    report()