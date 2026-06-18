"""Launching and lifecycle management of headless ``runSofa`` subprocesses.

Works with any SOFA build that ships SofaPython3: the executable, plugins and
environment all come from the :class:`~sofaopt.project.SofaOptProject`. On
Windows, child processes are attached to a kill-on-close Job Object so they
never outlive the optimizer.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from sofaopt.core import envkeys
from sofaopt.project import SofaOptProject

# --- Windows Job Object: kill all SOFA children if the optimizer dies ---------
SOFA_JOB_HANDLE = None


def ensure_windows_sofa_job() -> None:
    """Create one kill-on-close Job Object for SOFA children (Windows only)."""
    global SOFA_JOB_HANDLE
    if os.name != "nt" or SOFA_JOB_HANDLE is not None:
        return

    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise OSError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")

    SOFA_JOB_HANDLE = job


def attach_process_to_sofa_job(proc: subprocess.Popen) -> None:
    """Attach a child process to the kill-on-close job (Windows only)."""
    if os.name != "nt":
        return
    ensure_windows_sofa_job()
    if SOFA_JOB_HANDLE is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    if not kernel32.AssignProcessToJobObject(
        SOFA_JOB_HANDLE, wintypes.HANDLE(proc._handle)
    ):
        err = ctypes.get_last_error()
        print(f"[warn] Could not attach SOFA process {proc.pid} to job (winerr={err}).")


def launch_sofa(
    project: SofaOptProject,
    *,
    scene_file: Path,
    test_name: str,
    test_run_index: int,
    test_run_total: int,
    trial_state_path: Path,
    params_path: Path,
    run_slot: int,
    gen_index: int,
    trial_index: int,
    run_index: int,
    env: dict,
) -> subprocess.Popen:
    """Launch one ``runSofa`` instance for a single run of a trial.

    ``env`` is the base scene environment already merged with any per-trial
    prepare-hook env; this adds the per-run identity keys (see
    :mod:`sofaopt.core.envkeys`). Returns the started process.
    """
    trial_env = env.copy()
    trial_env[envkeys.TRIAL_STATE_PATH] = str(trial_state_path)
    trial_env[envkeys.PARAMS_PATH] = str(params_path)
    trial_env[envkeys.RUN_SLOT] = str(run_slot)
    trial_env[envkeys.GEN] = str(gen_index)
    trial_env[envkeys.TRIAL] = str(trial_index)
    trial_env[envkeys.RUN] = str(run_index)
    trial_env[envkeys.TEST_NAME] = test_name
    trial_env[envkeys.TEST_RUN_INDEX] = str(test_run_index)
    trial_env[envkeys.TEST_RUN_TOTAL] = str(test_run_total)

    cmd = [str(project.runsofa_exe)]
    for plugin in project.sofa_plugins:
        cmd += ["-l", plugin]
    cmd += ["-g", project.gui_mode, str(scene_file)]

    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
        if hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
            creation_flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS

    # One log per run slot next to trial_state.json, so early crashes are
    # diagnosable (overwritten on each relaunch).
    log_path = trial_state_path.parent / f"sofa_run{run_slot}.log"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        env=trial_env,
        cwd=str(project.work_dir),
        creationflags=creation_flags,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()
    attach_process_to_sofa_job(proc)
    return proc


def active_sofa_process_count(processes: list[tuple]) -> int:
    """Count running SOFA children across the generation's launched runs.

    Each entry's third element is its ``runs`` list of ``(Popen, path, slot)``.
    """
    active = 0
    for entry in processes:
        if len(entry) < 3:
            continue
        for p, _, _ in entry[2]:
            if p.poll() is None:
                active += 1
    return active


def wait_for_slot(
    processes: list[tuple], limit: int, gen_index: int, trial_index: int
) -> None:
    """Block until the active SOFA process count drops below ``limit``."""
    if limit <= 0:
        return
    warned = False
    while active_sofa_process_count(processes) >= limit:
        if not warned:
            print(
                f"[throttle] Gen {gen_index:04d} Trial {trial_index:02d} "
                f"waiting for active SOFA < {limit}"
            )
            warned = True
        time.sleep(0.2)
