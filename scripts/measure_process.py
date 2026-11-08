"""Run a benchmark with a time limit and sampled Windows process memory.
Example: python -I scripts/measure_process.py --report results/memory.json --log results/run.log -- target/release/model_bench.exe ...
Working set/private commit do not include all driver-managed shared GPU memory.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time

class Counters(ctypes.Structure):
    _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [(name,ctypes.c_size_t) for name in
        ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
         "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--log",required=True,type=Path)
    parser.add_argument("--timeout",type=float,default=600)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command: parser.error("missing command")
    args.log.parent.mkdir(parents=True,exist_ok=True)
    peak_working_set=peak_private=peak_commit=0
    started=time.monotonic()
    timed_out=False
    with args.log.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
        handle=None
        if os.name=="nt":
            kernel=ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            psapi=ctypes.WinDLL("psapi",use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            handle=kernel.OpenProcess(0x410,False,process.pid)
        try:
            while process.poll() is None:
                if handle:
                    counters=Counters(); counters.cb=ctypes.sizeof(counters)
                    if psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):
                        peak_working_set=max(peak_working_set,counters.PeakWorkingSetSize)
                        peak_private=max(peak_private,counters.PrivateUsage)
                        peak_commit=max(peak_commit,counters.PeakPagefileUsage)
                if time.monotonic()-started>args.timeout:
                    timed_out=True; process.kill(); break
                time.sleep(0.2)
            returncode=process.wait(timeout=10)
        finally:
            if handle: kernel.CloseHandle(handle)
            if process.poll() is None: process.kill(); process.wait()
    report={"command":command,"exit_code":returncode,"timed_out":timed_out,"elapsed_s":time.monotonic()-started,
            "peak_working_set_bytes":peak_working_set,"sampled_peak_private_bytes":peak_private,"peak_commit_bytes":peak_commit,
            "limitation":"Process counters only; driver-managed shared GPU memory and system RAM pressure require separate measurement."}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(1 if timed_out or returncode else 0)

if __name__=="__main__":main()
