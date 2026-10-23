"""
Step 22 -- record the software and hardware environment quoted in Section 5
(Environment paragraph). The hardware queries use PowerShell, so this script
runs on Windows only.

Usage
-----
    python 22_environment.py --out ../results/environment.json
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-c", cmd],
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="../results/environment.json")
    args = ap.parse_args()
    import capstone, lightgbm, numpy, pefile, powerlaw, scipy, sklearn, xgboost
    env = {
        "cpu": ps("(Get-CimInstance Win32_Processor).Name"),
        "ram_gb": float(ps("[math]::Round((Get-CimInstance Win32_ComputerSystem)"
                           ".TotalPhysicalMemory/1GB,1)")),
        "os": ps("(Get-CimInstance Win32_OperatingSystem).Caption"),
        "os_build": platform.version(),
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__, "powerlaw": powerlaw.__version__,
        "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "capstone": capstone.__version__, "pefile": pefile.__version__,
    }
    Path(args.out).write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
