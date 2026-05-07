#!/usr/bin/env python3
"""跨平台打包脚本 — 打包 CLI 工具 (etf_signal / etf_backtest)

用法:
  python build_exe.py

输出在 dist/ 目录。
Streamlit 可视化界面通过 run_app.py 启动，无需打包。
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"


def clean():
    for d in ["build", "__pycache__"]:
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p)
    for spec in ROOT.glob("*.spec"):
        spec.unlink()
    if DIST.exists():
        shutil.rmtree(DIST)


def pack(script: str, name: str) -> bool:
    spec = script
    data = f"etf_config.json{';' if sys.platform == 'win32' else ':'}."
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--console", f"--name={name}",
        "--add-data", data,
        str(ROOT / spec),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    clean()
    DIST.mkdir(parents=True, exist_ok=True)

    ok1 = pack("etf_signal.py", "etf_signal")
    ok2 = pack("etf_backtest.py", "etf_backtest")

    if ok1 and ok2:
        exes = list(DIST.glob("*"))
        print(f"\n打包完成，共 {len(exes)} 个文件: {DIST}")
        print("  streamlit run etf_app.py  → 可视化界面（无需打包）")
    else:
        print("打包失败")


if __name__ == "__main__":
    main()
