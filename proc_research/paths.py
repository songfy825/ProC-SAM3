"""Machine-specific locations are configuration, never model identity."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def runtime_root():
    return Path(os.environ.get('P01_RUNTIME',ROOT/'.p01-runtime')).expanduser().resolve()


def checkpoint_path():
    explicit=os.environ.get('P01_CHECKPOINT')
    if explicit:return Path(explicit).expanduser().resolve()
    local=ROOT/'weights/sam3.pt'
    if local.is_file():return local
    legacy=Path('/home/fengyi/sam3.pt')
    return legacy if legacy.is_file() else local


def interpreter_command():
    import sys
    if os.access(sys.executable,os.X_OK):return [sys.executable]
    return ['/lib64/ld-linux-x86-64.so.2',sys.executable]
