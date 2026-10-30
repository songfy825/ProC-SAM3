"""Offline CPU process policy, installed before optional library imports."""
import os
from pathlib import Path

RUNTIME = Path('/home/fengyi/CodeHome/Proc/runtime/p02')
_POLICY = {'hardware_probes': 0, 'model_inference_calls': 0, 'api_calls': 0,
           'scope': 'policy for this guarded process only; not a historical session audit',
           'availability_answer': 'constant False by CPU policy, no driver query'}


def force_environment():
    fixed = {'P01_ENABLE_GPU': '0', 'CUDA_VISIBLE_DEVICES': '',
             'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
             'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONDONTWRITEBYTECODE': '1',
             'P01_RUNTIME': str(RUNTIME/'compat'), 'PYTHONNOUSERSITE': '1',
             'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    os.environ.update(fixed)
    for key, suffix in {'HF_HOME': 'cache/hf', 'TORCH_HOME': 'cache/torch',
                        'XDG_CACHE_HOME': 'cache', 'TMPDIR': 'tmp',
                        'TRITON_CACHE_DIR': 'cache/triton', 'MPLCONFIGDIR': 'cache/matplotlib'}.items():
        os.environ[key] = str(RUNTIME/'compat'/suffix)
    (RUNTIME/'compat/tmp').mkdir(parents=True, exist_ok=True)


def denied(*args, **kwargs):
    raise RuntimeError('P02 CPU policy forbids GPU initialization/probes and network access')


def _denial(name):
    def blocked(*args, **kwargs):
        return denied(*args, **kwargs)
    blocked.__name__ = name
    return blocked


def install_cpu_guard():
    force_environment()
    import torch
    # Old CPU-only transform tests import CUDA autocast decorators. Give their
    # availability check a fixed policy answer without touching the driver.
    torch.cuda.is_available = lambda: False
    torch.cuda.is_initialized = lambda: False
    for name in ('init', '_lazy_init', 'device_count', 'current_device', 'synchronize',
                 'get_device_properties', 'get_device_name', 'set_device', 'mem_get_info'):
        setattr(torch.cuda, name, _denial(name))
    import socket
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    return dict(_POLICY)


def output_directory(value):
    path = Path(value).resolve()
    if not path.is_relative_to(RUNTIME.resolve()):
        raise ValueError('P02 artifacts must be under /home/fengyi/CodeHome/Proc/runtime/p02')
    path.mkdir(parents=True, exist_ok=False)
    return path
