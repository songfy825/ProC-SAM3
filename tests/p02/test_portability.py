"""Exercise P02's path and interpreter contracts without models or hardware probes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from proc_research.correction import runtime


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATHS = {
    'HF_HOME': 'cache/hf', 'TORCH_HOME': 'cache/torch',
    'XDG_CACHE_HOME': 'cache', 'TMPDIR': 'tmp',
    'TRITON_CACHE_DIR': 'cache/triton', 'MPLCONFIGDIR': 'cache/matplotlib',
}


class PortabilityContracts(unittest.TestCase):
    def setUp(self):
        # Ordinary fixtures stay under the caller's audit destination. Native
        # symlink fixtures use explicit disposable /tmp directories below.
        base = Path(os.environ.get('P02_RUNTIME', ROOT/'.p02-runtime')).expanduser().resolve()
        temporary = base/'compat/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary, prefix='portability-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_default_runtime_is_relative_to_checkout_not_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            with patch.dict(os.environ):
                os.environ.pop('P02_RUNTIME', None)
                self.assertEqual(runtime.runtime_root(), (ROOT/'.p02-runtime').resolve())
        finally:
            os.chdir(previous)

    def test_custom_runtime_overrides_inherited_cpu_and_cache_settings(self):
        selected = self.directory/'runtime with spaces'
        inherited = self.directory/'foreign runtime'
        hostile = {'P02_RUNTIME': str(selected/'..'/selected.name),
                   'P01_RUNTIME': str(inherited), 'P01_ENABLE_GPU': '1',
                   'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '0',
                   'TRANSFORMERS_OFFLINE': '0', 'HF_HUB_DISABLE_TELEMETRY': '0'}
        hostile.update({key: str(inherited/key) for key in CACHE_PATHS})
        with patch.dict(os.environ, hostile):
            runtime.force_environment()
            self.assertEqual(runtime.runtime_root(), selected.resolve())
            self.assertEqual(os.environ['P01_RUNTIME'], str(selected/'compat'))
            self.assertEqual(os.environ['P01_ENABLE_GPU'], '0')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
                self.assertEqual(os.environ[key], '1', key)
            for key, suffix in CACHE_PATHS.items():
                self.assertEqual(Path(os.environ[key]), selected/'compat'/suffix, key)
            self.assertTrue((selected/'compat/tmp').is_dir())
            self.assertFalse(inherited.exists())

    def test_relative_runtime_is_rejected_before_directory_creation(self):
        before = set(self.directory.rglob('*'))
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            with patch.dict(os.environ, {'P02_RUNTIME': 'must-not-exist/runtime'}):
                with self.assertRaisesRegex(ValueError, 'absolute'):
                    runtime.force_environment()
        finally:
            os.chdir(previous)
        self.assertEqual(set(self.directory.rglob('*')), before)

    def test_output_subdirectory_is_created_but_existing_output_is_rejected(self):
        selected = self.directory/'runtime'
        with patch.dict(os.environ, {'P02_RUNTIME': str(selected)}):
            output = runtime.output_directory(selected/'nested'/'..'/'result')
            self.assertEqual(output, selected/'result')
            marker = output/'keep.txt'
            marker.write_text('do not overwrite')
            with self.assertRaises(FileExistsError):
                runtime.output_directory(output)
            self.assertEqual(marker.read_text(), 'do not overwrite')

    def test_output_must_be_a_child_of_runtime_root(self):
        selected = self.directory/'runtime'
        with patch.dict(os.environ, {'P02_RUNTIME': str(selected)}):
            with self.assertRaises(ValueError):
                runtime.output_directory(selected)
        self.assertFalse(selected.exists())

    def test_output_expands_tilde_and_preserves_containment(self):
        selected = self.directory/'runtime'
        configured = '~/' + os.path.relpath(selected, Path.home())
        with patch.dict(os.environ, {'P02_RUNTIME': configured}):
            output = runtime.output_directory(configured+'/audit/result')
            self.assertEqual(output, selected/'audit/result')
            self.assertTrue(output.is_dir())
            with self.assertRaises(ValueError):
                runtime.output_directory(configured+'/../escape')
        self.assertFalse((self.directory/'escape').exists())

    def test_output_rejects_parent_and_sibling_escape(self):
        selected = self.directory/'runtime'
        selected.mkdir()
        escapes = [selected/'..'/'escape', self.directory/'runtime-sibling'/'escape']
        with patch.dict(os.environ, {'P02_RUNTIME': str(selected)}):
            for output in escapes:
                with self.subTest(output=str(output)):
                    with self.assertRaises(ValueError):
                        runtime.output_directory(output)
                    self.assertFalse(output.exists())
        self.assertFalse((self.directory/'runtime-sibling').exists())

    def test_output_rejects_symlink_escape(self):
        # The shared checkout filesystem does not support creating symlinks.
        # Only this disposable fixture uses /tmp; audit logs stay in P02_RUNTIME.
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='p02-symlink-') as temporary:
            directory = Path(temporary)
            selected = directory/'runtime'
            selected.mkdir()
            outside = directory/'outside'
            outside.mkdir()
            (selected/'link').symlink_to(outside, target_is_directory=True)
            with patch.dict(os.environ, {'P02_RUNTIME': str(selected)}):
                with self.assertRaises(ValueError):
                    runtime.output_directory(selected/'link'/'escape')
            self.assertEqual(list(outside.iterdir()), [])

    def relocated_checkout(self, destination=None):
        destination = destination or self.directory/'relocated checkout with spaces'
        files = ('scripts/p01/python.sh', 'scripts/p02/python.sh',
                 'scripts/p02/analyze.py', 'proc_research/__init__.py',
                 'proc_research/correction/__init__.py',
                 'proc_research/correction/runtime.py')
        for name in files:
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        return destination

    def interpreter_link(self, prefix):
        interpreter = os.environ.get('P01_PYTHON') or sys.executable
        target = prefix/'bin/python'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(interpreter)
        return target

    def launch(self, checkout, arguments, **overrides):
        environment = os.environ.copy()
        # P01_PYTHON can legitimately be the original interpreter when Python is
        # started via the ELF loader on a noexec filesystem.
        interpreter = os.environ.get('P01_PYTHON') or sys.executable
        environment.update({'P02_PYTHON': interpreter,
                            'P02_RUNTIME': str(self.directory/'runtime with spaces'),
                            'P01_RUNTIME': str(self.directory/'foreign runtime'),
                            'P01_ENABLE_GPU': '1', 'CUDA_VISIBLE_DEVICES': '0',
                            'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'})
        for key, value in overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        return subprocess.run(['bash', str(checkout/'scripts/p02/python.sh'), *arguments],
                              cwd=self.directory, env=environment, capture_output=True,
                              text=True, timeout=90)

    def test_relocated_help_and_guard_use_custom_runtime(self):
        checkout = self.relocated_checkout()
        result = self.launch(checkout, [str(checkout/'scripts/p02/analyze.py'), '--help'])
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('--manifest', result.stdout)
        # The policy is installed before torch is imported, exactly as in the
        # actual CLI. The tested CUDA calls are denials, never hardware probes.
        code = """
import json, os
from proc_research.correction.runtime import install_cpu_guard, runtime_root
policy = install_cpu_guard()
import torch
blocked = []
for name in ('init', 'device_count'):
    try:
        getattr(torch.cuda, name)()
    except RuntimeError as error:
        blocked.append('P02 CPU' in str(error))
keys = ('P01_RUNTIME', 'P01_ENABLE_GPU', 'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE',
        'TRANSFORMERS_OFFLINE', 'P01_PYTHON', 'P02_PYTHON', 'HF_HOME',
        'TORCH_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'TRITON_CACHE_DIR', 'MPLCONFIGDIR')
print(json.dumps({'root': str(runtime_root()), 'env': {key: os.environ[key] for key in keys},
                  'policy': policy, 'blocked': blocked,
                  'available': torch.cuda.is_available()}))
"""
        result = self.launch(checkout, ['-c', code])
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        evidence = json.loads(result.stdout)
        selected = self.directory/'runtime with spaces'
        self.assertEqual(evidence['root'], str(selected))
        environment = evidence['env']
        self.assertEqual(environment['P01_RUNTIME'], str(selected/'compat'))
        self.assertEqual(environment['P01_ENABLE_GPU'], '0')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
        self.assertEqual(environment['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(environment['P01_PYTHON'], environment['P02_PYTHON'])
        for key, suffix in CACHE_PATHS.items():
            self.assertEqual(environment[key], str(selected/'compat'/suffix), key)
        self.assertEqual(evidence['blocked'], [True, True])
        self.assertFalse(evidence['available'])
        self.assertEqual(evidence['policy']['hardware_probes'], 0)
        self.assertEqual(evidence['policy']['model_inference_calls'], 0)
        self.assertEqual(evidence['policy']['api_calls'], 0)
        self.assertFalse((self.directory/'foreign runtime').exists())
        self.assertFalse((checkout/'.p02-runtime').exists())

    def test_launcher_rejects_relative_runtime_before_mkdir(self):
        checkout = self.relocated_checkout()
        result = self.launch(checkout, ['-c', "print('INTERPRETER_STARTED')"],
                             P02_RUNTIME='must-not-exist/runtime')
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('P02_RUNTIME', result.stderr)
        self.assertIn('absolute', result.stderr)
        self.assertNotIn('INTERPRETER_STARTED', result.stdout)
        self.assertFalse((self.directory/'must-not-exist').exists())
        self.assertFalse((self.directory/'foreign runtime').exists())

    def test_relocated_launcher_defaults_runtime_to_its_checkout(self):
        checkout = self.relocated_checkout()
        result = self.launch(checkout, ['-c',
            'from proc_research.correction.runtime import runtime_root; print(runtime_root())'],
            P02_RUNTIME=None)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(Path(result.stdout.strip()), checkout/'.p02-runtime')
        self.assertTrue((checkout/'.p02-runtime/compat/tmp').is_dir())
        self.assertFalse((self.directory/'foreign runtime').exists())

    def test_invalid_explicit_interpreter_never_falls_back(self):
        checkout = self.relocated_checkout()
        result = self.launch(checkout, ['-c', "print('INTERPRETER_STARTED')"],
                             P02_PYTHON=str(self.directory/'missing interpreter'))
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('P02_PYTHON', result.stderr)
        self.assertNotIn('INTERPRETER_STARTED', result.stdout)
        self.assertFalse((self.directory/'runtime with spaces').exists())

    def test_active_virtual_environment_precedes_conda(self):
        checkout = self.relocated_checkout()
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='p02-active-') as temporary:
            directory = Path(temporary)
            virtual = self.interpreter_link(directory/'virtual environment')
            conda = self.interpreter_link(directory/'conda environment')
            result = self.launch(checkout, ['-c', "import os; print(os.environ['P01_PYTHON'])"],
                                 P02_PYTHON=None, P01_PYTHON='/must-not-inherit/python',
                                 VIRTUAL_ENV=str(virtual.parent.parent),
                                 CONDA_PREFIX=str(conda.parent.parent))
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(result.stdout.strip(), str(virtual))

    def test_active_conda_environment_is_used_without_virtual_environment(self):
        checkout = self.relocated_checkout()
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='p02-active-') as temporary:
            conda = self.interpreter_link(Path(temporary)/'conda environment')
            result = self.launch(checkout, ['-c', "import os; print(os.environ['P01_PYTHON'])"],
                                 P02_PYTHON=None, P01_PYTHON='/must-not-inherit/python',
                                 VIRTUAL_ENV=None, CONDA_PREFIX=str(conda.parent.parent))
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(result.stdout.strip(), str(conda))

    def test_local_virtual_environment_precedes_local_p01_environment(self):
        # The two local environments need distinct symlinks to an existing
        # interpreter, so this tiny copied checkout also uses the /tmp filesystem.
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='p02-local-env-') as temporary:
            checkout = self.relocated_checkout(Path(temporary)/'checkout with spaces')
            virtual = self.interpreter_link(checkout/'.venv')
            p01 = self.interpreter_link(checkout/'.p01-env')
            for selected in (virtual, p01):
                with self.subTest(interpreter=str(selected)):
                    result = self.launch(checkout,
                        ['-c', "import os; print(os.environ['P01_PYTHON'])"],
                        P02_PYTHON=None, P01_PYTHON='/must-not-inherit/python',
                        VIRTUAL_ENV=None, CONDA_PREFIX=None)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(result.stdout.strip(), str(selected))
                selected.unlink()

    def test_path_python_is_used_without_local_or_active_environment(self):
        checkout = self.relocated_checkout()
        expected = shutil.which('python3')
        self.assertIsNotNone(expected, 'the test runner must provide python3 on PATH')
        result = self.launch(checkout, ['-c', "import os; print(os.environ['P01_PYTHON'])"],
                             P02_PYTHON=None, P01_PYTHON='/must-not-inherit/python',
                             VIRTUAL_ENV=None, CONDA_PREFIX=None)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(Path(result.stdout.strip()).resolve(), Path(expected).resolve())


if __name__ == '__main__':
    unittest.main()
