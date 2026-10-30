"""P02 process contract; no hardware detection is needed to test denial."""
import os
from pathlib import Path
import unittest


class CPUContract(unittest.TestCase):
    def test_guard_forces_cpu_offline_and_rejects_gpu_initialization(self):
        from proc_research.correction.runtime import install_cpu_guard
        runtime = Path(os.environ.get('P02_RUNTIME', str(Path(__file__).resolve().parents[2]/'.p02-runtime'))).expanduser().resolve()
        os.environ['P01_ENABLE_GPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        evidence = install_cpu_guard()
        import torch
        self.assertEqual(os.environ['P01_ENABLE_GPU'], '0')
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['P01_RUNTIME'], str(runtime/'compat'))
        self.assertFalse(torch.cuda.is_available())  # policy answer, not hardware probe
        with self.assertRaisesRegex(RuntimeError, 'P02 CPU'):
            torch.cuda.init()
        with self.assertRaisesRegex(RuntimeError, 'P02 CPU'):
            torch.cuda.device_count()
        self.assertEqual(evidence['hardware_probes'], 0)

    def test_entrypoints_exist(self):
        root = Path(__file__).resolve().parents[2]
        for name in ('verify.py', 'analyze.py', 'python.sh'):
            self.assertTrue((root/'scripts/p02'/name).is_file(), name)


if __name__ == '__main__':
    unittest.main()

class GuardCompatibility(unittest.TestCase):
    def test_torch_rule_registry_accepts_guard(self):
        from proc_research.correction.runtime import install_cpu_guard
        install_cpu_guard()
        import torch._dynamo.trace_rules as rules
        self.assertTrue(rules.get_torch_obj_rule_map())
