"""Exercise released configuration loading after a case-sensitive relocation."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from proc_research import baseline
from proc_research.prepare import prompt_inventory


ROOT = Path(__file__).resolve().parents[2]


class ConfigPaths(unittest.TestCase):
    def setUp(self):
        # The source checkout can be case-insensitive. Linux /tmp must expose
        # wrong-case lookups instead of inheriting the checkout's filesystem.
        temporary = tempfile.TemporaryDirectory(prefix='p01-config-case-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.relocated = Path(temporary.name)
        for relative in (
            'configs/cfg_iSAID.py',
            'configs/cls_iSAID.txt',
            'prompt_pools/cfg_iSAID_expanded_prompt_pool_hybrid_strict.pkl',
            'configs/cfg_loveda.py',
            'configs/cls_loveda.txt',
            'prompt_pools/cfg_loveda_expanded_prompt_pool_hybrid_strict.pkl',
        ):
            target = self.relocated / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.assertFalse(
            (self.relocated / 'configs/cfg_isaid.py').exists(),
            'This Linux relocation regression requires case-sensitive /tmp.',
        )

    def check_relocated_dataset(self, dataset, class_file, pool_file, classes, prompts):
        # Relocate only the asset root; run the real loader, frozen pool hash
        # check, class-file parsing and vocabulary expansion without a model.
        with patch.object(baseline, 'ROOT', self.relocated):
            try:
                options = baseline.model_options(dataset)
                inventory = prompt_inventory(dataset)
            except FileNotFoundError as error:
                self.fail(f'{dataset} must load the tracked asset filenames: {error}')
        self.assertEqual(Path(options['classname_path']), self.relocated / class_file)
        self.assertEqual(Path(options['expanded_prompt_pool_path']), self.relocated / pool_file)
        self.assertEqual(inventory['num_classes'], classes)
        self.assertEqual(inventory['num_prompts'], prompts)
        self.assertEqual(set(inventory['prompt_to_class']), set(range(classes)))
        self.assertEqual(options['prob_thd'], 0.1)
        self.assertEqual(options['confidence_threshold'], 0.4)
        self.assertEqual(options['semantic_instance_fusion'], 'pgrf')
        self.assertEqual(options['class_aggregation'], 'max')
        self.assertEqual(options['inference_dtype'], 'fp32')
        self.assertEqual(options['image_encoder_dtype'], 'bf16')
        self.assertEqual(options['image_batch_size'], 1)
        self.assertFalse(options['compile_model'])

    def test_isaid_loads_released_assets_on_case_sensitive_filesystem(self):
        self.check_relocated_dataset(
            'iSAID', 'configs/cls_iSAID.txt',
            'prompt_pools/cfg_iSAID_expanded_prompt_pool_hybrid_strict.pkl',
            classes=16, prompts=24,
        )

    def test_loveda_loads_released_assets_after_relocation(self):
        self.check_relocated_dataset(
            'LoveDA', 'configs/cls_loveda.txt',
            'prompt_pools/cfg_loveda_expanded_prompt_pool_hybrid_strict.pkl',
            classes=7, prompts=28,
        )


if __name__ == '__main__':
    unittest.main()
