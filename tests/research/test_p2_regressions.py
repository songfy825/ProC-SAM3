"""CPU-only regressions for Astra F1/F2; synthetic data is never GPU evidence."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from test_contracts import sample
from proc_research.cache import RESERVE_BYTES, digest, sha256, write_json, write_sample
from proc_research.calibration.run import (
    ROOT, METHODS, code_identity, decode_stage, finish, fit_stage, load_config,
)
from proc_research.evaluate import evaluate_predictions
from proc_research.launch import preserve_incomplete_cpu
from proc_research.protocol import create_run
from proc_research.report import build_report


class P2Regressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cfg = load_config(ROOT / 'research/p01/config.json')
        ident = dict(config=cfg, config_sha256=digest(cfg), manifest_sha256='b'*64,
                     method_sha256=sha256(ROOT/'METHOD.md'), code_sha256=code_identity(),
                     weight_sha256='c'*64, pool_sha256={'LoveDA':'d'*64, 'iSAID':'e'*64},
                     evidence_kind='synthetic_cpu')
        labels = []
        for split in ('calibration', 'pilot'):
            run = create_run(cls.root, split, dict(ident, stage=split))
            entries = []
            for dataset in ('LoveDA', 'iSAID'):
                for i in range(2):
                    for view in (1008, 504):
                        s = sample(scene=f'{split}-s{i}', image=f'{split}-i{i}', view=view)
                        s.dataset = dataset
                        s.scores[2, :, i:i+2] = .95
                        s.prompt_scores = s.scores.clone()
                        s.prediction = s.scores.argmax(0)
                        name = f'cache/{dataset}-{i}-{view}'
                        receipt = write_sample(run/name, s)
                        entries.append(dict(dataset=dataset, image_id=s.image_id,
                            scene_id=s.scene_id, view=view, split=split, path=name, receipt=receipt))
                    if split == 'pilot':
                        y = s.prediction.numpy().astype('uint8') + (1 if dataset=='LoveDA' else 0)
                        path = cls.root/f'{dataset}-{i}-synthetic-label.png'
                        Image.fromarray(y).save(path)
                        labels.append(dict(dataset=dataset, image_id=s.image_id,
                                           path=str(path), sha256=sha256(path)))
            write_json(run/'cache_index.json', {'entries':entries})
            finish(run, {'kind':'synthetic_cpu_fixture'})
        for name in ('bank-a', 'bank-b'):
            fit_stage(SimpleNamespace(cache_run=str(cls.root/'calibration'),
                                      output=str(cls.root), run_id=name))
        decode_stage(cls.decode_args(cls.root, 'pred-a'))
        cls.labels = cls.root/'labels.jsonl'
        cls.labels.write_text(''.join(json.dumps(row)+'\n' for row in labels))
        evaluate_predictions(cls.root/'pred-a/prediction_index.json', cls.labels, cls.root/'eval')
        cls.original_cache_receipt = sha256(cls.root/'pilot/complete.json')
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    @classmethod
    def decode_args(cls, output, run_id='predictions'):
        return SimpleNamespace(cache_run=str(cls.root/'pilot'), bank_run=str(cls.root/'bank-a'),
                               output=str(output), run_id=run_id)

    def decode(self):
        decode_stage(self.decode_args(self.out))
        return self.out/'predictions'

    def assert_no_predictions(self):
        self.assertEqual(list(self.out.rglob('*.npy')), [])
        self.assertFalse((self.out/'predictions/complete.json').exists())
        self.assertEqual(sha256(self.root/'pilot/complete.json'), self.original_cache_receipt)

    def test_decode_rejects_below_reserve_before_writing(self):
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=RESERVE_BYTES-1)):
            with self.assertRaises(OSError):
                self.decode()
        self.assert_no_predictions()

    def test_decode_budgets_all_images_before_first_prediction(self):
        # One image fits, but four images * eleven int64 8x12 maps do not.
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=RESERVE_BYTES+11000)):
            with self.assertRaises(OSError):
                self.decode()
        self.assert_no_predictions()

    def test_decode_budget_includes_npy_headers(self):
        # Exact raw array payload (44 * 8 * 12 * 8) leaves no room for NPY headers.
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=RESERVE_BYTES+33792)):
            with self.assertRaises(OSError):
                self.decode()
        self.assert_no_predictions()

    def test_decode_sufficient_space_preserves_all_predictions(self):
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=RESERVE_BYTES+16*1024**2)):
            run = self.decode()
        self.assertTrue((run/'complete.json').is_file())
        actual = sorted((run/'predictions').rglob('*.npy'))
        self.assertEqual(len(actual), 44)
        for path in actual:
            expected = self.root/'pred-a/predictions'/path.relative_to(run/'predictions')
            self.assertEqual(path.read_bytes(), expected.read_bytes())
        self.assertFalse(torch.cuda.is_initialized())

    def test_space_loss_between_images_stops_and_retains_failed_output(self):
        target = self.out/'predictions'
        def available(_):
            written = len(list((target/'predictions').rglob('*.npy')))
            return SimpleNamespace(free=RESERVE_BYTES-1 if written>=11 else RESERVE_BYTES+16*1024**2)
        with patch('shutil.disk_usage', side_effect=available):
            with self.assertRaises(OSError):
                self.decode()
        self.assertEqual(len(list((target/'predictions').rglob('*.npy'))), 11)
        self.assertTrue((target/'manifest.json').is_file())
        self.assertFalse((target/'complete.json').exists())
        # Exercise the launcher's actual CPU recovery helper, then its decode entry.
        audit = self.out/'recovery'
        audit.mkdir()
        saved = preserve_incomplete_cpu({'stage':'decode','output':str(target)}, audit, True)
        with patch('shutil.disk_usage', return_value=SimpleNamespace(free=RESERVE_BYTES+16*1024**2)):
            self.decode()
        self.assertEqual(len(list((saved/'predictions').rglob('*.npy'))), 11)
        self.assertEqual(len(list((target/'predictions').rglob('*.npy'))), 44)
        self.assertTrue((target/'complete.json').exists())
        self.assertEqual(sha256(self.root/'pilot/complete.json'), self.original_cache_receipt)

    def test_report_accepts_exact_prediction_bank(self):
        report = build_report(self.root/'pred-a', self.root/'eval', self.root/'bank-a', self.out/'report')
        self.assertEqual(report['research_status'], 'PENDING_SYNTHETIC_ONLY')
        provenance = json.loads((self.out/'report/provenance.json').read_text())
        self.assertEqual(provenance['bank_receipt_sha256'], sha256(self.root/'bank-a/complete.json'))

    def test_report_rejects_different_same_experiment_bank_before_output(self):
        self.assertNotEqual(sha256(self.root/'bank-a/complete.json'), sha256(self.root/'bank-b/complete.json'))
        with self.assertRaises(ValueError):
            build_report(self.root/'pred-a', self.root/'eval', self.root/'bank-b', self.out/'report')
        self.assertFalse((self.out/'report').exists())

    def clone_with_identity(self, source, name, updates=None, remove=(), kind=None):
        """Reseal only disposable synthetic fixtures to isolate semantic checks."""
        target = self.out/name
        shutil.copytree(source, target)
        receipt = json.loads((target/'complete.json').read_text())
        manifest = json.loads((target/'manifest.json').read_text())
        manifest['identity'].update(updates or {})
        for key in remove:
            manifest['identity'].pop(key, None)
        manifest['identity_sha256'] = digest(manifest['identity'])
        write_json(target/'manifest.json', manifest)
        (target/'complete.json').unlink()
        finish(target, {'kind':kind or receipt['kind']})
        return target

    def test_report_rejects_missing_prediction_bank_binding(self):
        pred = self.clone_with_identity(self.root/'pred-a', 'missing', remove=('bank_receipt_sha256',))
        with self.assertRaises(ValueError):
            build_report(pred, self.root/'eval', self.root/'bank-a', self.out/'report')
        self.assertFalse((self.out/'report').exists())

    def test_report_rejects_wrong_bank_stage_and_kind_even_if_hash_matches(self):
        for name, stage, kind in (('stage','pilot','calibration_banks'), ('kind','fit','predictions')):
            with self.subTest(name=name):
                bank = self.clone_with_identity(self.root/'bank-a', name+'-bank', {'stage':stage}, kind=kind)
                pred = self.clone_with_identity(self.root/'pred-a', name+'-pred',
                                               {'bank_receipt_sha256':sha256(bank/'complete.json')})
                output = self.out/(name+'-report')
                with self.assertRaises(ValueError):
                    build_report(pred, self.root/'eval', bank, output)
                self.assertFalse(output.exists())

    def test_report_confirmation_must_match_frozen_bank(self):
        for name, frozen in (('valid',sha256(self.root/'bank-a/complete.json')), ('invalid','f'*64)):
            pred = self.clone_with_identity(self.root/'pred-a', name+'-confirmation',
                {'evaluated_split':'confirmation', 'frozen_bank_receipt_sha256':frozen})
            output = self.out/(name+'-report')
            if name == 'valid':
                report = build_report(pred, self.root/'eval', self.root/'bank-a', output)
                self.assertEqual(report['research_status'], 'PENDING_SYNTHETIC_ONLY')
            else:
                with self.assertRaises(ValueError):
                    build_report(pred, self.root/'eval', self.root/'bank-a', output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
