"""Synthetic provenance fixtures; none establish a real-model production history."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from proc_research.cache import SampleCache, digest, sha256, write_json, write_sample


def write_fixture(directory):
    """Create a SMALL explicitly synthetic offline input bundle, for CLI tests too."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    image = root / 'synthetic.png'
    Image.new('RGB', (8, 4), color=(10, 20, 30)).save(image)
    scores = torch.zeros((7, 4, 8), dtype=torch.float32)
    scores[0] = .51
    scores[1] = .50
    prompt = scores.clone()
    prompt[1, 0, 0] = .55
    scores = prompt.clone()
    provenance = {'image_sha256': sha256(image), 'weight_sha256':'b'*64,
                  'pool_sha256':'c'*64, 'config_sha256':'d'*64, 'code_sha256':'e'*64,
                  'prompts':[f'synthetic-{i}' for i in range(7)], 'kind':'synthetic'}
    sample = SampleCache('synthetic-001', '', 'LoveDA', 1008, scores, prompt,
                         torch.ones(7), torch.ones(7), torch.arange(7), torch.arange(7),
                         tuple(range(7)), scores.argmax(0),
                         {'kind':'full_resize','original_hw':[4,8],
                          'encoder_input':[1,3,1008,1008], 'feature_shapes':[[1,8,72,72]]}, provenance)
    receipt = write_sample(root/'cache'/'synthetic-001', sample)
    row = {'dataset':'LoveDA','image_id':sample.image_id,'split':'development',
           'image_path':str(image), 'image_sha256':sha256(image), 'height':4,'width':8,
           'source_scene_id':'', 'scene_evidence':''}
    development = root/'synthetic-development.jsonl'
    development.write_text(json.dumps(row)+'\n')
    index = root/'synthetic-index.json'
    write_json(index, {'kind':'synthetic','entries':[{'dataset':'LoveDA','image_id':sample.image_id,
                         'scene_id':'','view':1008,'path':'cache/synthetic-001','receipt':receipt}]})
    manifest = {'schema':1,'kind':'synthetic',
                'development':{'path':str(development),'sha256':sha256(development)},
                'production_index':{'path':str(index),'sha256':sha256(index)}}
    path = root/'synthetic-input.json'
    write_json(path, manifest)
    return path


def write_structural_audit_fixture(root, path):
    """Synthetic record graph for validator tests; NEVER a real production receipt."""
    root = Path(root)
    manifest = json.loads(path.read_text())
    config = {'synthetic_contract_fixture': True, 'views': [1008]}
    write_json(root/'producer-config.json', config)
    code_files = {'synthetic-test-only.py': '1'*64}
    write_json(root/'producer-code.json', {'files': code_files})
    ident = {'evidence_kind': 'real_model', 'config': config,
             'config_sha256': sha256(root/'producer-config.json'),
             'code_sha256': digest(code_files), 'weight_sha256': 'b'*64,
             'pool_sha256': {'LoveDA': 'c'*64}}
    write_json(root/'producer-manifest.json', {'status': 'PASS', 'identity': ident,
                                              'identity_sha256': digest(ident)})
    write_json(root/'producer-complete.json', {'status': 'PASS', 'kind': 'real_gpu_cache',
               'files': {'producer-manifest.json': sha256(root/'producer-manifest.json'),
                         'synthetic-index.json': sha256(root/'synthetic-index.json')}})
    write_json(root/'assets.json', {'weight': {'sha256': 'b'*64}})
    row = json.loads((root/'synthetic-development.jsonl').read_text())
    write_json(root/'history.json', {'development_sha256': sha256(root/'synthetic-development.jsonl'),
               'selection_basis': 'existing-development-list', 'rows': [dict(
                   dataset=row['dataset'], image_id=row['image_id'], image_sha256=row['image_sha256'],
                   development_eligible=True, prior_uses=['synthetic contract fixture'],
                   eligibility_evidence='Synthetic validator test, not a legal/real development receipt')]})
    (root/'producer.log').write_text('SYNTHETIC validator fixture. No inference has run.\n')
    names = {'production_manifest': 'producer-manifest.json', 'production_receipt': 'producer-complete.json',
             'production_log': 'producer.log', 'producer_config': 'producer-config.json',
             'producer_code_inventory': 'producer-code.json', 'asset_receipt': 'assets.json',
             'use_history': 'history.json'}
    for key, filename in names.items():
        manifest[key] = {'path': str(root/filename), 'sha256': sha256(root/filename)}
    meta = json.loads((root/'cache/synthetic-001/meta.json').read_text())
    receipt = json.loads((root/'cache/synthetic-001/receipt.json').read_text())
    audit = {'kind': 'independent_cache_audit', 'status': 'PASS',
             'auditor': 'SYNTHETIC TEST FIXTURE, NOT AN AUDITOR', 'recorded_at': 'synthetic',
             'producer_command': 'SYNTHETIC TEST ONLY; NO MODEL CALL',
             'bindings': {key: value['sha256'] for key, value in manifest.items() if isinstance(value, dict)},
             'exports': [{'dataset': 'LoveDA', 'image_id': 'synthetic-001', 'view': 1008,
                          'receipt': receipt, 'geometry': meta['geometry'], 'provenance': meta['provenance']}]}
    (root/'independent-synthetic').mkdir()
    audit_path = root/'independent-synthetic/audit.json'
    write_json(audit_path, audit)
    manifest['independent_receipt'] = {'path': str(audit_path), 'sha256': sha256(audit_path)}
    # Top-level remains synthetic: these records must never be presented as real.
    write_json(path, manifest)
    return manifest


class ProvenanceContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        from proc_research.correction.provenance import load_manifest
        return load_manifest(self.path)

    def test_synthetic_is_explicit_and_scene_unknown(self):
        result = self.load()
        self.assertEqual(result['kind'], 'synthetic')
        self.assertEqual(len(result['samples']), 1)
        self.assertEqual(result['evidence']['real_action_space'], 'PENDING')
        self.assertFalse(result['evidence']['scene_independence_established'])

    def test_corrupt_local_receipt_fails_against_parent(self):
        write_json(self.root/'cache/synthetic-001/receipt.json', {'meta_sha256':'a'*64,'arrays_sha256':'b'*64})
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.load()

    def test_parent_hash_corruption_fails(self):
        with (self.root/'synthetic-index.json').open('a') as f:
            f.write(' ')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.load()

    def test_missing_cache_pending_never_success(self):
        from proc_research.correction.provenance import PendingEvidence
        (self.root/'cache/synthetic-001/arrays.npz').unlink()
        with self.assertRaises(PendingEvidence):
            self.load()

    def test_missing_prompt_response_pending_after_repinning_parent(self):
        from proc_research.correction.provenance import PendingEvidence
        cache = self.root/'cache/synthetic-001'
        with np.load(cache/'arrays.npz', allow_pickle=False) as f:
            arrays = {name:f[name] for name in f.files if name != 'prompt_scores'}
        np.savez(cache/'arrays.npz', **arrays)
        self.repin_cache()
        with self.assertRaisesRegex(PendingEvidence, 'prompt_scores'):
            self.load()

    def repin_cache(self):
        cache = self.root/'cache/synthetic-001'
        meta = json.loads((cache/'meta.json').read_text())
        meta['arrays_sha256'] = sha256(cache/'arrays.npz')
        write_json(cache/'meta.json', meta)
        receipt = {'meta_sha256':sha256(cache/'meta.json'),'arrays_sha256':sha256(cache/'arrays.npz')}
        write_json(cache/'receipt.json', receipt)
        index = json.loads((self.root/'synthetic-index.json').read_text())
        index['entries'][0]['receipt'] = receipt
        write_json(self.root/'synthetic-index.json', index)
        manifest = json.loads(self.path.read_text())
        manifest['production_index']['sha256'] = sha256(self.root/'synthetic-index.json')
        write_json(self.path, manifest)

    def test_class_max_must_match_true_prompt_aggregation(self):
        cache = self.root/'cache/synthetic-001'
        with np.load(cache/'arrays.npz', allow_pickle=False) as f:
            arrays = {name:f[name] for name in f.files}
        arrays['scores'][1,0,0] = .99
        np.savez(cache/'arrays.npz', **arrays)
        self.repin_cache()
        with self.assertRaisesRegex(ValueError, 'aggregation'):
            self.load()

    def test_wrong_view_rejected(self):
        cache = self.root/'cache/synthetic-001'
        meta = json.loads((cache/'meta.json').read_text())
        meta['view'] = 504
        meta['geometry']['encoder_input'] = [1,3,504,504]
        write_json(cache/'meta.json', meta)
        self.repin_cache()
        with self.assertRaisesRegex(ValueError, '1008|view'):
            self.load()

    def test_real_flag_alone_is_not_real_provenance(self):
        from proc_research.correction.provenance import PendingEvidence
        value = json.loads(self.path.read_text())
        value['kind'] = 'real'
        value['real'] = True
        write_json(self.path, value)
        with self.assertRaises((PendingEvidence, ValueError)):
            self.load()

    def test_selection_first50_before_cache_availability(self):
        from proc_research.correction.provenance import select_development
        rows = [{'dataset':'LoveDA','image_id':f'{i:03d}','split':'development',
                 'image_path':f'/never-read/{i}.png','image_sha256':f'{i:064x}',
                 'height':2,'width':3} for i in range(60)]
        selected = select_development(rows[::-1])
        self.assertEqual([r['image_id'] for r in selected], [f'{i:03d}' for i in range(50)])

    def test_gt_fields_and_confirmation_are_rejected(self):
        from proc_research.correction.provenance import select_development
        row = {'dataset':'LoveDA','image_id':'1','split':'confirmation','image_path':'i',
               'image_sha256':'a'*64,'height':1,'width':2}
        with self.assertRaises(ValueError):
            select_development([row])
        row['split'] = 'development'
        row['gt_path'] = 'forbidden'
        with self.assertRaises(ValueError):
            select_development([row])

    def test_stream_defers_arrays_and_finishes_evidence(self):
        from proc_research.correction.provenance import load_manifest
        result = load_manifest(self.path, stream=True)
        self.assertEqual(result['evidence']['caches'], [])
        samples = list(result['samples'])
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(result['evidence']['caches']), 1)

    def test_stream_missing_selected_cache_raises_during_iteration(self):
        from proc_research.correction.provenance import load_manifest, PendingEvidence
        result = load_manifest(self.path, stream=True)
        (self.root/'cache/synthetic-001/arrays.npz').unlink()
        with self.assertRaises(PendingEvidence):
            list(result['samples'])

    def test_missing_independent_real_evidence_is_pending(self):
        from proc_research.correction.provenance import load_manifest, PendingEvidence
        manifest = json.loads(self.path.read_text())
        index_path = self.root/'synthetic-index.json'
        index = json.loads(index_path.read_text())
        index.pop('kind')
        write_json(index_path, index)
        manifest['kind'] = 'real'
        manifest['production_index']['sha256'] = sha256(index_path)
        write_json(self.path, manifest)
        with self.assertRaisesRegex(PendingEvidence, 'production_manifest'):
            load_manifest(self.path)

    def test_prompt_pool_and_class_mapping_is_pinned(self):
        from proc_research.correction.provenance import _validate_responses
        sample = self.load()['samples'][0]
        with self.assertRaisesRegex(ValueError, 'published prompt'):
            _validate_responses(sample, real=True)

    def test_published_metadata_does_not_import_model_or_query_cuda(self):
        import sys
        from unittest.mock import patch
        from proc_research.correction.provenance import _validate_responses, _published_inventory
        sample = self.load()['samples'][0]
        _published_inventory.cache_clear()
        before = set(sys.modules)
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA query forbidden')):
            with self.assertRaisesRegex(ValueError, 'published prompt'):
                _validate_responses(sample, real=True)
        self.assertFalse(any(name.startswith('sam3') for name in set(sys.modules) - before))

    def structural_audit(self, manifest):
        from proc_research.correction.provenance import _real_evidence, _reference, select_development
        evidence = {'files': {}, 'limitations': []}
        for key in ('development', 'production_index'):
            _reference(manifest, key, self.root, evidence)
        selected = select_development([json.loads((self.root/'synthetic-development.jsonl').read_text())])
        return _real_evidence(manifest, self.root, selected, evidence)

    def test_synthetic_record_graph_exercises_real_evidence_contract_only(self):
        manifest = write_structural_audit_fixture(self.root, self.path)
        identity, exports = self.structural_audit(manifest)
        self.assertEqual(identity['weight_sha256'], 'b'*64)
        self.assertIn(('LoveDA', 'synthetic-001', 1008), exports)
        # A synthetic bundle never becomes a real-model validation result.
        self.assertEqual(self.load()['evidence']['real_action_space'], 'PENDING')

    def test_independent_binding_corruption_is_failure(self):
        manifest = write_structural_audit_fixture(self.root, self.path)
        audit_path = Path(manifest['independent_receipt']['path'])
        audit = json.loads(audit_path.read_text())
        audit['bindings']['production_index'] = 'f'*64
        write_json(audit_path, audit)
        manifest['independent_receipt']['sha256'] = sha256(audit_path)
        with self.assertRaisesRegex(ValueError, 'independent receipt binding'):
            self.structural_audit(manifest)

    def test_missing_independent_export_is_pending(self):
        from proc_research.correction.provenance import PendingEvidence
        manifest = write_structural_audit_fixture(self.root, self.path)
        audit_path = Path(manifest['independent_receipt']['path'])
        audit = json.loads(audit_path.read_text())
        audit['exports'][0]['image_id'] = 'some-other-image'
        write_json(audit_path, audit)
        manifest['independent_receipt']['sha256'] = sha256(audit_path)
        with self.assertRaisesRegex(PendingEvidence, 'selected 1008'):
            self.structural_audit(manifest)

    def test_changed_producer_code_inventory_rejected(self):
        manifest = write_structural_audit_fixture(self.root, self.path)
        inventory_path = Path(manifest['producer_code_inventory']['path'])
        write_json(inventory_path, {'files': {'changed.py': 'f'*64}})
        manifest['producer_code_inventory']['sha256'] = sha256(inventory_path)
        audit_path = Path(manifest['independent_receipt']['path'])
        audit = json.loads(audit_path.read_text())
        audit['bindings']['producer_code_inventory'] = sha256(inventory_path)
        write_json(audit_path, audit)
        manifest['independent_receipt']['sha256'] = sha256(audit_path)
        with self.assertRaisesRegex(ValueError, 'code inventory'):
            self.structural_audit(manifest)

    def test_no_real_template_claim(self):
        from proc_research.correction.provenance import PendingEvidence, load_manifest
        template = Path(__file__).resolve().parents[2]/'research/p02/input.real.template.json'
        with self.assertRaises(PendingEvidence):
            load_manifest(template)


if __name__ == '__main__':
    unittest.main()
