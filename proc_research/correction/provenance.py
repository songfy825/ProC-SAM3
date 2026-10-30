"""Offline evidence admission. Consistent hashes are necessary, never proof of origin.

A real run needs an independently produced audit receipt, producer records and a
pre-existing development/use-history record. This module does not manufacture
those records. Synthetic fixtures remain explicitly synthetic even when all
structural integrity checks pass. No model, GPU API, or network is invoked.
"""
import json
from functools import lru_cache
import pickle
import runpy
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ..cache import digest, load_sample, sha256


class PendingEvidence(ValueError):
    """Required real evidence has not yet been supplied; CLI should exit 2."""
    def __init__(self, reasons):
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__('PENDING: ' + '; '.join(self.reasons))


ROW_FIELDS = {'dataset', 'image_id', 'split', 'image_path', 'image_sha256',
              'height', 'width', 'source_scene_id', 'scene_evidence', 'city_or_flight'}
HASH_FIELDS = ('image_sha256', 'weight_sha256', 'pool_sha256', 'config_sha256', 'code_sha256')


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def select_development(rows):
    """Select before reading cache availability, GT, scene IDs or performance."""
    if not rows:
        raise PendingEvidence('existing development manifest is empty')
    seen = set()
    for row in rows:
        required = ROW_FIELDS - {'source_scene_id', 'scene_evidence', 'city_or_flight'}
        if not isinstance(row, dict) or set(row) - ROW_FIELDS or required - set(row):
            raise ValueError('development schema invalid; GT/label fields forbidden')
        if row['dataset'] not in ('LoveDA', 'iSAID') or row['split'] not in ('development', 'pilot'):
            raise ValueError('only existing development/pilot images are permitted')
        if not isinstance(row['image_id'], str) or not row['image_id']:
            raise ValueError('image_id must be a nonempty string')
        if not _hash(row['image_sha256']):
            raise ValueError('invalid image_sha256')
        if any(type(row[x]) is not int or row[x] <= 0 for x in ('height', 'width')):
            raise ValueError('invalid image dimensions')
        key = (row['dataset'], row['image_id'])
        if key in seen:
            raise ValueError('duplicate development image')
        seen.add(key)
    selected = []
    for dataset in sorted({r['dataset'] for r in rows}):
        selected.extend(sorted((r for r in rows if r['dataset'] == dataset),
                               key=lambda r: r['image_id'])[:50])
    return selected


def _path(value, parent):
    p = Path(value)
    return p if p.is_absolute() else parent / p


def _reference(manifest, key, parent, evidence):
    ref = manifest.get(key)
    if not isinstance(ref, dict) or not ref.get('path') or not ref.get('sha256'):
        raise PendingEvidence(f'missing {key} path and independently pinned SHA256')
    if set(ref) != {'path', 'sha256'} or not _hash(ref['sha256']):
        raise ValueError(f'invalid {key} reference/hash')
    p = _path(ref['path'], parent)
    if not p.is_file():
        raise PendingEvidence(f'missing {key}: {p}')
    if sha256(p) != ref['sha256']:
        raise ValueError(f'{key} hash mismatch: {p}')
    evidence['files'][key] = {'path': str(p.resolve()), 'sha256': ref['sha256']}
    return p


def _json(path):
    return json.loads(Path(path).read_text())


def _real_evidence(manifest, parent, selected, evidence):
    names = ('production_manifest', 'production_receipt', 'independent_receipt',
             'production_log', 'use_history', 'producer_config', 'producer_code_inventory',
             'asset_receipt')
    paths = {key: _reference(manifest, key, parent, evidence) for key in names}
    producer = _json(paths['production_manifest'])
    completed = _json(paths['production_receipt'])
    audit = _json(paths['independent_receipt'])
    ident = producer.get('identity', {})
    if producer.get('status') != 'PASS' or producer.get('identity_sha256') != digest(ident):
        raise ValueError('producer identity or completion status mismatch')
    if ident.get('evidence_kind') != 'real_model':
        raise ValueError('producer record is not a real-model production record')
    if completed.get('status') != 'PASS' or completed.get('kind') not in ('real_gpu_cache', 'real_gpu_g0'):
        raise ValueError('production receipt is not a completed real cache run')
    # Do not scan/check every old cache in the run. Bind small parent files now;
    # each selected cache is independently verified below.
    for key in ('production_manifest', 'production_index'):
        record = evidence['files'][key]
        rel = Path(record['path']).resolve().relative_to(paths['production_receipt'].resolve().parent)
        if completed.get('files', {}).get(str(rel)) != record['sha256']:
            raise ValueError(f'production receipt does not bind {key}')
    if paths['independent_receipt'].resolve().parent == paths['production_receipt'].resolve().parent:
        raise ValueError('independent receipt must be outside the producer run directory')
    if audit.get('kind') != 'independent_cache_audit' or audit.get('status') != 'PASS':
        raise ValueError('independent production audit is missing or failed')
    if not audit.get('auditor') or not audit.get('recorded_at') or not audit.get('producer_command'):
        raise ValueError('independent audit lacks who/when/production command evidence')
    bindings = audit.get('bindings', {})
    for key in ('development', 'production_index', 'production_manifest', 'production_receipt',
                'production_log', 'use_history', 'producer_config', 'producer_code_inventory', 'asset_receipt'):
        if bindings.get(key) != evidence['files'][key]['sha256']:
            raise ValueError(f'independent receipt binding mismatch: {key}')
    if not paths['production_log'].read_text().strip():
        raise ValueError('production log is empty')
    if ident.get('config_sha256') != evidence['files']['producer_config']['sha256']:
        raise ValueError('producer config hash mismatch')
    if ident.get('config') != _json(paths['producer_config']):
        raise ValueError('producer config content mismatch')
    inventory = _json(paths['producer_code_inventory'])
    files = inventory.get('files', {})
    if not files or any(not _hash(h) for h in files.values()) or digest(files) != ident.get('code_sha256'):
        raise ValueError('producer code inventory hash mismatch')
    assets = _json(paths['asset_receipt'])
    if assets.get('weight', {}).get('sha256') != ident.get('weight_sha256'):
        raise ValueError('independent asset/weight identity mismatch')
    history = _json(paths['use_history'])
    if (history.get('development_sha256') != evidence['files']['development']['sha256']
            or history.get('selection_basis') != 'existing-development-list'):
        raise ValueError('data use history is not bound to pre-existing development list')
    history_rows = history.get('rows', [])
    lookup = {(r['dataset'], r['image_id']): r for r in history_rows}
    if len(lookup) != len(history_rows):
        raise ValueError('duplicate data use history entries')
    for row in selected:
        item = lookup.get((row['dataset'], row['image_id']))
        if item is None:
            raise PendingEvidence(f'data use history missing for {row["dataset"]}/{row["image_id"]}')
        if (item.get('development_eligible') is not True or not isinstance(item.get('prior_uses'), list)
                or not item.get('eligibility_evidence') or item.get('image_sha256') != row['image_sha256']):
            raise ValueError('unsubstantiated development eligibility/use history')
        if row.get('source_scene_id'):
            if (item.get('source_scene_id') != row['source_scene_id']
                    or item.get('scene_evidence') != row.get('scene_evidence')
                    or not row.get('scene_evidence')):
                raise ValueError('scene association lacks matching independent evidence')
    evidence['limitations'].append('External auditor attests production origin; hashes verify consistency, not authenticity. '
                                   'Published pool construction data history remains unknown.')
    exports = audit.get('exports')
    if not isinstance(exports, list) or not exports:
        raise ValueError('independent audit lacks per-image cache export records')
    export_lookup = {(e['dataset'], e['image_id'], e['view']): e for e in exports}
    if len(export_lookup) != len(exports):
        raise ValueError('duplicate independent export record')
    for row in selected:
        if (row['dataset'], row['image_id'], 1008) not in export_lookup:
            raise PendingEvidence('independent audit missing a selected 1008 export record')
    return ident, export_lookup


@lru_cache(maxsize=2)
def _published_inventory(dataset):
    # Importing core.prompt_bank normally executes core/__init__.py and imports
    # SAM3/autocast. Read this pinned, pure vocabulary module directly instead.
    # No PromptBank constructor, processor, model, or embedding method is called.
    from ..baseline import ROOT, model_options
    options = model_options(dataset)  # checks released pool SHA256 before pickle
    cls = runpy.run_path(str(ROOT/'core/prompt_bank.py'))['PromptBank']
    bank = cls.__new__(cls)
    words, ids = bank.get_cls_idx(options['classname_path'])
    with open(options['expanded_prompt_pool_path'], 'rb') as f:
        pool = pickle.load(f)
    prompts, mapping = bank._expand_query_words(words, ids, prompt_pool=pool)
    if len(prompts) <= len(words) or set(mapping) != set(ids):
        raise ValueError('published prompt expansion requires unsupported fallback')
    return {'prompts': prompts, 'prompt_to_class': mapping,
            'num_classes': max(ids) + 1,
            'pool_sha256': sha256(options['expanded_prompt_pool_path'])}


def _validate_responses(sample, real):
    if sample.view != 1008 or sample.geometry['encoder_input'] != [1, 3, 1008, 1008]:
        raise ValueError('P02 requires the original 1008 view')
    scores = sample.scores.numpy()
    prompts = sample.prompt_scores.numpy()
    mapping = sample.prompt_to_class.numpy()
    if prompts.shape[0] == 0:
        raise PendingEvidence('missing per-prompt responses: prompt_scores is empty')
    c = scores.shape[0]
    if set(mapping.tolist()) != set(range(c)) or sample.active.tolist() != list(range(c)):
        raise ValueError('full published class vocabulary required; no routing or absent prompt classes')
    pooled = np.stack([prompts[mapping == k].max(axis=0) for k in range(c)])
    if not np.array_equal(scores, pooled):
        raise ValueError('class aggregation differs from original max of per-prompt responses')
    expected = scores.argmax(0)
    expected[scores.max(0) < .1] = 0
    if not np.array_equal(sample.prediction.numpy(), expected):
        raise ValueError('y0 differs from original max response / 0.1 background rule')
    if real:
        # This reads only pinned tracked pools and class metadata, never a model.
        inventory = _published_inventory(sample.dataset)
        if (sample.class_ids != tuple(range(inventory['num_classes']))
                or mapping.tolist() != inventory['prompt_to_class']
                or sample.provenance.get('prompts') != inventory['prompts']
                or sample.provenance['pool_sha256'] != inventory['pool_sha256']):
            raise ValueError('published prompt pool, class vocabulary or prompt→class mismatch')


def load_manifest(path, *, stream=False):
    """Return {kind,samples,selected,evidence}; never read or accept GT paths.

    Set stream=True to return a single-pass sample iterator; consume it fully
    before using final cache evidence. A late missing/corrupt selected cache
    raises during iteration, and the whole analysis must remain incomplete.
    Paths in references resolve beside this manifest. Cache paths resolve beside
    the production index. Development image paths resolve beside the development
    JSONL file. Missing selected records cause PENDING, never backfill/skip.
    """
    path = Path(path)
    if not path.is_file():
        raise PendingEvidence(f'missing input manifest: {path}')
    manifest = _json(path)
    allowed = {'schema', 'kind', 'development', 'production_index', 'production_manifest',
               'production_receipt', 'independent_receipt', 'production_log', 'use_history',
               'producer_config', 'producer_code_inventory', 'asset_receipt'}
    if set(manifest) - allowed or manifest.get('schema') != 1 or manifest.get('kind') not in ('real', 'synthetic'):
        raise ValueError('invalid manifest schema/kind; real=true is not provenance')
    kind = manifest['kind']
    evidence = {'input_sha256': sha256(path), 'kind': kind, 'files': {},
                'real_action_space': 'PENDING' if kind == 'synthetic' else 'AVAILABLE_FOR_DIAGNOSIS',
                'scene_independence_established': False, 'limitations': [],
                'sampling': 'lexicographic image_id first 50 per dataset, selected before cache/GT availability'}
    development = _reference(manifest, 'development', path.parent, evidence)
    rows = [json.loads(line) for line in development.read_text().splitlines() if line.strip()]
    selected = select_development(rows)
    evidence['selected'] = [{'dataset': r['dataset'], 'image_id': r['image_id'],
                             'image_sha256': r['image_sha256']} for r in selected]
    index_path = _reference(manifest, 'production_index', path.parent, evidence)
    index = _json(index_path)
    entries = index.get('entries')
    if not isinstance(entries, list):
        raise ValueError('production index entries missing')
    if kind == 'synthetic' and index.get('kind') != 'synthetic':
        raise ValueError('synthetic input requires an explicitly synthetic index')
    if kind == 'real' and index.get('kind') == 'synthetic':
        raise ValueError('synthetic cache index cannot become real by relabeling input')
    ident, audited_exports = (_real_evidence(manifest, path.parent, selected, evidence)
                              if kind == 'real' else (None, {}))
    lookup = {}
    for entry in entries:
        key = (entry['dataset'], entry['image_id'], entry['view'])
        if key in lookup:
            raise ValueError('duplicate production cache key')
        lookup[key] = entry
    cache_evidence = []
    evidence['caches'] = cache_evidence
    def iter_samples():
        for row in selected:
            key = (row['dataset'], row['image_id'], 1008)
            entry = lookup.get(key)
            if entry is None:
                raise PendingEvidence(f'missing selected 1008 per-prompt cache: {key}; no replacement sampling')
            directory = _path(entry['path'], index_path.parent)
            for name in ('meta.json', 'arrays.npz', 'receipt.json'):
                if not (directory/name).is_file():
                    raise PendingEvidence(f'missing selected cache file {directory/name}')
            expected = entry.get('receipt', {})
            if set(expected) != {'meta_sha256', 'arrays_sha256'} or not all(_hash(v) for v in expected.values()):
                raise ValueError('external parent cache receipt is invalid')
            if _json(directory/'receipt.json') != expected:
                raise ValueError('local receipt differs from independently pinned parent receipt')
            if sha256(directory/'meta.json') != expected['meta_sha256'] or sha256(directory/'arrays.npz') != expected['arrays_sha256']:
                raise ValueError('cache content hash mismatch')
            with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
                if 'prompt_scores' not in arrays.files:
                    raise PendingEvidence(f'missing per-prompt prompt_scores: {directory}; cannot reconstruct from class maps')
            sample = load_sample(directory, expected=expected)
            if (sample.dataset, sample.image_id, sample.view) != key:
                raise ValueError('cache dataset/image/view association mismatch')
            if list(sample.scores.shape[-2:]) != [row['height'], row['width']]:
                raise ValueError('development/cache original geometry mismatch')
            if sample.scene_id != row.get('source_scene_id', '') or entry.get('scene_id', '') != sample.scene_id:
                raise ValueError('cache scene association mismatch')
            image = _path(row['image_path'], development.parent)
            if not image.is_file():
                raise PendingEvidence(f'missing selected image: {image}')
            if sha256(image) != row['image_sha256'] or sample.provenance['image_sha256'] != row['image_sha256']:
                raise ValueError('selected image hash association mismatch')
            with Image.open(image) as im:
                if (im.height, im.width) != (row['height'], row['width']):
                    raise ValueError('image byte dimensions differ from original cache geometry')
            if ident is not None:
                if sample.provenance.get('kind') == 'synthetic':
                    raise ValueError('synthetic cache cannot be promoted to real')
                audited = audited_exports[key]
                if (audited.get('receipt') != expected
                        or audited.get('geometry') != sample.geometry
                        or any(audited.get('provenance', {}).get(k) != sample.provenance[k] for k in HASH_FIELDS)):
                    raise ValueError('independent per-image export evidence mismatch')
                for field in ('weight_sha256', 'config_sha256', 'code_sha256'):
                    if sample.provenance[field] != ident.get(field):
                        raise ValueError(f'cache/producer identity mismatch: {field}')
                if sample.provenance['pool_sha256'] != ident.get('pool_sha256', {}).get(sample.dataset):
                    raise ValueError('cache/producer pool identity mismatch')
            elif sample.provenance.get('kind') != 'synthetic':
                raise ValueError('synthetic cache must retain its synthetic marker')
            _validate_responses(sample, kind == 'real')
            cache_evidence.append({'dataset': sample.dataset, 'image_id': sample.image_id,
                                   'path': str(directory.resolve()), **expected,
                                   'provenance': sample.provenance, 'geometry': sample.geometry})
            yield sample
    evidence['scene_ids_with_evidence'] = sum(bool(r.get('source_scene_id') and r.get('scene_evidence')) for r in selected)
    evidence['limitations'].append('Scene association is not proof of geographic independence; no independence claim.')
    if kind == 'synthetic':
        evidence['limitations'].append('Synthetic algorithm evidence only; real action-space statistics and effects remain PENDING.')
    samples = iter_samples()
    return {'kind': kind, 'samples': samples if stream else list(samples),
            'selected': selected, 'evidence': evidence}
