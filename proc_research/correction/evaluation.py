"""Independent GT evaluation and bounded pooled-mIoU reference search.

This module alone reads P02 labels. Action construction never imports it.
All metrics are full-image, dataset-specific pooled confusion metrics.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from proc_research.cache import sha256
from proc_research.evaluate import map_labels, confusion_matrix, metrics

DATASET_CLASSES = {'LoveDA': 7, 'iSAID': 16}


def _array(value):
    if hasattr(value, 'detach'):
        if value.device.type != 'cpu':
            raise ValueError('evaluation accepts CPU arrays only')
        return value.detach().numpy()
    return np.asarray(value)


def pooled_metrics(targets, predictions, num_classes):
    """Sum full-image matrices, never average image mIoU values."""
    if len(targets) != len(predictions):
        raise ValueError('target/prediction count mismatch')
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        cm += confusion_matrix(target, prediction, num_classes)
    return metrics(cm)


def repair_damage(y0, prediction, target, num_classes):
    """Counts use GT-valid pixels; coverage also records all geometry pixels."""
    y0, prediction, target = map(_array, (y0, prediction, target))
    confusion_matrix(target, y0, num_classes)
    confusion_matrix(target, prediction, num_classes)
    valid = target != 255
    changed = prediction != y0
    repaired = valid & changed & (y0 != target) & (prediction == target)
    damaged = valid & changed & (y0 == target) & (prediction != target)
    return {
        'valid_pixels': int(valid.sum()),
        'geometry_pixels': int(y0.size),
        'changed_pixels': int(changed.sum()),
        'changed_valid': int((changed & valid).sum()),
        'changed_ignored': int((changed & ~valid).sum()),
        'modification_coverage': float(changed.mean()) if changed.size else 0.0,
        'repaired': int(repaired.sum()), 'damaged': int(damaged.sum()),
        'net_repaired': int(repaired.sum() - damaged.sum()),
        'per_class': [
            {'class_id': c,
             'repaired': int((repaired & (target == c)).sum()),
             'damaged': int((damaged & (target == c)).sum()),
             'prediction_count_delta': int((prediction == c).sum() - (y0 == c).sum()),
             'valid_prediction_count_delta': int(((prediction == c) & valid).sum() - ((y0 == c) & valid).sum())}
            for c in range(num_classes)],
    }


def _validated_options(item, num_classes, arm):
    """Build sparse confusion deltas for legal, disjoint shared choices."""
    y0, target = _array(item['y0']), _array(item['target'])
    base = confusion_matrix(target, y0, num_classes)
    table = item['table']
    if list(y0.shape) != table['shape']:
        raise ValueError('action table geometry mismatch')
    if len(table['regions']) > 4:
        raise ValueError('region budget exceeded')
    if arm not in ('MASK', 'REGION'):
        raise ValueError('unknown action arm')
    vocabulary = table['class_ids']
    if (not vocabulary or any(not isinstance(c, (int, np.integer)) or isinstance(c, bool)
                              or not 0 <= c < num_classes for c in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or not np.isin(y0, vocabulary).all()):
        raise ValueError('baseline/class IDs outside table vocabulary')
    region_options = []
    region_ids = set()
    occupied = set()
    flat_y, flat_t = y0.ravel(), target.ravel()
    for region in table['regions']:
        region_id = region['region_id']
        if not isinstance(region_id, str) or not region_id or region_id in region_ids:
            raise ValueError('duplicate/invalid region ID')
        region_ids.add(region_id)
        raw_pixels = region['pixels']
        if not raw_pixels:
            raise ValueError('empty region is not a legal action domain')
        if any(not isinstance(p, (int, np.integer)) for p in raw_pixels):
            raise ValueError('non-integer region coordinate')
        pixels = set(raw_pixels)
        if len(pixels) != len(raw_pixels) or occupied.intersection(pixels):
            raise ValueError('duplicate or overlapping regions')
        if any(p < 0 or p >= y0.size for p in pixels):
            raise ValueError('region coordinate outside image')
        occupied.update(pixels)
        candidates = region['candidates']
        if len(candidates) > 3:
            raise ValueError('candidate budget exceeded')
        options = [('KEEP', np.zeros_like(base))]
        seen_ids = {'KEEP'}
        for candidate in candidates:
            cid = candidate['candidate_id']
            if not isinstance(cid, str) or cid in seen_ids:
                raise ValueError('duplicate/invalid shared candidate ID')
            seen_ids.add(cid)
            c = candidate['class_id']
            if not isinstance(c, (int, np.integer)) or isinstance(c, bool) or c not in vocabulary:
                raise ValueError('candidate class outside table vocabulary')
            signatures = {}
            for arm_key in ('mask', 'region'):
                changes = candidate[arm_key + '_changes']
                seen_pixels = set()
                for pair in changes:
                    if (len(pair) != 2 or any(not isinstance(v, (int, np.integer)) for v in pair)):
                        raise ValueError('non-integer label change')
                    p, label = pair
                    if p not in pixels or p in seen_pixels or label != c or flat_y[p] == c:
                        raise ValueError('invalid or out-of-region label change')
                    seen_pixels.add(p)
                signatures[arm_key] = seen_pixels
            expected_region = {p for p in pixels if flat_y[p] != c}
            if signatures['region'] != expected_region or not signatures['mask'] <= expected_region:
                raise ValueError('action changes violate MASK/REGION semantics')
            chosen = np.asarray(sorted(signatures[arm.lower()]), dtype=np.int64)
            chosen = chosen[flat_t[chosen] != 255]
            delta = np.zeros_like(base)
            if len(chosen):
                np.add.at(delta, (flat_t[chosen], flat_y[chosen]), -1)
                np.add.at(delta, (flat_t[chosen], np.full(len(chosen), c)), 1)
            options.append((cid, delta))
        region_options.append(options)
    combinations = []
    for joint in itertools.product(*region_options):
        choices = [entry[0] for entry in joint]
        cm = base.copy()
        for _, delta in joint:
            cm += delta
        if (cm < 0).any() or cm.sum() != base.sum():
            raise AssertionError('confusion replacement violated conservation')
        combinations.append((choices, cm))
    if len(combinations) > 256:
        raise ValueError('joint action budget exceeded')
    return base, combinations


def bounded_oracle(items, num_classes, arm):
    """At most three coordinate sweeps of <=256 legal joint choices per image.

    The objective is the cumulative whole-subset confusion matrix. This is a
    bounded GT reference, never a global optimum or an upper-bound claim.
    """
    if not items:
        raise ValueError('bounded reference requires a nonempty dataset')
    ordered = sorted(items, key=lambda x: x['image_id'])
    ids = [x['image_id'] for x in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate image IDs in reference dataset')
    prepared = [_validated_options(item, num_classes, arm) for item in ordered]
    current = [base.copy() for base, _ in prepared]
    total = sum(current, np.zeros((num_classes, num_classes), dtype=np.int64))
    choices = {item['image_id']: ['KEEP'] * len(item['table']['regions']) for item in ordered}
    initial_choices = {key: list(value) for key, value in choices.items()}
    initial = metrics(total)
    trace = []
    evaluations = 0
    sweeps = 0
    for sweep in range(1, 4):
        sweeps = sweep
        improved = False
        for i, item in enumerate(ordered):
            before = metrics(total)['miou']
            best_score = before
            best_choices = list(choices[item['image_id']])
            best_cm = current[i]
            other = total - current[i]
            trials = []
            for candidate_choices, cm in prepared[i][1]:
                value = metrics(other + cm)['miou']
                evaluations += 1
                trials.append({'choices': candidate_choices, 'pooled_miou': value})
                # None means every target pixel is ignored, with no metric to improve.
                if value is not None and (best_score is None or value > best_score):
                    best_score, best_choices, best_cm = value, list(candidate_choices), cm
            accepted = best_score is not None and (before is None or best_score > before)
            if accepted:
                total = other + best_cm
                current[i] = best_cm
                choices[item['image_id']] = best_choices
                improved = True
            trace.append({'sweep': sweep, 'image_id': item['image_id'],
                          'before_miou': before, 'after_miou': metrics(total)['miou'],
                          'accepted': accepted, 'choices': list(choices[item['image_id']]),
                          'trials': trials})
        if not improved:
            break
    return {'name': 'bounded oracle reference', 'arm': arm,
            'budget': {'maximum_sweeps': 3, 'maximum_joint_actions_per_image': 256},
            'initial_choices': initial_choices, 'initial': initial, 'final': metrics(total),
            'choices': choices, 'sweeps': sweeps, 'evaluations': evaluations, 'trace': trace}


def _coverable(item, num_classes):
    y0, target = _array(item['y0']).ravel(), _array(item['target']).ravel()
    wrong = (target != 255) & (y0 != target)
    result = {}
    for arm in ('MASK', 'REGION'):
        union = set()
        regions = []
        for region in item['table']['regions']:
            covered = set()
            for candidate in region['candidates']:
                for p, c in candidate[arm.lower() + '_changes']:
                    if wrong[p] and target[p] == c:
                        covered.add(p)
            union.update(covered)
            regions.append({'region_id': region['region_id'], 'coverable_errors': len(covered)})
        result[arm] = {'baseline_errors': int(wrong.sum()), 'coverable_errors': len(union),
                       'per_class': [{'class_id': c, 'coverable_errors': sum(target[p] == c for p in union)}
                                     for c in range(num_classes)],
                       'regions': regions}
        for row in result[arm]['per_class']:
            row['coverable_errors'] = int(row['coverable_errors'])
    return result


def _read_label_index(label_manifest):
    if label_manifest is None:
        return None, None
    manifest = Path(label_manifest)
    if not manifest.is_file():
        return None, None
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    lookup = {}
    for row in rows:
        key = (row['dataset'], row['image_id'])
        if key in lookup:
            raise ValueError('duplicate label keys')
        lookup[key] = row
    return lookup, manifest


def _evaluate_complete(items, num_classes, kind):
    from .actions import apply_actions
    targets = [item['target'] for item in items]
    base = [item['y0'] for item in items]
    predictions = {'BASE': base, 'KEEP': base}
    for arm in ('MASK', 'REGION'):
        predictions['MAJORITY_' + arm] = [
            apply_actions(item['y0'], item['table'], item['table']['majority_ids'], arm)
            for item in items]
    references = {arm: bounded_oracle(items, num_classes, arm) for arm in ('MASK', 'REGION')}
    for arm in ('MASK', 'REGION'):
        predictions['ORACLE_' + arm] = [
            apply_actions(item['y0'], item['table'], references[arm]['choices'][item['image_id']], arm)
            for item in items]
    methods = {name: pooled_metrics(targets, preds, num_classes) for name, preds in predictions.items()}
    per_image = []
    for i, item in enumerate(items):
        region_effects = []
        for region in item['table']['regions']:
            pixels = np.asarray(region['pixels'], dtype=np.int64)
            region_effects.append({'region_id': region['region_id'], 'methods': {
                name: repair_damage(item['y0'].ravel()[pixels][None],
                                    preds[i].ravel()[pixels][None],
                                    item['target'].ravel()[pixels][None], num_classes)
                for name, preds in predictions.items()}})
        per_image.append({'image_id': item['image_id'], 'scene_id': item['scene_id'],
                          'coverable': _coverable(item, num_classes), 'region_effects': region_effects,
                          'methods': {name: repair_damage(item['y0'], preds[i], item['target'], num_classes)
                                      for name, preds in predictions.items()}})
    count_keys = ('valid_pixels', 'geometry_pixels', 'changed_pixels', 'changed_valid',
                  'changed_ignored', 'repaired', 'damaged', 'net_repaired')
    changes = {}
    for name in predictions:
        records = [row['methods'][name] for row in per_image]
        total = {key: sum(row[key] for row in records) for key in count_keys}
        total['modification_coverage'] = (total['changed_pixels'] / total['geometry_pixels']
                                          if total['geometry_pixels'] else 0.0)
        total['per_class'] = [dict(class_id=c, **{
            key: sum(row['per_class'][c][key] for row in records)
            for key in ('repaired', 'damaged', 'prediction_count_delta', 'valid_prediction_count_delta')})
            for c in range(num_classes)]
        changes[name] = total
    coverable = {arm: sum(row['coverable'][arm]['coverable_errors'] for row in per_image)
                 for arm in ('MASK', 'REGION')}
    cheap_names = ('KEEP', 'MAJORITY_MASK', 'MAJORITY_REGION')
    scored = [(methods[name]['miou'], name) for name in cheap_names if methods[name]['miou'] is not None]
    strongest = max(scored, key=lambda pair: pair[0])[1] if scored else None
    mask = methods['ORACLE_MASK']['miou']
    deltas = {}
    for name, comparator in (('BASE', 'BASE'), ('REGION_REFERENCE', 'ORACLE_REGION'),
                             ('STRONGEST_FROZEN_CHEAP', strongest)):
        value = methods[comparator]['miou'] if comparator else None
        deltas[name] = 100 * (mask - value) if mask is not None and value is not None else None
    finite = [x for x in deltas.values() if x is not None]
    minimum = min(finite) if len(finite) == 3 else None
    novel = 0
    for item in items:
        for region in item['table']['regions']:
            region_outputs = {()} | {tuple(map(tuple, c['region_changes'])) for c in region['candidates']}
            novel += len({tuple(map(tuple, c['mask_changes'])) for c in region['candidates']} - region_outputs)
    screen = ('STOP_NO_NEW_MASK_OUTPUTS' if novel == 0 else
              'PENDING_NO_VALID_GT_PIXELS' if minimum is None else
              'CONFIGURATION_DOES_NOT_SUPPORT_FURTHER_INVESTMENT' if minimum < 0.5 else
              'SCREENING_LINE_REACHED')
    return {'status': 'COMPLETE_DESCRIPTIVE', 'kind': kind, 'image_count': len(items),
            'scientific_conclusion': 'INCONCLUSIVE',
            'limitations': ['bounded search is not a global optimum or upper bound',
                            'small development subset does not establish generalization',
                            'scene IDs alone do not prove independent geographic scenes',
                            'synthetic results only validate algorithms' if kind == 'synthetic' else
                            'screening line is not an effectiveness proof'],
            'methods': methods, 'per_image': per_image, 'changes': changes,
            'coverable_errors': coverable, 'oracle': references,
            'strongest_frozen_cheap': strongest, 'mask_increment_pp': deltas,
            'minimum_increment_pp': minimum, 'new_mask_outputs': novel,
            'configuration_screen': screen,
            'real_action_space': 'PENDING' if kind == 'synthetic' else 'DESCRIBED'}


def evaluate_dataset(samples, tables, label_manifest=None, *, kind='real'):
    """Read separate labels, group datasets and refuse incomplete-subset oracles.

    Label JSONL entries require dataset/image_id/path/sha256/image_sha256.
    Relative label paths resolve against the label manifest. Optional scene_id
    must match cache metadata. No missing label is replaced or silently dropped.
    """
    if kind not in ('real', 'synthetic'):
        raise ValueError('invalid evidence kind')
    if len(samples) != len(tables):
        raise ValueError('sample/action count mismatch')
    lookup, manifest = _read_label_index(label_manifest)
    grouped = {}
    seen = set()
    for sample, table in zip(samples, tables):
        key = (sample.dataset, sample.image_id)
        if key in seen or key != (table['dataset'], table['image_id']):
            raise ValueError('duplicate or mismatched sample/action association')
        seen.add(key)
        if sample.dataset not in DATASET_CLASSES:
            raise ValueError('unsupported dataset')
        grouped.setdefault(sample.dataset, []).append((sample, table))
    result = {}
    for dataset, pairs in sorted(grouped.items()):
        items = []
        missing = []
        for sample, table in sorted(pairs, key=lambda pair: pair[0].image_id):
            key = (dataset, sample.image_id)
            row = lookup.get(key) if lookup is not None else None
            if row is None:
                missing.append({'image_id': sample.image_id, 'reason': 'label manifest/association unavailable'})
                continue
            if row.get('image_sha256') != sample.provenance['image_sha256']:
                raise ValueError('label image association hash mismatch')
            scene = getattr(sample, 'scene_id', None)
            if row.get('scene_id') is not None and row['scene_id'] != scene:
                raise ValueError('label scene association mismatch')
            path = Path(row['path'])
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                missing.append({'image_id': sample.image_id, 'reason': 'label file unavailable', 'path': str(path)})
                continue
            if sha256(path) != row['sha256']:
                raise ValueError('label content hash mismatch')
            with Image.open(path) as image:
                target = map_labels(np.asarray(image), dataset)
            y0 = _array(sample.prediction)
            # Production P02 provenance requires the original dataset vocabulary;
            # cached prediction stores channel IDs, so apply explicit class mapping.
            labels = np.asarray(sample.class_ids, dtype=np.int64)
            if y0.size and ((y0 < 0).any() or (y0 >= len(labels)).any()):
                raise ValueError('cached prediction index outside class mapping')
            y0 = labels[y0]
            confusion_matrix(target, y0, DATASET_CLASSES[dataset])
            items.append({'image_id': sample.image_id, 'scene_id': scene, 'y0': y0,
                          'table': table, 'target': target})
        if missing:
            result[dataset] = {'status': 'PENDING_GT', 'kind': kind, 'image_count': len(pairs),
                               'missing': missing,
                               'reason': 'all selected labels required; no partial-subset oracle'}
        else:
            result[dataset] = _evaluate_complete(items, DATASET_CLASSES[dataset], kind)
            result[dataset]['label_manifest_sha256'] = sha256(manifest)
    return result
