"""P02 CPU offline action-space analysis; no inference, tuning or downloads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proc_research.correction.runtime import install_cpu_guard, output_directory


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def summarize(tables):
    from proc_research.correction.actions import summarize_tables
    return summarize_tables(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Pinned development and producer evidence JSON')
    parser.add_argument('--labels', help='Separate optional evaluation-only JSONL; never affects actions')
    parser.add_argument('--output', required=True, help='New directory under runtime/p02 (no overwrite)')
    args = parser.parse_args()
    policy = install_cpu_guard()
    from proc_research.cache import sha256
    from proc_research.correction.provenance import load_manifest, PendingEvidence
    from proc_research.correction.actions import build_actions, apply_actions
    from proc_research.correction.evaluation import evaluate_dataset
    out = output_directory(args.output)
    try:
        loaded = load_manifest(args.manifest, stream=True)
        from types import SimpleNamespace
        samples, tables = [], []
        for sample in loaded['samples']:
            tables.append(build_actions(sample))
            samples.append(SimpleNamespace(dataset=sample.dataset, image_id=sample.image_id,
                scene_id=sample.scene_id, prediction=sample.prediction,
                class_ids=sample.class_ids, provenance=sample.provenance))
        del sample
    except PendingEvidence as exc:
        result = {'ENGINEERING': 'ENTRYPOINT_EXECUTED', 'REAL_ACTION_SPACE': 'PENDING',
                  'status': 'PENDING', 'reasons': exc.reasons, 'runtime_policy': policy}
        write(out/'status.json', result)
        print(json.dumps(result, sort_keys=True))
        return 2
    except (ValueError, KeyError, OSError) as exc:
        write(out/'status.json', {'status': 'FAIL', 'error': str(exc), 'REAL_ACTION_SPACE': 'PENDING'})
        print(str(exc), file=sys.stderr)
        return 1
    # Written before importing/reading any target labels. GT permutations/deletion
    # are tested by byte-comparing this exact artifact across complete CLI runs.
    write(out/'actions.json', tables)
    summary = summarize(tables)
    for sample, table in zip(samples, tables):
        import numpy as np
        y0 = np.asarray(sample.class_ids, dtype=np.int64)[sample.prediction.numpy()]
        image = next(row for row in summary[sample.dataset]['per_image_stats'] if row['image_id'] == sample.image_id)
        image['selectors'] = {}
        for selector, choices in [('KEEP', ['KEEP']*len(table['regions'])), ('majority', table['majority_ids'])]:
            for arm in ('MASK', 'REGION'):
                pred = apply_actions(y0, table, choices, arm)
                changed = int((pred != y0).sum())
                image['selectors'][selector+'/'+arm] = {'shared_ids': choices,
                    'changed_pixels': changed, 'changed_fraction': changed/y0.size,
                    'no_op': changed == 0, 'all_keep': all(x == 'KEEP' for x in choices)}
    for dataset_summary in summary.values():
        dataset_summary['kind'] = loaded['kind']
        dataset_summary['real_action_space'] = 'PENDING' if loaded['kind'] == 'synthetic' else 'DESCRIBED'
        dataset_summary['effect_judgment'] = 'PENDING_SEE_INDEPENDENT_EVALUATION'
    write(out/'summary.json', summary)
    try:
        evaluation = evaluate_dataset(samples, tables, args.labels, kind=loaded['kind'])
    except (ValueError, KeyError, OSError) as exc:
        write(out/'status.json', {'status': 'FAIL', 'error': str(exc),
              'phase': 'independent_evaluation', 'kind': loaded['kind'],
              'REAL_ACTION_SPACE': 'PENDING' if loaded['kind'] == 'synthetic' else 'DESCRIBED'})
        write(out/'output_hashes.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
        print(str(exc), file=sys.stderr)
        return 1
    write(out/'evaluation.json', evaluation)
    result = {'status': 'DIAGNOSTICS_COMPLETE', 'kind': loaded['kind'],
              'REAL_ACTION_SPACE': 'PENDING' if loaded['kind'] == 'synthetic' else 'DESCRIBED',
              'scientific_conclusion': 'PENDING_SYNTHETIC_ONLY' if loaded['kind'] == 'synthetic' else 'SEE_DATASET_EVALUATION',
              'runtime_policy': policy, 'evidence': loaded['evidence'],
              'manifest_sha256': sha256(args.manifest),
              'actions_sha256': sha256(out/'actions.json'),
              'labels_sha256': sha256(args.labels) if args.labels and Path(args.labels).is_file() else None}
    write(out/'status.json', result)
    write(out/'output_hashes.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({k: result[k] for k in ('status', 'kind', 'REAL_ACTION_SPACE')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
