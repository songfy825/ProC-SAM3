"""Verify CPU P02 contracts, unchanged P01 regression and CLI negative exits."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proc_research.correction.runtime import install_cpu_guard, output_directory


def check_status(directory, expected, needle=None):
    """An expected nonzero exit must be the intended rejection, not a crash."""
    try:
        status = json.loads((Path(directory)/'status.json').read_text())
    except (OSError, ValueError):
        return False
    return status.get('status') == expected and (needle is None or needle in json.dumps(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New runtime/p02 audit directory')
    args = parser.parse_args()
    policy = install_cpu_guard()
    root = Path(__file__).resolve().parents[2]
    out = output_directory(args.output)
    from proc_research.cache import sha256, write_json
    launcher = ['bash', str(root/'scripts/p02/python.sh')]
    records = []

    def run(name, arguments, expected=0, raw=False):
        command = arguments if raw else launcher + arguments
        start = time.monotonic()
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        log = out/(name+'.log')
        log.write_text(completed.stdout+completed.stderr)
        record = {'name': name, 'command': command, 'cwd': str(root),
                  'expected_exit': expected, 'exit_code': completed.returncode,
                  'seconds': time.monotonic()-start, 'log': str(log), 'log_sha256': sha256(log)}
        record['passed'] = completed.returncode == expected
        records.append(record)
        print(name, 'exit', completed.returncode, 'expected', expected, flush=True)
        return completed

    for suite in ('p01', 'p02'):
        result = run(suite+'_contracts', ['scripts/p02/cpu_tests.py', '--suite', suite])
        matched = re.search(r'Ran (\d+) tests', result.stdout+result.stderr)
        records[-1]['tests'] = int(matched.group(1)) if matched else 0
        if not matched or (suite == 'p01' and int(matched.group(1)) != 60):
            records[-1]['passed'] = False
    for entry in ('analyze', 'verify'):
        run(entry+'_help', [f'scripts/p02/{entry}.py', '--help'])
        run(entry+'_usage_failure', [f'scripts/p02/{entry}.py', '--nonexistent-option'], expected=2)
    run('real_missing_pending', ['scripts/p02/analyze.py', '--manifest',
        'research/p02/input.real.template.json', '--output', str(out/'real-pending')], expected=2)
    records[-1]['status_contract'] = check_status(out/'real-pending', 'PENDING', 'development')
    records[-1]['passed'] &= records[-1]['status_contract']

    sys.path.insert(0, str(root/'tests/p02'))
    from test_provenance import write_fixture
    from PIL import Image
    import numpy as np
    fixture = out/'synthetic-input'
    manifest = write_fixture(fixture)
    image_hash = sha256(fixture/'synthetic.png')
    label_path = fixture/'synthetic-label.png'
    labels = fixture/'synthetic-labels.jsonl'

    def analyze(name, label_manifest=None, expected=0, reason=None):
        command = ['scripts/p02/analyze.py', '--manifest', str(manifest), '--output', str(out/name)]
        if label_manifest:
            command += ['--labels', str(label_manifest)]
        completed = run(name, command, expected)
        status = {0: 'DIAGNOSTICS_COMPLETE', 1: 'FAIL', 2: 'PENDING'}[expected]
        records[-1]['status_contract'] = check_status(out/name, status, reason)
        records[-1]['passed'] &= records[-1]['status_contract']
        return completed

    analyze('synthetic-no-gt')
    for number, value in enumerate((1, 2)):
        Image.fromarray(np.full((4, 8), value, dtype=np.uint8)).save(label_path)
        labels.write_text(json.dumps({'dataset':'LoveDA','image_id':'synthetic-001',
            'path':str(label_path),'sha256':sha256(label_path),'image_sha256':image_hash})+'\n')
        analyze('synthetic-gt-'+str(number), labels)
    label_path.unlink()
    analyze('synthetic-gt-deleted', labels)
    action_paths = [out/name/'actions.json' for name in ('synthetic-no-gt','synthetic-gt-0',
                                                        'synthetic-gt-1','synthetic-gt-deleted')]
    gt_invariant = all(p.is_file() for p in action_paths) and len({sha256(p) for p in action_paths}) == 1
    cache = fixture/'cache/synthetic-001'
    receipt = cache/'receipt.json'
    original = receipt.read_bytes()
    receipt.write_text('{"meta_sha256":"corrupt","arrays_sha256":"corrupt"}\n')
    analyze('synthetic-corrupt-receipt', expected=1, reason='receipt')
    receipt.write_bytes(original)
    analyze('synthetic-restored')
    (cache/'arrays.npz').rename(cache/'arrays.saved.npz')
    analyze('synthetic-cache-missing', expected=2, reason='cache')
    (cache/'arrays.saved.npz').rename(cache/'arrays.npz')
    run('frozen_p01_contract', ['sha256sum','-c','research/p01/evidence/contract-freeze.sha256'], raw=True)
    run('p01_original_unchanged', ['git','diff','--exit-code',
        'ec7f57ef2a3ec5cb166149a2fa85f5eff4a2f566','--','proc_research/cache.py',
        'proc_research/evaluate.py','proc_research/baseline.py','proc_research/calibration',
        'scripts/p01','tests/research','research/p01','sam3_segmentor_cached.py','core',
        'sam3','configs','prompt_pools','METHOD.md','PROGRESS.md','BLOCKED.md'], raw=True)
    run('whitespace', ['git','diff','--check'], raw=True)
    evidence = {'status':'PASS' if gt_invariant and all(r['passed'] for r in records) else 'FAIL',
                'kind':'CPU_SYNTHETIC_ENGINEERING_ONLY', 'REAL_ACTION_SPACE':'PENDING',
                'gt_actions_byte_identical':gt_invariant, 'runtime_policy':policy,
                'base_commit':'ec7f57ef2a3ec5cb166149a2fa85f5eff4a2f566',
                'method_sha256':sha256(root/'research/p02/METHOD.md'), 'commands':records,
                'input_hashes':{str(p):sha256(p) for p in sorted(fixture.rglob('*')) if p.is_file()},
                'source_hashes':{str(p.relative_to(root)):sha256(p) for folder in
                    ('proc_research/correction','scripts/p02','tests/p02')
                    for p in sorted((root/folder).glob('*')) if p.is_file()}}
    write_json(out/'verification.json', evidence)
    print('ENGINEERING VERIFICATION', evidence['status'], '; REAL_ACTION_SPACE=PENDING', flush=True)
    return 0 if evidence['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
