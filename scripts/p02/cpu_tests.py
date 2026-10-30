"""Run unchanged CPU tests behind a no-driver/no-network process policy."""
import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proc_research.correction.runtime import install_cpu_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('p01', 'p02'), required=True)
    args = p.parse_args()
    install_cpu_guard()
    root = Path(__file__).resolve().parents[2]
    directory = root/'tests'/('research' if args.suite == 'p01' else 'p02')
    suite = unittest.defaultTestLoader.discover(str(directory), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print('Skipped tests forbidden:', result.skipped, file=sys.stderr)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
