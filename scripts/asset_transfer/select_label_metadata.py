"""Derive evaluator manifests AFTER scene splitting, without opening GT images."""
import argparse
import json
from pathlib import Path


def select_labels(manifest,inventory,output):
    rows=[json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    labels=[json.loads(x) for x in Path(inventory).read_text().splitlines() if x.strip()]
    lookup={(r['dataset'],r['image_id']):r for r in labels}
    if len(lookup)!=len(labels) or any(set(r)!={'dataset','image_id','path','sha256'} for r in labels):
        raise ValueError('invalid or duplicate label metadata')
    output=Path(output);paths={s:output/f'{s}-labels.jsonl' for s in ('pilot','confirmation')}
    if any(p.exists() for p in paths.values()):raise FileExistsError('label manifests are immutable')
    selected={s:[lookup[(r['dataset'],r['image_id'])] for r in rows if r['split']==s] for s in paths}
    if any(not items for items in selected.values()):raise ValueError('both frozen evaluation splits are required')
    output.mkdir(parents=True,exist_ok=True)
    for s,path in paths.items():
        with path.open('x') as f:
            for r in selected[s]:f.write(json.dumps(r)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','inventory','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();select_labels(a.manifest,a.inventory,a.output)


if __name__=='__main__':main()
