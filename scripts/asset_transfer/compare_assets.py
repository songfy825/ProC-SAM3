"""Read-only byte comparison; Python 3.8+; expected inventory arrives on stdin."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',required=True)
    a=p.parse_args();roots={k:Path(v) for k,v in json.loads(a.roots).items()}
    raw=sys.stdin.buffer.read();expected=json.loads(raw);results=[];actual=[];start=time.monotonic()
    if not roots or not expected or set(roots)!={r['dataset'] for r in expected}:
        raise ValueError('nonempty exact dataset coverage is required')
    for dataset in sorted(roots):
        root=roots[dataset];records=[r for r in expected if r['dataset']==dataset]
        report={'dataset':dataset,'root':str(root),'expected_files':len(records),'matched_files':0,
                'checked_bytes':0,'missing':[],'size_mismatch':[],'hash_mismatch':[],'changed_during_read':[]}
        names=set()
        for i,r in enumerate(records):
            relative=Path(r['relative_path'])
            if relative.is_absolute() or '..' in relative.parts or str(relative) in names:
                raise ValueError('unsafe or duplicate expected path')
            names.add(str(relative));f=root/relative
            if not f.is_file():report['missing'].append(str(relative));continue
            before=f.stat();h=sha(f);after=f.stat();report['checked_bytes']+=before.st_size
            actual.append({'dataset':dataset,'relative_path':str(relative),'bytes':before.st_size,'sha256':h})
            stable=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            if not stable:report['changed_during_read'].append(str(relative))
            if before.st_size!=r['bytes']:report['size_mismatch'].append(str(relative))
            if h!=r['sha256']:report['hash_mismatch'].append(str(relative))
            if stable and before.st_size==r['bytes'] and h==r['sha256']:report['matched_files']+=1
            if (i+1)%2000==0:print('{} {}/{} files hashed'.format(dataset,i+1,len(records)),file=sys.stderr,flush=True)
        found={str(f.relative_to(root)) for sub in ('img_dir/val','ann_dir/val')
               for f in (root/sub).rglob('*') if f.is_file()}
        report['extra_files']=sorted(found-names)
        report['required_files_match']=report['matched_files']==len(records)
        report['exact_validation_tree_match']=report['required_files_match'] and not report['extra_files']
        results.append(report)
    canonical=lambda rows:json.dumps(sorted(rows,key=lambda r:(r['dataset'],r['relative_path'])),sort_keys=True,separators=(',',':')).encode()
    expected_digest=hashlib.sha256(canonical(expected)).hexdigest();actual_digest=hashlib.sha256(canonical(actual)).hexdigest()
    result={'status':'PASS' if all(r['exact_validation_tree_match'] for r in results) and expected_digest==actual_digest else 'DIFFERENT',
        'inventory_sha256':hashlib.sha256(raw).hexdigest(),'expected_canonical_sha256':expected_digest,
        'actual_canonical_sha256':actual_digest,
        'datasets':results,'seconds':time.monotonic()-start,'writes_to_dataset':False,'GPU_used':False}
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':sys.exit(1)


if __name__=='__main__':main()
