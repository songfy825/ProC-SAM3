"""CPU-only asset transport. Never imports model code or interprets label pixels."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

PREFIX='p01-assets'
RESERVE=20*1024**3


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in name:
        raise ValueError('unsafe archive path')
    return str(p)


class HashReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.count=0
    def read(self,size=-1):
        b=self.stream.read(size);self.digest.update(b);self.count+=len(b);return b


def pack_files(entries,output,metadata):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    partial=Path(str(output)+'.partial')
    if output.exists() or partial.exists():raise FileExistsError('archive/run is immutable; choose a new output')
    names=[safe_name(e['name']) for e in entries]
    if len(names)!=len(set(names)) or 'MANIFEST.json' in names:raise ValueError('duplicate/reserved member')
    total=sum(e['bytes'] for e in entries)
    if shutil.disk_usage(output.parent).free < total+len(entries)*2048+64*1024**2+RESERVE:
        raise OSError('insufficient conservative archive capacity plus 20 GiB reserve')
    completed=[];written=0;next_notice=1024**3;start=time.monotonic()
    with partial.open('xb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT,bufsize=1024**2) as tf:
            tf.copybufsize=1024**2
            for folder in (PREFIX,PREFIX+'/inputs',PREFIX+'/runtime'):
                info=tarfile.TarInfo(folder);info.type=tarfile.DIRTYPE;info.mode=0o755;tf.addfile(info)
            for entry,name in zip(entries,names):
                source=Path(entry['source']);before=source.stat()
                if before.st_size!=entry['bytes']:raise ValueError(f'source size changed: {source}')
                info=tarfile.TarInfo(PREFIX+'/'+name);info.size=entry['bytes'];info.mode=0o644
                with source.open('rb') as f:
                    reader=HashReader(f);tf.addfile(info,reader)
                after=source.stat()
                if (reader.count!=entry['bytes'] or reader.digest.hexdigest()!=entry['sha256']
                    or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)):
                    raise ValueError(f'source hash or stat changed: {source}')
                completed.append(dict(entry,name=name));written+=entry['bytes']
                if written>=next_notice:
                    print(f'packed {len(completed)}/{len(entries)} files; input {written/1024**3:.2f} GiB; elapsed {time.monotonic()-start:.1f}s',flush=True)
                    next_notice=written+1024**3
                    if shutil.disk_usage(output.parent).free<RESERVE:raise OSError('20 GiB reserve reached')
            manifest={'schema':1,'metadata':metadata,'files':completed}
            data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
            info=tarfile.TarInfo(PREFIX+'/MANIFEST.json');info.mode=0o644;info.size=len(data);tf.addfile(info,io.BytesIO(data))
    partial.rename(output)
    return manifest


def verify_archive(path):
    observed={};manifest=None;members=set();start=time.monotonic();total=0;notice=1024**3
    with gzip.open(path,'rb') as compressed,tarfile.open(fileobj=compressed,mode='r|',bufsize=1024**2) as tf:
        for member in tf:
            name=safe_name(member.name)
            if name in members:raise ValueError('duplicate archive member')
            members.add(name)
            if not name.startswith(PREFIX+'/') and name!=PREFIX:raise ValueError('unexpected archive root')
            if member.isdir():continue
            if not member.isfile():raise ValueError('archive links and special files forbidden')
            stream=tf.extractfile(member)
            if name==PREFIX+'/MANIFEST.json':
                manifest=json.loads(stream.read());continue
            h=hashlib.sha256();size=0
            for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b);size+=len(b)
            observed[name[len(PREFIX)+1:]]={'bytes':size,'sha256':h.hexdigest()};total+=size
            if total>=notice:
                print(f'verified {len(observed)} files; {total/1024**3:.2f} GiB; elapsed {time.monotonic()-start:.1f}s',flush=True)
                notice=total+1024**3
        # tar stops at its end markers; gzip must also validate its footer/CRC.
        while compressed.read(4*1024**2):pass
    if manifest is None:raise ValueError('missing archive manifest')
    expected={e['name']:{'bytes':e['bytes'],'sha256':e['sha256']} for e in manifest['files']}
    if len(expected)!=len(manifest['files']) or observed!=expected:raise ValueError('archive member hash/size mismatch')
    return {'status':'PASS','files':len(observed),'payload_bytes':total,'archive_bytes':Path(path).stat().st_size,
            'archive_sha256':sha256(path),'seconds':time.monotonic()-start,'GPU':'not used'}


def collect(repo,support):
    repo=Path(repo);support=Path(support);entries=[];counts={}
    initial=json.loads((repo/'research/p01/evidence/assets_initial.json').read_text())
    original=Path('/home/fengyi/CodeHome/ProC-SAM3-repro/manifests')
    dataset_meta={d['dataset']:d for d in json.loads((original/'datasets.json').read_text())}
    def add(path,name,expected=None,size=None):
        path=Path(path);entries.append({'source':str(path),'name':name,
            'bytes':path.stat().st_size if size is None else size,'sha256':expected or sha256(path)})
    weight=initial['weight'];add(weight['path'],'weights/sam3.pt',weight['sha256'],weight['bytes'])
    labels=[]
    for dataset in ('LoveDA','iSAID'):
        infer=original/f'{dataset}-inference.jsonl';paired=original/f'{dataset}.jsonl'
        if sha256(infer)!=initial['assets'][str(infer)]['sha256']:raise ValueError('inference manifest changed')
        if sha256(paired)!=dataset_meta[dataset]['manifest_sha256']:raise ValueError('paired manifest changed')
        rows=[json.loads(x) for x in paired.read_text().splitlines()]
        images=[json.loads(x) for x in infer.read_text().splitlines()]
        if len(rows)!=len(images) or {r['id']:r['image_sha256'] for r in rows}!={r['id']:r['image_sha256'] for r in images}:
            raise ValueError('paired and inference inventory mismatch')
        root=Path('/home/fengyi/E/data/SegEarth')/dataset
        for r in rows:
            for role in ('image','gt'):
                source=Path(r[role]);relative=source.relative_to(root)
                add(source,f'datasets/{dataset}/{relative}',r[role+'_sha256'],r[role+'_bytes'])
            labels.append({'dataset':dataset,'image_id':r['id'],'path':r['gt'],'sha256':r['gt_sha256']})
        counts[dataset]={'images':len(rows),'GT_files':len(rows),
                         'image_bytes':sum(r['image_bytes'] for r in rows),'GT_bytes':sum(r['gt_bytes'] for r in rows)}
        add(infer,'manifests/'+infer.name);add(paired,'manifests/'+paired.name)
    label_file=support/'label_inventory.jsonl'
    with label_file.open('x') as f:
        for row in labels:f.write(json.dumps(row)+'\n')
    add(label_file,'evaluation/label_inventory.jsonl')
    for name in ('cfg_loveda_expanded_prompt_pool_hybrid_strict.pkl','cfg_iSAID_expanded_prompt_pool_hybrid_strict.pkl'):
        path=repo/'prompt_pools'/name
        add(path,'released_assets/prompt_pools/'+name,initial['assets'][str(path)]['sha256'])
    for name in ('cls_loveda.txt','cls_loveda_exp.txt','cls_iSAID.txt','cls_iSAID_exp.txt'):
        path=repo/'configs'/name;add(path,'released_assets/configs/'+name,initial['assets'][str(path)]['sha256'])
    add(repo/'sam3/assets/bpe_simple_vocab_16e6.txt.gz','released_assets/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    add(repo/'research/p01/transfer.example.json','inputs/4090-v1.json')
    add(repo/'requirements.txt','reference/requirements.txt')
    add(repo/'research/p01/evidence/transfer-verification.json','reference/cpu-verification.json')
    for name in ('package.py','setup_links.py','prepare_inputs.sh','select_label_metadata.py'):
        add(repo/'scripts/asset_transfer'/name,name)
    add(repo/'research/p01/asset-transfer/RESTORE.md','RESTORE.md')
    return entries,{'counts':counts,'base_code_commit':'ec7f57ef2a3ec5cb166149a2fa85f5eff4a2f566',
        'production_code_sha256':'ca1816ca8606689c2f4077851d89819758b38dead540c6be86807563659eb9a3',
        'scope':'Asset transport only; no selection by labels, model execution, training or GPU work.',
        'pending':['reviewed source-scene map','real isolated split manifests','GPU G0/pilot/confirmation'],
        'method_unchanged':True,'full_pool_transport_is_not_full_pool_inference':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--repo',required=True);b.add_argument('--output-dir',required=True)
    v=sub.add_parser('verify');v.add_argument('archive');v.add_argument('--receipt')
    a=p.parse_args()
    if a.command=='build':
        out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);support=out/'support';support.mkdir()
        entries,metadata=collect(a.repo,support)
        (out/'source_inventory.json').write_text(json.dumps({'files':entries,'metadata':metadata},indent=2)+'\n')
        archive=out/'p01-assets.tar.gz';pack_files(entries,archive,metadata)
        result=verify_archive(archive)
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'SHA256SUMS').write_text(result['archive_sha256']+'  '+archive.name+'\n')
    else:
        result=verify_archive(a.archive)
        if a.receipt:
            with open(a.receipt,'x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
