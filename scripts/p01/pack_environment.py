"""Pack a read-only environment through a disposable, permission-correct copy.

No --ignore-missing-files. Original Conda archives supply link targets/modes;
installed file contents are retained. Run with a Python containing conda-pack
and zstandard, not with the environment being archived.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import zipfile


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def restore_member(root,member):
    root=Path(root).resolve();relative=Path(member.name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe Conda member')
    target=root/relative
    if member.issym():
        destination=(target.parent/member.linkname).resolve()
        if not destination.is_relative_to(root):raise ValueError('external environment symlink')
        if target.is_symlink():target.unlink()
        elif target.is_dir():shutil.rmtree(target) # disposable staging copy only
        elif target.exists():target.unlink()
        target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(member.linkname)
        return {'path':member.name,'link':member.linkname}
    if target.exists() and not target.is_symlink():target.chmod(member.mode)
    return None


def export_locks(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    packages=[json.loads(p.read_text()) for p in sorted((source/'conda-meta').glob('*.json'))]
    explicit=['# platform: linux-64','@EXPLICIT']
    for p in packages:explicit.append(p['url']+'#'+p['md5'])
    (output/'conda-linux-64.explicit.txt').write_text('\n'.join(explicit)+'\n')
    root=Path(__file__).resolve().parents[2]
    existing=(root/'research/p01/environment-lock.txt').read_text().splitlines()
    conda_names={p['name'].lower().replace('_','-') for p in packages}
    pip=[line for line in existing if line and not line.startswith('#') and line.split('==')[0].lower().replace('_','-') not in conda_names]
    pip=[x.replace('torch==2.7.1','torch==2.7.1+cu126').replace('torchvision==0.22.1','torchvision==0.22.1+cu126') for x in pip]
    (output/'requirements-pip.lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu126\n'+'\n'.join(pip)+'\n')
    lines=['name: p01-sam3','channels:','  - conda-forge','dependencies:']
    lines += [f'  - {p["name"]}={p["version"]}={p["build"]}' for p in packages]
    lines += ['  - pip:','      - --extra-index-url https://download.pytorch.org/whl/cu126']
    lines += ['      - '+x for x in pip]
    (output/'environment.yml').write_text('\n'.join(lines)+'\n')
    return packages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--locks',required=True);parser.add_argument('--staging-parent',default='/tmp')
    args=parser.parse_args()
    import conda_pack
    import zstandard
    source=Path(args.source).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('archive is immutable')
    packages=export_locks(source,args.locks);repairs=[];archives=[]
    if shutil.disk_usage(args.staging_parent).free<14*1024**3:raise OSError('need 14 GiB temporary staging space')
    with tempfile.TemporaryDirectory(prefix='p01-pack-',dir=args.staging_parent) as tmp:
        stage=Path(tmp)/'environment'
        print('Copying read-only installed environment to',stage,flush=True)
        shutil.copytree(source,stage,symlinks=True)
        for p in packages:
            cache=Path(p['link']['source']);archive=cache.with_name(p['fn'])
            if not archive.is_file():raise FileNotFoundError(f'original Conda archive required: {archive}')
            if sha(archive)!=p['sha256']:raise ValueError(f'Conda archive integrity mismatch: {archive}')
            archives.append({'name':p['fn'],'sha256':p['sha256']})
            with zipfile.ZipFile(archive) as z:
                name=next(x for x in z.namelist() if x.startswith('pkg-'))
                with z.open(name) as f,zstandard.ZstdDecompressor().stream_reader(f) as stream,tarfile.open(fileobj=stream,mode='r|') as tf:
                    for member in tf:
                        repair=restore_member(stage,member)
                        if repair:repairs.append(repair)
        # Pip console scripts also lost execute bits on the original filesystem.
        for p in (stage/'bin').iterdir():
            if p.is_file() and not p.is_symlink():
                with p.open('rb') as f:magic=f.read(4)
                if magic.startswith(b'#!') or magic==b'\x7fELF':p.chmod(p.stat().st_mode|0o111)
        env=conda_pack.CondaEnv.from_prefix(str(stage)) # strict file verification
        # Installed binaries contain the original prefix, not the staging prefix.
        from conda_pack.core import File
        installed=[]
        for f in env.files:
            # Use the verified installed copy, never the original noexec package
            # cache (whose links/modes have the same filesystem damage).
            placeholder=str(source) if f.file_mode in ('text','binary') else f.prefix_placeholder
            installed.append(File(str(stage/f.target),f.target,is_conda=f.is_conda,
                                  file_mode=f.file_mode,prefix_placeholder=placeholder))
        packed=conda_pack.CondaEnv(str(source),installed)
        print('Strict validation passed;',len(env.files),'files;',len(repairs),'links restored in staging only',flush=True)
        packed.pack(output=str(output),format='tar.gz',compress_level=1,n_threads=2,verbose=True)
    receipt={'source_prefix':str(source),'archive':str(output),'bytes':output.stat().st_size,'sha256':sha(output),
             'source_unchanged':True,'repairs_in_disposable_copy':repairs,'conda_archive_hashes':archives,
             'platform':'linux-x86_64','GPU_validation':'PENDING','note':'Never packed weights or datasets; source environment not modified.'}
    Path(str(output)+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
    Path(str(output)+'.sha256').write_text(receipt['sha256']+'  '+output.name+'\n')
    print('Packed:',output,'bytes:',receipt['bytes'],'sha256:',receipt['sha256'],flush=True)


if __name__=='__main__':main()
