"""Target-only path aliases. Default is a dry run; never overwrite any path."""
import argparse
import json
from pathlib import Path


def install_links(bundle,code=None,root=Path('/'),apply=False,existing_data_root=None,loveda_name='LoveDA',checkpoint=None):
    bundle=Path(bundle).resolve();root=Path(root).resolve()
    if loveda_name in ('','..','.') or Path(loveda_name).name!=loveda_name:raise ValueError('unsafe LoveDA folder name')
    data=Path(existing_data_root).resolve() if existing_data_root else bundle/'datasets'
    loveda=data/loveda_name;isaid=data/'iSAID'
    weight=Path(checkpoint).resolve() if checkpoint else bundle/'weights/sam3.pt'
    mapping={
        '/home/fengyi/sam3.pt':weight,
        '/home/fengyi/E/data/SegEarth/LoveDA':loveda,
        '/home/fengyi/E/data/SegEarth/iSAID':isaid,
        '/home/fengyi/data/SegEarth/LoveDA':loveda,
        '/home/fengyi/data/SegEarth/iSAID':isaid,
        '/home/fengyi/CodeHome/ProC-SAM3-repro/manifests':bundle/'manifests',
        '/home/fengyi/CodeHome/Proc/runtime/p01':bundle/'runtime',
        '/data/weights/sam3.pt':weight,
        '/data/datasets/LoveDA':loveda,
        '/data/datasets/iSAID':isaid,
        '/data/source-manifests':bundle/'manifests',
        '/data/p01-inputs':bundle/'inputs',
        '/data/p01-runs':bundle/'runtime',
    }
    if code is not None:
        code=Path(code).resolve()
        if not (code/'scripts/p01/launch.sh').is_file():raise ValueError('not a P01 code checkout')
        mapping['/home/fengyi/CodeHome/Proc/src/ProC-SAM3']=code
        mapping['/home/fengyi/E/CodeHome/Proc/src/ProC-SAM3']=code
    plan=[]
    for alias,target in mapping.items():
        if not target.exists():raise FileNotFoundError(f'missing payload: {target}')
        dest=root/alias.lstrip('/')
        if not dest.parent.resolve().is_relative_to(root):raise ValueError('target parent escapes installation root')
        if dest.exists() or dest.is_symlink():
            if dest.resolve()!=target.resolve():raise FileExistsError(f'refusing conflicting existing path: {dest}')
            action='already correct'
        else:
            action='create'
            for parent in dest.parents:
                if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
                    raise FileExistsError(f'parent is missing or not a directory: {parent}')
        plan.append({'alias':str(dest),'target':str(target),'action':action})
    # Preflight all conflicts before making any changes.
    if apply:
        for item in plan:
            if item['action']=='create':
                dest=Path(item['alias'])
                # Existing parent aliases can make two planned names the same path.
                if dest.exists() or dest.is_symlink():
                    if dest.resolve()==Path(item['target']).resolve():continue
                    raise FileExistsError(f'path changed after preflight: {dest}')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(item['target'])
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',default=str(Path(__file__).resolve().parent));p.add_argument('--code')
    p.add_argument('--existing-data-root');p.add_argument('--loveda-name',default='LoveDA');p.add_argument('--checkpoint')
    p.add_argument('--root',default='/',help='alternate prefix for CPU contract testing')
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    print(json.dumps(install_links(a.bundle,a.code,a.root,a.apply,a.existing_data_root,a.loveda_name,a.checkpoint),indent=2))


if __name__=='__main__':main()
