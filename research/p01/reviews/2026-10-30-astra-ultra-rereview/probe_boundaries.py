"""Independent additional boundary probes; all tensors and labels are synthetic CPU."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import shutil
import torch

from proc_research.cache import SampleCache,write_sample,write_json,sha256,digest,RESERVE_BYTES
from proc_research.protocol import create_run
from proc_research.calibration.run import ROOT,load_config,code_identity,finish,fit_stage,decode_stage,prediction_budget
from proc_research.report import build_report

OUT=Path(__file__).resolve().parent
WORK=OUT/'synthetic-boundaries'
WORK.mkdir(exist_ok=False)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('P01_ENABLE_GPU')=='0'
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
cfg=load_config(ROOT/'research/p01/config.json')
ident=dict(config=cfg,config_sha256=digest(cfg),manifest_sha256='b'*64,method_sha256=sha256(ROOT/'METHOD.md'),
           code_sha256=code_identity(),weight_sha256='c'*64,pool_sha256={'LoveDA':'d'*64,'iSAID':'e'*64},evidence_kind='synthetic_cpu')
entries_by_split={}
for split in ('calibration','pilot'):
    run=create_run(WORK,split,dict(ident,stage=split));entries=[]
    for di,dataset in enumerate(('LoveDA','iSAID')):
        for i in range(2):
            h,w=8+4*di,12+4*i
            for view in (1008,504):
                scores=torch.full((3,h,w),.1);scores[1]=.6;scores[2,:,i:i+2]=.8
                sample=SampleCache(f'{split}-i{i}',f'{split}-s{i}',dataset,view,scores,scores.clone(),
                    torch.ones(3),torch.ones(3),torch.arange(3),torch.arange(3),(0,1,2),scores.argmax(0),
                    {'kind':'full_resize','original_hw':[h,w],'encoder_input':[1,3,view,view],
                     'feature_shapes':[[1,2,1,1]],'note':'synthetic metadata, not encoder evidence'},
                    {k:'a'*64 for k in ('image_sha256','weight_sha256','pool_sha256','config_sha256','code_sha256')})
                name=f'cache/{dataset}-{i}-{view}';receipt=write_sample(run/name,sample)
                entries.append(dict(dataset=dataset,image_id=sample.image_id,scene_id=sample.scene_id,
                    view=view,split=split,path=name,receipt=receipt))
    write_json(run/'cache_index.json',{'entries':entries});finish(run,{'kind':'synthetic_cpu_fixture'})
    entries_by_split[split]=entries
for name in ('bank-a','bank-b'):
    fit_stage(SimpleNamespace(cache_run=str(WORK/'calibration'),output=str(WORK),run_id=name))
def decode(name):
    decode_stage(SimpleNamespace(cache_run=str(WORK/'pilot'),bank_run=str(WORK/'bank-a'),output=str(WORK),run_id=name))
def prediction_files(name):return sorted((WORK/name/'predictions').rglob('*.npy'))
results={}
for name,cutoff in [('drop_between_methods',1),('drop_before_metadata',44)]:
    calls=[]
    def disk_available(path):
        written=len(prediction_files(name));calls.append({'path':str(path),'written':written})
        return SimpleNamespace(free=RESERVE_BYTES-1 if written>=cutoff else RESERVE_BYTES+32*1024**2)
    try:
        with patch('shutil.disk_usage',side_effect=disk_available):decode(name)
    except OSError as error:
        count=len(prediction_files(name))
        assert count==cutoff
        assert (WORK/name/'manifest.json').is_file() and not (WORK/name/'complete.json').exists()
        if cutoff==44:assert not (WORK/name/'prediction_index.json').exists()
        results[name]={'status':'PASS','written_before_rejection':count,'error':str(error),'queries':len(calls),
                       'completed_receipt':False,'partial_outputs_preserved':True}
    else:raise AssertionError('late disk loss was not rejected')

budget,metadata=prediction_budget(WORK/'pilot',entries_by_split['pilot'],cfg['datasets'])
sizes=[(8,12),(8,16),(12,12),(12,16)]
raw_bytes=sum(h*w*8*11 for h,w in sizes)
assert budget>=raw_bytes+44*128+metadata
decode('complete-normal')
files=prediction_files('complete-normal')
assert len(files)==44
assert sum(torch.from_numpy(__import__('numpy').load(p)).numel()*8 for p in files)==raw_bytes
results['heterogeneous_shapes']={'status':'PASS','unique_images':4,'views_per_image':2,'methods':11,
    'raw_prediction_bytes':raw_bytes,'budget_bytes':budget,'metadata_headroom':metadata,
    'output_files':len(files),'completed_receipt':(WORK/'complete-normal/complete.json').is_file()}

# Use a deliberately absent evaluator directory to prove rejection precedes evaluator reads.
pred=WORK/'complete-normal'
for name,bank in [('wrong_bank',WORK/'bank-b')]:
    try:build_report(pred,WORK/'absent-evaluation',bank,WORK/(name+'-report'))
    except ValueError as error:
        assert 'exact calibration bank' in str(error)
        assert not (WORK/(name+'-report')).exists()
        results[name]={'status':'PASS','error':str(error),'evaluation_read_before_rejection':False}
    else:raise AssertionError('wrong bank not rejected')

for name,frozen in [('missing_frozen',None),('wrong_frozen','f'*64)]:
    copied=WORK/name;shutil.copytree(pred,copied)
    m=json.loads((copied/'manifest.json').read_text());m['identity']['evaluated_split']='confirmation'
    if frozen is not None:m['identity']['frozen_bank_receipt_sha256']=frozen
    else:m['identity'].pop('frozen_bank_receipt_sha256',None)
    m['identity_sha256']=digest(m['identity']);write_json(copied/'manifest.json',m)
    (copied/'complete.json').unlink();finish(copied,{'kind':'predictions'})
    try:build_report(copied,WORK/'absent-evaluation',WORK/'bank-a',WORK/(name+'-report'))
    except ValueError as error:
        assert 'exact frozen calibration bank' in str(error)
        assert not (WORK/(name+'-report')).exists()
        results[name]={'status':'PASS','error':str(error),'evaluation_read_before_rejection':False}
    else:raise AssertionError('invalid frozen binding not rejected')

assert not torch.cuda.is_initialized()
result={'status':'PASS','scope':'CPU synthetic boundary probes only; no model construction or real GPU acceptance',
        'cuda_initialized':False,'code_sha256':ident['code_sha256'],'probes':results}
write_json(OUT/'boundary-results.json',result)
print(json.dumps(result,indent=2))
