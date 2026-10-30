"""Independent P01 review probes; synthetic CPU evidence only, no model construction."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

from proc_research.cache import SampleCache, write_sample, write_json, sha256, RESERVE_BYTES, digest
from proc_research.protocol import create_run
from proc_research.calibration.run import ROOT, load_config, code_identity, finish, fit_stage, decode_stage
from proc_research.evaluate import evaluate_predictions
from proc_research.report import build_report

OUT = Path(__file__).resolve().parent
WORK = OUT / 'synthetic-probes'
WORK.mkdir(exist_ok=False)
torch.set_num_threads(1)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not torch.cuda.is_initialized()
cfg = load_config(ROOT / 'research/p01/config.json')
ident = dict(config=cfg, config_sha256=digest(cfg), manifest_sha256='b'*64,
             method_sha256=sha256(ROOT/'METHOD.md'), code_sha256=code_identity(),
             weight_sha256='c'*64, pool_sha256={'LoveDA':'d'*64,'iSAID':'e'*64},
             evidence_kind='synthetic_cpu')
labels = []
for split in ('calibration', 'pilot'):
    run = create_run(WORK, split, dict(ident, stage=split))
    entries=[]
    for dataset in ('LoveDA','iSAID'):
        for i in range(2):
            for view in (1008,504):
                scores=torch.full((3,8,12), .1)
                scores[1]=.6
                scores[2,:,i:i+2]=.8
                sample=SampleCache(f'{split}-i{i}',f'{split}-s{i}',dataset,view,
                    scores,scores.clone(),torch.ones(3),torch.ones(3),torch.arange(3),
                    torch.arange(3),(0,1,2),scores.argmax(0),
                    {'kind':'full_resize','original_hw':[8,12],
                     'encoder_input':[1,3,view,view], 'feature_shapes':[[1,2,1,1]]},
                    {**{k:'a'*64 for k in ('image_sha256','weight_sha256','pool_sha256','config_sha256','code_sha256')},
                     'audit_only':'synthetic shape metadata; not an encoder measurement'})
                name=f'cache/{dataset}-{i}-{view}'
                receipt=write_sample(run/name,sample)
                entries.append(dict(dataset=dataset,image_id=sample.image_id,scene_id=sample.scene_id,
                    view=view,split=split,path=name,receipt=receipt))
            if split=='pilot':
                target=sample.prediction.numpy().astype('uint8')+(1 if dataset=='LoveDA' else 0)
                path=WORK/f'{dataset}-{i}-synthetic-label.png'
                Image.fromarray(target).save(path)
                labels.append(dict(dataset=dataset,image_id=sample.image_id,path=str(path),sha256=sha256(path)))
    write_json(run/'cache_index.json',{'entries':entries})
    finish(run,{'kind':'synthetic_cpu_fixture'})

fit_stage(SimpleNamespace(cache_run=str(WORK/'calibration'),output=str(WORK),run_id='banks-a'))
fit_stage(SimpleNamespace(cache_run=str(WORK/'calibration'),output=str(WORK),run_id='banks-b'))
with patch('shutil.disk_usage',return_value=SimpleNamespace(free=RESERVE_BYTES-1)) as usage:
    decode_stage(SimpleNamespace(cache_run=str(WORK/'pilot'),bank_run=str(WORK/'banks-a'),
        output=str(WORK),run_id='predictions-under-reserve'))
    disk_queries=usage.call_count
pred=WORK/'predictions-under-reserve'
output_files=list((pred/'predictions').rglob('*.npy'))
label_file=WORK/'labels.jsonl'
label_file.write_text(''.join(json.dumps(r)+'\n' for r in labels))
evaluate_predictions(pred/'prediction_index.json',label_file,WORK/'evaluation')
report=build_report(pred,WORK/'evaluation',WORK/'banks-b',WORK/'report-with-other-bank')
expected_bank=json.loads((pred/'manifest.json').read_text())['identity']['bank_receipt_sha256']
actual_bank=sha256(WORK/'banks-b/complete.json')

changed=dict(cfg,datasets=['LoveDA'])
changed_file=WORK/'single-dataset-config.json'
write_json(changed_file,changed)
accepted=load_config(changed_file)['datasets']

result={
    'kind':'INDEPENDENT_CPU_SYNTHETIC_REVIEW_ONLY',
    'cuda_initialized':torch.cuda.is_initialized(),
    'disk_reserve_probe':{'simulated_free_bytes':RESERVE_BYTES-1,'required_reserve_bytes':RESERVE_BYTES,
        'decode_completed':(pred/'complete.json').is_file(),'disk_queries':disk_queries,
        'prediction_files_written':len(output_files),'prediction_bytes_written':sum(p.stat().st_size for p in output_files)},
    'report_bank_probe':{'prediction_bank_receipt_sha256':expected_bank,
        'provided_bank_receipt_sha256':actual_bank,'different_receipt':expected_bank!=actual_bank,
        'report_completed':True,'research_status':report['research_status'],
        'caveat':'Both banks are deterministic fits to the same synthetic cache; this proves absent receipt binding, not changed scientific metrics.'},
    'config_probe':{'changed_field':'datasets','accepted':accepted,
        'caveat':'Normal launcher copies the tracked two-dataset config; this concerns standalone config validation.'}}
assert not result['cuda_initialized']
write_json(OUT/'cpu-probe-results.json',result)
print(json.dumps(result,indent=2))
