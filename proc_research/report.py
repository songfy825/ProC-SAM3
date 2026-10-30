"""Reproducible scene-paired comparisons and deliberately qualified decisions."""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
from .cache import sha256,write_json
from .statistics import paired_bootstrap
from .calibration.run import _receipt,_same_experiment,validate_bank_binding,METHODS


def build_report(prediction_run,evaluation_dir,bank_run,output):
    pred=Path(prediction_run);ev=Path(evaluation_dir);banks=Path(bank_run)
    manifest,_=_receipt(pred);bank_manifest,bank_receipt=_receipt(banks)
    _same_experiment(manifest['identity'],bank_manifest['identity'])
    if bank_manifest['identity'].get('stage')!='fit' or bank_receipt.get('kind')!='calibration_banks':
        raise ValueError('report requires a fit-stage calibration bank receipt')
    bank_hash=sha256(banks/'complete.json')
    if manifest['identity'].get('bank_receipt_sha256')!=bank_hash:
        raise ValueError('report must use the exact calibration bank that produced predictions')
    validate_bank_binding(dict(manifest['identity'],stage=manifest['identity'].get('evaluated_split')),bank_hash)
    provenance=json.loads((ev/'provenance.json').read_text())
    if provenance['prediction_index_sha256']!=sha256(pred/'prediction_index.json'):
        raise ValueError('evaluation belongs to a different prediction run')
    # Verify all evaluator artifacts, not just the input index.
    for name,h in provenance['output_sha256'].items():
        if sha256(ev/name)!=h:raise ValueError('evaluation artifact changed')
    results=json.loads((ev/'metrics.json').read_text())
    scenes=json.loads((ev/'per_scene.json').read_text())
    per_image=json.loads((ev/'per_image.json').read_text())
    fallback=json.loads((pred/'fallback.json').read_text())
    p=Path(output);p.mkdir(parents=True,exist_ok=False)
    candidate='multiscale_ecd_conditioned';comparisons=[];restricted=[];support_summary={}
    for dataset in manifest['identity']['config']['datasets']:
        lookup={}
        for row in scenes:
            if row['dataset']==dataset:lookup.setdefault(row['method'],{})[row['scene_id']]=row['confusion_matrix']
        ids=sorted(lookup[candidate]);a=np.array([lookup[candidate][s] for s in ids])
        for method in METHODS:
            if method==candidate:continue
            if sorted(lookup[method])!=ids:raise ValueError('unpaired original scene sets')
            b=np.array([lookup[method][s] for s in ids])
            comparison=paired_bootstrap(a,b,2000,20261030)
            per_class=[]
            for cls,(new,old) in enumerate(zip(results[f'{dataset}/{candidate}']['iou'],results[f'{dataset}/{method}']['iou'])):
                change=None if new is None or old is None else 100*(new-old)
                per_class.append(change)
                if change is not None and change < -1.:
                    restricted.append({'dataset':dataset,'baseline':method,'class_id':cls,'delta_pp':change})
            comparisons.append(dict(dataset=dataset,candidate=candidate,baseline=method,
                                    class_delta_pp=per_class,**comparison))
        meta=json.loads((banks/dataset/'multi/metadata.json').read_text())
        support_summary[dataset]={'support':meta['support'],'fallback_rows':meta['fallback'],
                                  'calibration_scenes':len(meta['meta']['scene_ids']),
                                  'evaluation_scenes':len(ids),
                                  'all_condition_rows_fallback':all(all(x) for x in meta['fallback'].values())}
    rates={}
    for r in fallback:
        key=f'{r["dataset"]}/{r["method"]}'
        total=rates.setdefault(key,{k:0 for k in ('pixels','fallback_pixels','disagreement_pixels','uncalibrated_pixels','no_evidence_pixels','signature_row_fallback_pixels')})
        for k in total:total[k]+=r[k]
    for r in rates.values():r['fallback_rate']=r['fallback_pixels']/max(r['pixels'],1)
    candidates=[r for r in per_image if r['method']==candidate]
    candidates.sort(key=lambda r:(float('inf') if r['miou'] is None else r['miou'],r['image_id']))
    random_cases=random.Random(20261030).sample(candidates,min(10,len(candidates)))
    scientific='EVIDENCE_INSUFFICIENT'
    if all(c['status']=='SUPERIOR' for c in comparisons) and not restricted and not any(s['all_condition_rows_fallback'] for s in support_summary.values()):
        scientific='SUPPORT_ON_FROZEN_SAMPLE'
    if manifest['identity'].get('evidence_kind')=='synthetic_cpu':scientific='PENDING_SYNTHETIC_ONLY'
    elif manifest['identity'].get('evaluated_split')!='confirmation':scientific='PENDING_CONFIRMATION'
    report={'research_status':scientific,'comparisons':comparisons,'class_degradation_over_1pp':restricted,
        'support':support_summary,'fallback':rates,'worst_cases':candidates[:10],'random_cases':random_cases,
        'pool_provenance':'construction images/splits/revision unknown; leakage-free pool NOT proven',
        'geographic_independence':'not inferred from parent scene ids',
        'resources':{'decode':json.loads((pred/'resources.json').read_text()),
                     'calibration':json.loads((banks/'resources.json').read_text())}}
    write_json(p/'comparisons.json',report)
    with open(p/'comparisons.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','candidate','baseline','delta_pp','lower95_pp','upper95_pp','scenes','status'])
        for r in comparisons:writer.writerow([r['dataset'],r['candidate'],r['baseline'],r['delta_pp'],*r['ci95_pp'],r['scenes'],r['status']])
    lines=['# P01 decision', '',f'Status: **{scientific}**.', '',
        'Scene-paired bootstrap: 2000 draws, seed 20261030, 95% percentile intervals. Only lower bound >0 supports superiority to that particular baseline.', '',
        'Single-view, same-view mean/RMS/global ECD and each shuffled seed are reported separately. A non-improving mean baseline never automatically stops calibration.', '',
        f'Protected class drops >1pp: {len(restricted)}. Any such drop restricts the conclusion.', '',
        'Calibration support and row/pixel fallback are in comparisons.json. All rows falling back to global means no conditional evidence.', '',
        'Released prompt-pool construction leakage and geographic independence are not proven. CPU synthetic results cannot establish model behavior or P01 gains.']
    (p/'decision.md').write_text('\n'.join(lines)+'\n')
    write_json(p/'provenance.json',{'prediction_receipt_sha256':sha256(pred/'complete.json'),
        'bank_receipt_sha256':bank_hash,'evaluation_provenance_sha256':sha256(ev/'provenance.json')})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prediction-run','evaluation-dir','bank-run','output'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();build_report(a.prediction_run,a.evaluation_dir,a.bank_run,a.output)


if __name__=='__main__':main()
