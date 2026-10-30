"""End-to-end CPU synthetic fixture. It cannot create real GPU acceptance."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from PIL import Image
from test_contracts import sample


class Pipeline(unittest.TestCase):
    def test_synthetic_fit_decode_evaluate_report(self):
        from proc_research.cache import write_sample,write_json,sha256
        from proc_research.protocol import create_run
        from proc_research.calibration.run import ROOT,load_config,code_identity,finish,fit_stage,decode_stage
        from proc_research.evaluate import evaluate_predictions
        from proc_research.report import build_report
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg=load_config(ROOT/'research/p01/config.json')
            ident={'config':cfg,'config_sha256':'a'*64,'manifest_sha256':'b'*64,
                'method_sha256':sha256(ROOT/'METHOD.md'),'code_sha256':code_identity(),
                'weight_sha256':'c'*64,'pool_sha256':{'LoveDA':'d'*64,'iSAID':'e'*64},
                'evidence_kind':'synthetic_cpu'}
            label_rows=[]
            for split in ('calibration','pilot'):
                run=create_run(p,split,dict(ident,stage=split));entries=[]
                for dataset in ('LoveDA','iSAID'):
                    for i in range(4):
                        for view in (1008,504):
                            s=sample(scene=f'{split}-s{i}',image=f'{split}-i{i}',view=view)
                            s.dataset=dataset
                            s.scores[1,:,i:i+2]=.2;s.scores[2,:,i:i+2]=.8
                            s.prompt_scores=s.scores.clone();s.prediction=s.scores.argmax(0)
                            name=f'cache/{dataset}-{i}-{view}';receipt=write_sample(run/name,s)
                            entries.append(dict(dataset=dataset,image_id=s.image_id,scene_id=s.scene_id,view=view,
                                                split=split,path=name,receipt=receipt))
                        if split=='pilot':
                            y=s.prediction.numpy().astype('uint8')+(1 if dataset=='LoveDA' else 0)
                            path=p/f'{dataset}-{i}-gt.png';Image.fromarray(y).save(path)
                            label_rows.append(dict(dataset=dataset,image_id=s.image_id,path=str(path),sha256=sha256(path)))
                write_json(run/'cache_index.json',{'entries':entries})
                finish(run,{'kind':'synthetic_cpu_fixture'})
            fit_stage(SimpleNamespace(cache_run=str(p/'calibration'),output=str(p),run_id='banks'))
            decode_stage(SimpleNamespace(cache_run=str(p/'pilot'),bank_run=str(p/'banks'),output=str(p),run_id='preds'))
            label_path=p/'labels.jsonl';label_path.write_text(''.join(json.dumps(x)+'\n' for x in label_rows))
            result=evaluate_predictions(p/'preds/prediction_index.json',label_path,p/'eval')
            self.assertEqual(len(result),22)
            report=build_report(p/'preds',p/'eval',p/'banks',p/'report')
            self.assertEqual(report['research_status'],'PENDING_SYNTHETIC_ONLY')
            self.assertEqual(len(report['comparisons']),20)
            self.assertTrue((p/'report/decision.md').exists())
    def test_bank_tamper_rejected(self):
        from proc_research.calibration.signatures import fit_signatures,save_bank,load_bank
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bank';bank=fit_signatures([sample(),sample(view=504)])
            receipt=save_bank(p,bank);load_bank(p,receipt)
            with open(p/'504.npz','ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):load_bank(p,receipt)
    def test_processor_real_cpu_transform_sizes(self):
        from sam3.model.sam3_image_processor import Sam3Processor
        from torchvision.transforms import v2
        for size in (504,1008):
            # Transform only: no model constructed, no encoder inference claimed.
            processor=Sam3Processor(None,resolution=size,device='cpu')
            out=processor.transform(v2.functional.to_image(Image.new('RGB',(123,37))))
            self.assertEqual(list(out.shape),[3,size,size]);self.assertEqual(out.dtype,torch.float32)


if __name__=='__main__':unittest.main()
