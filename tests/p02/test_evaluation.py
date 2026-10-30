"""Hand-computed CPU tests for the isolated P02 evaluator."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from PIL import Image


def table(image='i0', pixels=(0, 1), candidates=None):
    if candidates is None:
        candidates = [{'candidate_id':'C1', 'class_id':1, 'mask_changes':[[1,1]],
                       'region_changes':[[0,1],[1,1]]}]
    return {'dataset':'LoveDA','image_id':image,'shape':[1,2], 'class_ids':[0,1],
            'regions':[{'region_id':'R1','pixels':list(pixels),'candidates':candidates}],
            'majority_ids':['KEEP']}


class EvaluationContracts(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue((Path(__file__).resolve().parents[2]/'proc_research/correction/evaluation.py').is_file())

    def test_pooled_is_not_average_image_miou(self):
        from proc_research.correction.evaluation import pooled_metrics
        from proc_research.evaluate import metrics, confusion_matrix
        ys=[np.zeros((1,100),dtype=np.int64),np.ones((1,1),dtype=np.int64)]
        ps=[np.zeros_like(ys[0]),np.zeros_like(ys[1])]
        result=pooled_metrics(ys,ps,2)
        self.assertAlmostEqual(result['miou'],50/101)
        self.assertNotAlmostEqual(result['miou'],np.mean([metrics(confusion_matrix(y,p,2))['miou'] for y,p in zip(ys,ps)]))
        self.assertEqual(result['confusion_matrix'],[[100,0],[1,0]])

    def test_repair_damage_and_ignore_per_class(self):
        from proc_research.correction.evaluation import repair_damage
        r=repair_damage(np.array([[0,0,1,1,1]]),np.array([[1,0,0,1,0]]),np.array([[1,0,1,255,0]]),2)
        self.assertEqual((r['repaired'],r['damaged'],r['changed_valid']),(2,1,3))
        self.assertEqual(r['per_class'][0]['repaired'],1)
        self.assertEqual(r['per_class'][1]['damaged'],1)
        self.assertEqual(r['valid_pixels'],4)

    def test_bounded_oracle_matches_exhaustive_one_image(self):
        from proc_research.correction.evaluation import bounded_oracle
        from proc_research.correction.actions import apply_actions
        from proc_research.evaluate import confusion_matrix,metrics
        t=table(); y0=np.array([[0,0]],dtype=np.int64); gt=np.array([[0,1]],dtype=np.int64)
        item={'image_id':'i0','y0':y0,'table':t,'target':gt}
        for arm in ('MASK','REGION'):
            result=bounded_oracle([item],2,arm)
            scores=[metrics(confusion_matrix(gt,apply_actions(y0,t,[choice],arm),2))['miou'] for choice in ('KEEP','C1')]
            self.assertEqual(result['final']['miou'],max(scores))
            self.assertEqual(result['initial_choices'],{'i0':['KEEP']})
            self.assertLessEqual(result['sweeps'],3)
            for step in result['trace']:
                self.assertLessEqual(len(step['trials']),256)
                if step['accepted']:
                    self.assertGreater(step['after_miou'],step['before_miou'])

    def test_oracle_fixed_sort_and_no_acceptance_on_tie(self):
        from proc_research.correction.evaluation import bounded_oracle
        items=[]
        for image in ('z','a'):
            t=table(image); gt=np.array([[0,0]],dtype=np.int64)
            items.append({'image_id':image,'y0':gt.copy(),'table':t,'target':gt})
        r=bounded_oracle(items,2,'MASK')
        self.assertEqual([x['image_id'] for x in r['trace']],['a','z'])
        self.assertEqual(r['choices'],{'a':['KEEP'],'z':['KEEP']})
        self.assertEqual(r['sweeps'],1)

    def test_global_confusion_replacement_matches_trace(self):
        from proc_research.correction.evaluation import bounded_oracle
        from proc_research.correction.actions import apply_actions
        from proc_research.evaluate import metrics,confusion_matrix
        items=[{'image_id':i,'y0':np.array([[0,0]],dtype=np.int64),'table':table(i),
                'target':np.array([[0,1] if i=='a' else [1,1]],dtype=np.int64)} for i in ('b','a')]
        for arm in ('MASK','REGION'):
            r=bounded_oracle(items,2,arm)
            cm=sum((confusion_matrix(it['target'],apply_actions(it['y0'],it['table'],r['choices'][it['image_id']],arm),2) for it in items),np.zeros((2,2),dtype=np.int64))
            self.assertEqual(r['final'],metrics(cm))
            self.assertLessEqual(r['evaluations'],3*2*2)
            self.assertEqual(r['budget'],{'maximum_sweeps':3,'maximum_joint_actions_per_image':256})

    def test_global_objective_accepts_action_that_hurts_image_miou(self):
        from proc_research.correction.evaluation import bounded_oracle
        from proc_research.correction.actions import apply_actions
        from proc_research.evaluate import metrics,confusion_matrix
        small=table('a', pixels=(0,1,2), candidates=[{'candidate_id':'C1','class_id':1,
            'mask_changes':[[0,1],[1,1],[2,1]],'region_changes':[[0,1],[1,1],[2,1]]}])
        small['shape']=[1,4]
        big={'dataset':'LoveDA','image_id':'b','shape':[1,100],'class_ids':[0,1],
             'regions':[],'majority_ids':[]}
        y0=np.zeros((1,4),dtype=np.int64);target=np.array([[0,0,1,1]],dtype=np.int64)
        changed=apply_actions(y0,small,['C1'],'MASK')
        self.assertLess(metrics(confusion_matrix(target,changed,2))['miou'],
                        metrics(confusion_matrix(target,y0,2))['miou'])
        items=[{'image_id':'a','y0':y0,'table':small,'target':target},
               {'image_id':'b','y0':np.zeros((1,100),dtype=np.int64),'table':big,
                'target':np.zeros((1,100),dtype=np.int64)}]
        result=bounded_oracle(items,2,'MASK')
        self.assertEqual(result['choices'],{'a':['C1'],'b':[]})
        self.assertEqual(result['final']['confusion_matrix'],[[100,2],[1,1]])
        self.assertGreater(result['final']['miou'],result['initial']['miou'])

    def test_four_regions_exhaustive_256_legal_choices(self):
        from proc_research.correction.evaluation import bounded_oracle
        from proc_research.correction.actions import apply_actions
        from proc_research.evaluate import metrics,confusion_matrix
        y0=np.zeros((1,8),dtype=np.int64); target=np.array([[1,1,2,2,3,3,0,0]],dtype=np.int64)
        regions=[]
        for i in range(4):
            pixels=[2*i,2*i+1]
            candidates=[{'candidate_id':'C'+str(c),'class_id':c,
                         'mask_changes':[[p,c] for p in pixels],
                         'region_changes':[[p,c] for p in pixels]} for c in (1,2,3)]
            regions.append({'region_id':'R'+str(i+1),'pixels':pixels,'candidates':candidates})
        t={'dataset':'LoveDA','image_id':'x','shape':[1,8],'class_ids':[0,1,2,3],
           'regions':regions,'majority_ids':['KEEP']*4}
        scores=[metrics(confusion_matrix(target,apply_actions(y0,t,list(choice),'MASK'),4))['miou']
                for choice in itertools.product(('KEEP','C1','C2','C3'),repeat=4)]
        r=bounded_oracle([{'image_id':'x','y0':y0,'table':t,'target':target}],4,'MASK')
        self.assertEqual(len(r['trace'][0]['trials']),256)
        self.assertEqual(r['final']['miou'],max(scores))
        self.assertEqual(r['final']['miou'],1.0)
        self.assertEqual(r['choices']['x'],['C1','C2','C3','KEEP'])

    def test_action_budget_rejected_not_truncated(self):
        from proc_research.correction.evaluation import bounded_oracle
        t=table();t['regions']=t['regions']*5
        with self.assertRaisesRegex(ValueError,'region budget'):
            bounded_oracle([{'image_id':'i0','y0':np.zeros((1,2),dtype=np.int64),'table':t,'target':np.zeros((1,2),dtype=np.int64)}],2,'MASK')

    def test_oracle_rejects_empty_region(self):
        from proc_research.correction.evaluation import bounded_oracle
        t=table(pixels=(),candidates=[])
        item={'image_id':'i0','y0':np.array([[0,0]]),'table':t,'target':np.array([[0,0]])}
        with self.assertRaisesRegex(ValueError,'empty region'):
            bounded_oracle([item],2,'MASK')

    def test_oracle_rejects_duplicate_region_ids_even_when_disjoint(self):
        from proc_research.correction.evaluation import bounded_oracle
        t=table(pixels=(0,),candidates=[])
        second=copy.deepcopy(t['regions'][0]);second['pixels']=[1]
        t['regions'].append(second)
        item={'image_id':'i0','y0':np.array([[0,0]]),'table':t,'target':np.array([[0,0]])}
        with self.assertRaisesRegex(ValueError,'region ID'):
            bounded_oracle([item],2,'MASK')

    def test_oracle_rejects_candidate_outside_table_vocabulary(self):
        from proc_research.correction.evaluation import bounded_oracle
        t=table();t['class_ids']=[0]
        item={'image_id':'i0','y0':np.array([[0,0]]),'table':t,'target':np.array([[0,1]])}
        with self.assertRaisesRegex(ValueError,'table vocabulary'):
            bounded_oracle([item],2,'MASK')

    def test_oracle_rejects_baseline_outside_table_vocabulary(self):
        from proc_research.correction.evaluation import bounded_oracle
        t=table();t['class_ids']=[1]
        item={'image_id':'i0','y0':np.array([[0,0]]),'table':t,'target':np.array([[0,1]])}
        with self.assertRaisesRegex(ValueError,'table vocabulary'):
            bounded_oracle([item],2,'MASK')

    def test_labels_mapping_binding_and_missing_dataset_not_cherry_picked(self):
        from proc_research.correction.evaluation import evaluate_dataset
        samples=[SimpleNamespace(dataset='LoveDA',image_id=i,scene_id='',prediction=np.array([[0,0]],dtype=np.int64),class_ids=(0,1),provenance={'image_sha256':'a'*64}) for i in ('i0','i1')]
        tables=[table(i) for i in ('i0','i1')]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);im=p/'gt.png';Image.fromarray(np.array([[1,2]],dtype=np.uint8)).save(im)
            row={'dataset':'LoveDA','image_id':'i0','path':str(im),'sha256':hashlib.sha256(im.read_bytes()).hexdigest(),'image_sha256':'a'*64}
            manifest=p/'labels.jsonl';manifest.write_text(json.dumps(row)+'\n')
            r=evaluate_dataset(samples,tables,manifest,kind='synthetic')
            self.assertEqual(r['LoveDA']['status'],'PENDING_GT')
            self.assertNotIn('oracle',r['LoveDA'])
            r=evaluate_dataset(samples[:1],tables[:1],manifest,kind='synthetic')
            self.assertEqual(r['LoveDA']['kind'],'synthetic')
            self.assertEqual(r['LoveDA']['changes']['ORACLE_MASK']['repaired'],1)
            self.assertEqual(r['LoveDA']['coverable_errors']['MASK'],1)
            self.assertEqual(r['LoveDA']['per_image'][0]['region_effects'][0]['methods']['ORACLE_MASK']['repaired'],1)
            self.assertEqual(r['LoveDA']['methods']['BASE']['confusion_matrix'][0][:2],[1,0])
            self.assertEqual(r['LoveDA']['methods']['BASE']['confusion_matrix'][1][:2],[1,0])
            row['image_sha256']='b'*64;manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'image association'):
                evaluate_dataset(samples[:1],tables[:1],manifest)

    def test_gt_none_and_duplicate_manifest(self):
        from proc_research.correction.evaluation import evaluate_dataset
        sample=SimpleNamespace(dataset='iSAID',image_id='i0',scene_id='',prediction=np.array([[0,0]],dtype=np.int64),class_ids=(0,1),provenance={'image_sha256':'a'*64})
        t=table();t['dataset']='iSAID'
        self.assertEqual(evaluate_dataset([sample],[t])['iSAID']['status'],'PENDING_GT')
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'labels.jsonl';row={'dataset':'iSAID','image_id':'i0','path':'missing','sha256':'a'*64,'image_sha256':'a'*64}
            manifest.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                evaluate_dataset([sample],[t],manifest)

    def test_label_corruption_fails(self):
        from proc_research.correction.evaluation import evaluate_dataset
        sample=SimpleNamespace(dataset='LoveDA',image_id='i0',scene_id='',prediction=np.array([[0,0]],dtype=np.int64),class_ids=(0,1),provenance={'image_sha256':'a'*64})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);im=p/'gt.png';Image.fromarray(np.array([[1,2]],dtype=np.uint8)).save(im)
            row={'dataset':'LoveDA','image_id':'i0','path':str(im),'sha256':'b'*64,'image_sha256':'a'*64}
            manifest=p/'labels.jsonl';manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'hash'):
                evaluate_dataset([sample],[table()],manifest)

    def test_known_dataset_maps(self):
        from proc_research.correction.evaluation import map_labels
        self.assertEqual(map_labels(np.array([[0,1,7,255]]),'LoveDA').tolist(),[[255,0,6,255]])
        self.assertEqual(map_labels(np.array([[0,1,15,255]]),'iSAID').tolist(),[[0,1,15,255]])

if __name__=='__main__':unittest.main()
