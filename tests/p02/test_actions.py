"""Hand-computed GT-free action-space contracts; no model execution."""
import copy
import json
from types import SimpleNamespace
import unittest
import numpy as np

from proc_research.correction.actions import (
    apply_actions, build_actions, construct_region, trigger_regions, summarize_tables,
)


def sample(y0, prompts=None, mapping=None, scores=None, class_ids=(0, 1)):
    y0 = np.asarray(y0, dtype=np.int64)
    if prompts is None:
        prompts = np.zeros((0, *y0.shape), dtype=np.float32)
    if mapping is None:
        mapping = np.zeros(len(prompts), dtype=np.int64)
    if scores is None:
        scores = np.zeros((len(class_ids), *y0.shape), dtype=np.float32)
    return SimpleNamespace(dataset='LoveDA', image_id='synthetic-hand',
        prediction=y0, prompt_scores=np.asarray(prompts,dtype=np.float32),
        prompt_to_class=np.asarray(mapping,dtype=np.int64), class_ids=class_ids,
        scores=np.asarray(scores,dtype=np.float32), view=1008,
        geometry={'kind':'full_resize','original_hw':list(y0.shape)}, provenance={})


def table(y0, regions, classes=(0,1)):
    return {'shape':list(np.asarray(y0).shape),'class_ids':list(classes),
            'regions':regions,'majority_ids':[r['majority_id'] for r in regions]}


class ActionContracts(unittest.TestCase):
    def region(self, y, p, m, ids=None, pixels=None, classes=(0,1)):
        y=np.asarray(y,dtype=np.int64)
        return construct_region(y,np.asarray(p,dtype=np.float32),np.asarray(m,dtype=np.int64),
                                classes,list(range(y.size)) if pixels is None else pixels,
                                prompt_ids=ids)

    def test_empty_intersection_is_mask_keep_but_region_action(self):
        r=self.region([[0,0]], [[[.1,.2]]], [1])
        c=r['candidates'][0]
        self.assertEqual(c['mask_changes'],[])
        self.assertEqual(c['region_changes'],[[0,1],[1,1]])
        self.assertEqual(r['stats']['mask_effective_actions'],0)
        self.assertEqual(r['stats']['region_unique_outputs'],1)

    def test_same_class_different_masks_same_actual_output_deduplicated(self):
        r=self.region([[1,0]], [[[.9,.8]],[[.2,.8]]], [1,1], ['a','b'])
        self.assertEqual(len(r['candidates']),1)
        self.assertEqual(r['candidates'][0]['aliases'],['a','b'])
        self.assertEqual(r['candidates'][0]['mask_changes'],[[1,1]])
        self.assertEqual(r['stats']['pair_duplicates'],1)
        self.assertEqual(r['stats']['mask_raw_duplicates'],1)
        self.assertEqual(r['stats']['region_raw_duplicates'],1)
        self.assertEqual(r['stats']['mask_raw_duplicate_rate'],.5)

    def test_different_classes_same_mask_different_outputs(self):
        r=self.region([[0,0]], [[[.8,.8]],[[.8,.8]]], [1,2], classes=(0,1,2))
        self.assertEqual([x['class_id'] for x in r['candidates']],[1,2])
        self.assertEqual(r['stats']['mask_effective_actions'],2)

    def test_full_mask_and_region_outputs_coincide(self):
        r=self.region([[0,1]], [[[.5,.6]]], [1])
        self.assertEqual(r['candidates'][0]['mask_changes'],r['candidates'][0]['region_changes'])
        self.assertEqual(r['stats']['mask_unique_outputs'],0)
        self.assertEqual(r['stats']['region_unique_outputs'],0)
        self.assertEqual(r['stats']['raw_full_region_masks'],1)
        self.assertEqual(r['stats']['response_threshold_equal_pixels'],1)

    def test_all_region_outputs_used_for_novelty(self):
        # C1's MASK equals KEEP; C2's MASK equals REGION's full output.
        r=self.region([[0,1]], [[[.9,.1]],[[.4,.8]]], [0,0])
        self.assertEqual(r['stats']['mask_unique_outputs'],0)
        self.assertEqual(r['stats']['region_unique_outputs'],0)

    def test_double_keep_does_not_take_slot_then_pair_dedup_then_truncate(self):
        r=self.region([[0,0]], [[[1,1]],[[.95,.9]],[[.94,.9]],[[.8,.7]],[[.7,.7]],[[.6,.6]]],
                      [0,1,1,2,3,4],['keep','one','alias','two','three','four'],classes=(0,1,2,3,4))
        self.assertEqual([c['class_id'] for c in r['candidates']],[1,2,3])
        self.assertEqual(r['candidates'][0]['aliases'],['one','alias'])
        self.assertEqual(r['stats']['double_keep_candidates'],1)
        self.assertEqual(r['stats']['truncated_candidates'],1)
        self.assertEqual(r['stats']['raw_candidates'],6)

    def test_majority_selects_shared_id_even_when_mask_noop(self):
        r=self.region([[1,1,0]], [[[.9,.8,.1]],[[.8,.8,.8]]], [1,0])
        self.assertEqual(r['majority_id'],'C2')
        t=table([[1,1,0]],[r]); y=np.array([[1,1,0]])
        np.testing.assert_array_equal(apply_actions(y,t,t['majority_ids'],'MASK'),y)
        np.testing.assert_array_equal(apply_actions(y,t,t['majority_ids'],'REGION'),[[1,1,1]])

    def test_exact_half_and_absent_majority_candidate_keep(self):
        self.assertEqual(self.region([[0,1]],[[[.9,.9]]],[1])['majority_id'],'KEEP')
        self.assertEqual(self.region([[0,0,1]],[[[.9,.9,.9]]],[1])['majority_id'],'KEEP')

    def test_candidate_ties_class_then_prompt_id(self):
        r=self.region([[0,0]], [[[.8,.2]],[[.2,.8]],[[.8,.2]]], [2,1,1],['z','b','a'],classes=(0,1,2))
        self.assertEqual([(c['class_id'],c['prompt_id']) for c in r['candidates']],[(1,'a'),(1,'b'),(2,'z')])

    def test_zero_candidates(self):
        s=sample([[0,1]])
        a=build_actions(s)
        self.assertEqual(a['regions'][0]['candidates'],[])
        self.assertEqual(a['majority_ids'],['KEEP'])
        self.assertEqual(a['stats']['zero_candidate_region_rate'],1.0)
        np.testing.assert_array_equal(apply_actions(s.prediction,a,['KEEP'],'MASK'),s.prediction)

    def test_trigger_ties_coordinates_and_eight_connectivity(self):
        scores=np.zeros((2,4,10),dtype=np.float32)
        regions,stats=trigger_regions(scores)
        self.assertEqual(regions,[[0,1]])
        self.assertEqual(stats['trigger_pixels'],2)
        self.assertEqual(stats['margin_threshold_equal_pixels'],40)
        self.assertEqual(stats['margin_threshold_selected_equal_pixels'],2)
        scores[1]=1
        scores[1,0,0]=0; scores[1,1,1]=0
        regions,_=trigger_regions(scores)
        self.assertEqual(regions,[[0,11]])

    def test_components_area_order_and_four_region_budget(self):
        scores=np.ones((2,10,20),dtype=np.float32); scores[0]=0
        chosen=[0,1,2,4,6,8,10,12,14,16]
        scores[1].reshape(-1)[chosen]=0
        regions,stats=trigger_regions(scores)
        self.assertEqual(regions,[[0,1,2],[4],[6],[8]])
        self.assertEqual(stats['component_count'],8)
        self.assertEqual(stats['selected_region_pixels'],6)

    def test_non_square_semantic_class_mapping(self):
        s=sample([[0,0,1],[0,1,1]],[[[.8]*3]*2],[1],class_ids=(3,8))
        a=build_actions(s)
        self.assertEqual(a['regions'][0]['candidates'][0]['class_id'],8)
        self.assertEqual(a['regions'][0]['candidates'][0]['mask_changes'],[[0,8]])
        result=apply_actions(np.array(s.class_ids)[s.prediction],a,['C1'],'MASK')
        np.testing.assert_array_equal(result,[[8,3,8],[3,8,8]])

    def test_cache_channel_prediction_maps_reordered_vocabulary(self):
        s=sample([[0,1]],[[[.9,.9]]],[1],class_ids=(1,0))
        a=build_actions(s)
        self.assertEqual(a['regions'][0]['candidates'][0]['class_id'],0)
        self.assertEqual(a['regions'][0]['candidates'][0]['mask_changes'],[[0,0]])
        np.testing.assert_array_equal(apply_actions(np.array([[1,0]]),a,['C1'],'MASK'),[[0,0]])

    def test_mutually_exclusive_regions_and_out_of_range_rejected(self):
        r=self.region([[0,0,0]],[[[.9,.9,.9]]],[1],pixels=[0])
        t=table([[0,0,0]],[r,r])
        with self.assertRaises(ValueError): apply_actions(np.array([[0,0,0]]),t,['C1','C1'],'MASK')
        r['candidates'][0]['mask_changes']=[[1,1]]
        with self.assertRaises(ValueError): apply_actions(np.array([[0,0,0]]),table([[0,0,0]],[r]),['C1'],'MASK')
        with self.assertRaises(ValueError): self.region([[0]],[[[.9]]],[1],pixels=[1])

    def test_apply_rejects_bad_arm_choice_shape_class_and_duplicate_pixel(self):
        r=self.region([[0,0]],[[[.9,.9]]],[1]);t=table([[0,0]],[r]);y=np.array([[0,0]])
        for choice,arm in [(['C9'],'MASK'),(['C1'],'X'),([], 'MASK')]:
            with self.assertRaises(ValueError): apply_actions(y,t,choice,arm)
        with self.assertRaises(ValueError): apply_actions(y.T,t,['C1'],'MASK')
        for changes in ([[0,9]],[[0,1],[0,1]],[[-1,1]]):
            bad=copy.deepcopy(t);bad['regions'][0]['candidates'][0]['mask_changes']=changes
            with self.assertRaises(ValueError): apply_actions(y,bad,['C1'],'MASK')

    def test_gt_removed_or_replaced_produces_identical_bytes(self):
        s=sample([[0,1]],[[[.9,.9]]],[1]);s.gt=np.array([[255,255]])
        before=json.dumps(build_actions(s),sort_keys=True,separators=(',',':')).encode()
        s.gt=np.array([[1,0]])
        self.assertEqual(before,json.dumps(build_actions(s),sort_keys=True,separators=(',',':')).encode())
        del s.gt
        self.assertEqual(before,json.dumps(build_actions(s),sort_keys=True,separators=(',',':')).encode())

    def test_summary_sums_counts_not_image_rates_and_keeps_datasets_separate(self):
        first=sample([[0]],[[[.9]]],[1]);first.image_id='a'
        second=sample(np.zeros((1,99),dtype=np.int64),np.full((1,1,99),.1,dtype=np.float32),[1]);second.image_id='b'
        third=sample([[0]],[[[.9]]],[1]);third.dataset='iSAID'
        summary=summarize_tables([build_actions(first),build_actions(second),build_actions(third)])
        loveda=summary['LoveDA']
        self.assertEqual(loveda['images'],2)
        self.assertEqual(loveda['geometric_claim'],'NO_NEW_MASK_OUTPUTS_IN_SELECTED_SAMPLE')
        self.assertEqual(loveda['counts']['geometric_valid_pixels'],100)
        self.assertEqual(loveda['counts']['raw_candidates'],2)
        self.assertEqual(loveda['counts']['trigger_pixels'],6)
        self.assertEqual(loveda['rates']['trigger_coverage'],{'numerator':6,'denominator':100,'value':.06})
        self.assertEqual(loveda['rates']['mask_modifiable_image_coverage']['value'],.01)
        self.assertEqual(loveda['rates']['region_modifiable_image_coverage']['value'],.06)
        self.assertEqual(loveda['rates']['mask_raw_noop_rate']['value'],.5)
        self.assertEqual(loveda['selector_summary']['majority/MASK']['all_keep_image_rate']['value'],1.)
        self.assertEqual(summary['iSAID']['images'],1)
        self.assertEqual(len(loveda['per_image_stats']),2)
        self.assertEqual(len(loveda['per_region_stats']),2)

    def test_summary_new_geometry_is_not_effectiveness(self):
        s=sample(np.zeros((1,40),dtype=np.int64),np.concatenate((np.array([.9,.1]),np.zeros(38)))[None,None,:],[1])
        summary=summarize_tables([build_actions(s)])['LoveDA']
        self.assertEqual(summary['geometric_claim'],'NEW_MASK_OUTPUTS_OBSERVED_NOT_EFFECTIVENESS')
        self.assertEqual(summary['counts']['mask_unique_outputs'],1)

    def test_bad_geometries_response_and_mapping_fail(self):
        for change in [dict(view=504),dict(geometry={'kind':'crop','original_hw':[1,2]}),
                       dict(prompt_scores=np.ones((1,2,1),dtype=np.float32)),
                       dict(prompt_to_class=np.array([2])),dict(scores=np.full((2,1,2),np.nan))]:
            s=sample([[0,1]],[[[.9,.9]]],[1]);s.__dict__.update(change)
            with self.assertRaises(ValueError): build_actions(s)


if __name__=='__main__': unittest.main()
