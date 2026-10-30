"""Deterministic GT-free P02 action geometry on the original image grid.

This module uses NumPy only. It neither imports an evaluator nor reads labels,
files, model code or devices. Output signatures contain actual changed labels.
"""
import math
import numpy as np

KEEP = 'KEEP'


def _array(value):
    # Existing caches hold CPU tensors; refuse a device transfer implicitly.
    if hasattr(value, 'device') and getattr(value.device, 'type', 'cpu') != 'cpu':
        raise ValueError('P02 accepts CPU arrays only')
    if hasattr(value, 'detach'):
        value = value.detach().numpy()
    return np.asarray(value)


def _integer_array(value, name):
    a = _array(value)
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f'{name} must contain integers')
    return a


def _responses(value, name):
    a = _array(value)
    if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain finite floating responses')
    if ((a < 0) | (a > 1)).any():
        raise ValueError(f'{name} outside [0,1]')
    return a


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def trigger_regions(scores):
    """Return selected disjoint flat-pixel components and boundary-tie counts."""
    scores = _responses(scores, 'class scores')
    if scores.ndim != 3 or scores.shape[0] < 2:
        raise ValueError('at least two class response maps are required')
    _, height, width = scores.shape
    total = height * width
    top = np.partition(scores, -2, axis=0)[-2:]
    margin = (top[1].astype(np.float64) - top[0].astype(np.float64)).reshape(-1)
    take = int(math.ceil(total * .05))
    # Stable sort retains flat row-major coordinates on every margin tie.
    chosen = np.argsort(margin, kind='stable')[:take]
    threshold = float(margin[chosen[-1]]) if take else None
    remaining = set(int(p) for p in chosen)
    components = []
    for start in sorted(remaining):
        if start not in remaining:
            continue
        remaining.remove(start)
        todo = [start]
        component = []
        while todo:
            pixel = todo.pop()
            component.append(pixel)
            row, col = divmod(pixel, width)
            for rr in range(max(0, row-1), min(height, row+2)):
                for cc in range(max(0, col-1), min(width, col+2)):
                    neighbor = rr*width+cc
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        todo.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda c: (-len(c), c[0]))
    selected = components[:4]
    flattened = [p for c in selected for p in c]
    if len(flattened) != len(set(flattened)):
        raise ValueError('trigger regions overlap')
    return selected, {
        'geometric_valid_pixels': total, 'trigger_pixels': take,
        'trigger_coverage': _ratio(take, total),
        'margin_threshold': threshold,
        'margin_threshold_equal_pixels': int(np.count_nonzero(margin == threshold)) if take else 0,
        'margin_threshold_selected_equal_pixels': int(np.count_nonzero(margin[chosen] == threshold)) if take else 0,
        'component_count': len(components), 'selected_regions': len(selected),
        'selected_region_pixels': len(flattened),
        'selected_region_coverage': _ratio(len(flattened), total),
    }


def _signature(pixels, labels, class_id, covered):
    changed = pixels[covered & (labels != class_id)]
    return tuple((int(p), int(class_id)) for p in changed)


def construct_region(y0, prompt_scores, prompt_to_class, class_ids, pixels,
                     region_id='R1', prompt_ids=None, prompt_sources=None):
    """Construct one shared table; callers may use this for hand calculations."""
    y0 = _integer_array(y0, 'prediction')
    prompts = _responses(prompt_scores, 'prompt responses')
    mapping = _integer_array(prompt_to_class, 'prompt mapping')
    classes = _integer_array(class_ids, 'class IDs')
    raw_pixels = _integer_array(pixels, 'region pixels') if len(pixels) else np.array([], dtype=np.int64)
    if y0.ndim != 2 or raw_pixels.ndim != 1 or not len(raw_pixels):
        raise ValueError('region must be nonempty on an HxW grid')
    if len(set(raw_pixels.tolist())) != len(raw_pixels) or ((raw_pixels < 0) | (raw_pixels >= y0.size)).any():
        raise ValueError('region pixels duplicate or outside image')
    pixels = np.sort(raw_pixels)
    if classes.ndim != 1 or len(classes) < 2 or len(set(classes.tolist())) != len(classes) or (classes < 0).any():
        raise ValueError('invalid class vocabulary')
    if not np.isin(y0, classes).all():
        raise ValueError('prediction outside semantic class vocabulary')
    if prompts.ndim != 3 or tuple(prompts.shape[1:]) != y0.shape or mapping.shape != (len(prompts),):
        raise ValueError('prompt response geometry/mapping mismatch')
    if ((mapping < 0) | (mapping >= len(classes))).any():
        raise ValueError('prompt mapping outside score channels')
    count = len(prompts)
    prompt_ids = [str(i) for i in range(count)] if prompt_ids is None else list(prompt_ids)
    if len(prompt_ids) != count or any(not isinstance(p, str) or not p for p in prompt_ids) or len(set(prompt_ids)) != count:
        raise ValueError('prompt IDs must be unique nonempty strings')
    prompt_sources = prompt_ids if prompt_sources is None else list(prompt_sources)
    if len(prompt_sources) != count:
        raise ValueError('prompt sources mismatch')
    labels = y0.reshape(-1)[pixels]
    local = prompts.reshape(count, y0.size)[:, pixels]
    raw = []
    for index in range(count):
        class_id = int(classes[mapping[index]])
        covered = local[index] >= .5
        mask = _signature(pixels, labels, class_id, covered)
        region = _signature(pixels, labels, class_id, np.ones(len(pixels), dtype=bool))
        raw.append({
            'class_id': class_id, 'prompt_id': prompt_ids[index],
            'prompt_source': prompt_sources[index],
            'mean_response': float(local[index].mean(dtype=np.float64)),
            'mask_signature': mask, 'region_signature': region,
            'covered_pixels': int(covered.sum()),
        })
    raw.sort(key=lambda item: (-item['mean_response'], item['class_id'], item['prompt_id']))
    pairs = {}
    double_keep = 0
    duplicate_count = 0
    for row in raw:
        pair = (row['mask_signature'], row['region_signature'])
        if pair == ((), ()):
            double_keep += 1
            continue
        if pair in pairs:
            pairs[pair]['aliases'].append(row['prompt_id'])
            pairs[pair]['source_aliases'].append({'prompt_id':row['prompt_id'], 'source':row['prompt_source']})
            duplicate_count += 1
        else:
            pairs[pair] = dict(row, aliases=[row['prompt_id']],
                              source_aliases=[{'prompt_id':row['prompt_id'], 'source':row['prompt_source']}])
    retained = list(pairs.values())[:3]
    candidates = []
    for index, row in enumerate(retained):
        candidates.append({
            'candidate_id': f'C{index+1}', 'class_id': row['class_id'],
            'prompt_id': row['prompt_id'], 'prompt_source':row['prompt_source'],
            'aliases':row['aliases'], 'source_aliases':row['source_aliases'],
            'mean_response':row['mean_response'], 'covered_pixels':row['covered_pixels'],
            'mask_changes':[list(pair) for pair in row['mask_signature']],
            'region_changes':[list(pair) for pair in row['region_signature']],
        })
    mask_set = {row['mask_signature'] for row in retained} - {()}
    region_set = {row['region_signature'] for row in retained} - {()}
    mask_novel = mask_set - region_set
    region_novel = region_set - mask_set
    majority = KEEP
    shares = {int(c): float(np.count_nonzero(labels == c)/len(pixels)) for c in np.unique(labels)}
    # At most one class can have > half of E. Candidate rank resolves its prompts.
    ordered = sorted(enumerate(candidates), key=lambda pair: (-shares.get(pair[1]['class_id'], 0), pair[0]))
    if ordered and shares.get(ordered[0][1]['class_id'], 0) > .5:
        majority = ordered[0][1]['candidate_id']
    stats = {
        'region_pixels':len(pixels), 'raw_candidates':count,
        'double_keep_candidates':double_keep, 'pair_duplicates':duplicate_count,
        'unique_pair_candidates':len(pairs), 'truncated_candidates':max(0,len(pairs)-3),
        'shared_candidates':len(candidates),
        'zero_candidates':int(not candidates),
        'raw_all_low_response_candidates':int(np.count_nonzero(np.all(local < .5, axis=1))),
        'raw_full_region_masks':int(np.count_nonzero(np.all(local >= .5, axis=1))),
        'all_prompt_responses_low':bool(count and np.all(local < .5)),
        'all_prompt_masks_cover_region':bool(count and np.all(local >= .5)),
        'response_threshold_equal_pixels':int(np.count_nonzero(local == .5)),
        'prompt_region_pixel_pairs':count*len(pixels),
        'double_keep_rate':_ratio(double_keep,count),
        'pair_duplicate_rate':_ratio(duplicate_count,count-double_keep),
        'selected_full_region_masks':sum(row['covered_pixels']==len(pixels) for row in retained),
        'mask_equals_region_outputs':sum(row['mask_signature']==row['region_signature'] for row in retained),
        'mask_equals_region_rate':_ratio(sum(row['mask_signature']==row['region_signature'] for row in retained),len(retained)),
        'majority_keep':int(majority==KEEP),
    }
    for arm, signatures, other, novel in [('mask',mask_set,region_set,mask_novel),('region',region_set,mask_set,region_novel)]:
        key = arm+'_signature'
        selected_noop = sum(not row[key] for row in retained)
        raw_noop = sum(not row[key] for row in raw)
        raw_effective = len({row[key] for row in raw} - {()})
        raw_duplicates = count - raw_noop - raw_effective
        changed_pixels = {p for signature in signatures for p, _ in signature}
        chosen = next((row for row in candidates if row['candidate_id']==majority), None)
        majority_changes = chosen[arm+'_changes'] if chosen else []
        stats.update({
            arm+'_raw_noop_candidates':raw_noop,
            arm+'_raw_noop_rate':_ratio(raw_noop,count),
            arm+'_raw_effective_actions':raw_effective,
            arm+'_raw_duplicates':raw_duplicates,
            arm+'_raw_duplicate_rate':_ratio(raw_duplicates,count-raw_noop),
            arm+'_effective_actions':len(signatures),
            arm+'_selected_noop_candidates':selected_noop,
            arm+'_selected_noop_rate':_ratio(selected_noop,len(retained)),
            arm+'_selected_duplicates':len(retained)-selected_noop-len(signatures),
            arm+'_selected_duplicate_rate':_ratio(len(retained)-selected_noop-len(signatures),len(retained)-selected_noop),
            arm+'_unique_outputs':len(novel),
            arm+'_unique_output_rate':_ratio(len(novel),len(signatures)),
            arm+'_modifiable_pixels':len(changed_pixels),
            arm+'_modifiable_region_coverage':_ratio(len(changed_pixels),len(pixels)),
            arm+'_majority_noop':int(not majority_changes),
            arm+'_majority_changed_pixels':len(majority_changes),
            arm+'_majority_region_coverage':_ratio(len(majority_changes),len(pixels)),
        })
    return {'region_id':region_id, 'pixels':[int(p) for p in pixels],
            'candidates':candidates, 'majority_id':majority,
            'class_shares':{str(k):v for k,v in sorted(shares.items())}, 'stats':stats}


def build_actions(sample):
    """Read only cache response/prediction metadata; never inspect GT fields."""
    scores = _responses(sample.scores, 'class scores')
    y0 = _integer_array(sample.prediction, 'prediction')
    if scores.ndim != 3 or y0.ndim != 2 or tuple(scores.shape[1:]) != y0.shape:
        raise ValueError('class score/original prediction geometry mismatch')
    if sample.view != 1008 or sample.geometry.get('kind') != 'full_resize' or sample.geometry.get('original_hw') != list(y0.shape):
        raise ValueError('P02 requires single 1008 full_resize original-grid cache')
    classes = _integer_array(sample.class_ids, 'class IDs')
    if classes.ndim != 1 or len(classes) != scores.shape[0] or len(set(classes.tolist())) != len(classes) or (classes < 0).any():
        raise ValueError('class vocabulary and scores mismatch')
    if ((y0 < 0) | (y0 >= len(classes))).any():
        raise ValueError('cached prediction outside score channels')
    # SampleCache predictions index score channels; candidate outputs use the
    # semantic IDs carried by that channel vocabulary, including reordered IDs.
    y0 = classes[y0]
    prompts = _responses(sample.prompt_scores, 'prompt responses')
    mapping = _integer_array(sample.prompt_to_class, 'prompt mapping')
    if prompts.ndim != 3 or tuple(prompts.shape[1:]) != y0.shape or mapping.shape != (len(prompts),):
        raise ValueError('prompt response geometry/mapping mismatch')
    if ((mapping<0)|(mapping>=len(sample.class_ids))).any():
        raise ValueError('prompt mapping outside score channels')
    pixels, trigger_stats = trigger_regions(scores)
    provenance = getattr(sample,'provenance',{})
    prompt_ids = provenance.get('prompt_ids')
    prompt_sources = provenance.get('prompts')
    regions = [construct_region(y0,prompts,mapping,sample.class_ids,p,
        region_id=f'R{index+1}',prompt_ids=prompt_ids,prompt_sources=prompt_sources)
        for index,p in enumerate(pixels)]
    stats = dict(trigger_stats)
    count_keys = ('raw_candidates','double_keep_candidates','pair_duplicates','unique_pair_candidates',
        'truncated_candidates','shared_candidates','zero_candidates','raw_all_low_response_candidates',
        'raw_full_region_masks','response_threshold_equal_pixels','prompt_region_pixel_pairs',
        'selected_full_region_masks','mask_equals_region_outputs','majority_keep')
    arm_keys = ('raw_noop_candidates','raw_effective_actions','raw_duplicates','effective_actions','selected_noop_candidates','selected_duplicates',
                'unique_outputs','modifiable_pixels','majority_noop','majority_changed_pixels')
    for key in count_keys + tuple(arm+'_'+suffix for arm in ('mask','region') for suffix in arm_keys):
        stats[key] = sum(region['stats'][key] for region in regions)
    stats.update({'zero_candidate_region_rate':_ratio(stats['zero_candidates'],len(regions)),
                  'majority_all_keep':all(r['majority_id']==KEEP for r in regions),
                  'all_low_response_regions':sum(r['stats']['all_prompt_responses_low'] for r in regions),
                  'all_cover_region_regions':sum(r['stats']['all_prompt_masks_cover_region'] for r in regions)})
    for arm in ('mask','region'):
        stats[arm+'_modifiable_image_coverage'] = _ratio(stats[arm+'_modifiable_pixels'],y0.size)
        stats[arm+'_majority_image_coverage'] = _ratio(stats[arm+'_majority_changed_pixels'],y0.size)
        stats[arm+'_majority_all_noop'] = stats[arm+'_majority_changed_pixels']==0
    return {'schema':1,'dataset':sample.dataset,'image_id':sample.image_id,
            'shape':list(y0.shape),'class_ids':[int(c) for c in sample.class_ids],
            'regions':regions,'majority_ids':[r['majority_id'] for r in regions],'stats':stats}


def apply_actions(y0, action_table, choices, arm):
    """Apply shared IDs only after validating every write and disjoint region."""
    y0 = _integer_array(y0,'prediction')
    if arm not in ('MASK','REGION'):
        raise ValueError('unknown action arm')
    if y0.ndim != 2 or list(y0.shape) != action_table['shape']:
        raise ValueError('writeback shape mismatch')
    classes = set(action_table['class_ids'])
    if not np.isin(y0,list(classes)).all():
        raise ValueError('original labels outside class vocabulary')
    regions = action_table['regions']
    if len(regions)>4:
        raise ValueError('region budget exceeded')
    if isinstance(choices,dict):
        if set(choices)!=set(r['region_id'] for r in regions):
            raise ValueError('choice region IDs mismatch')
        choices = [choices[r['region_id']] for r in regions]
    if len(choices)!=len(regions):
        raise ValueError('one shared choice per region required')
    claimed = set()
    writes = []
    region_ids = set()
    for region,choice in zip(regions,choices):
        if region['region_id'] in region_ids:
            raise ValueError('duplicate region ID')
        region_ids.add(region['region_id'])
        pixels = _integer_array(region['pixels'],'region pixels')
        if pixels.ndim != 1 or not len(pixels) or ((pixels<0)|(pixels>=y0.size)).any():
            raise ValueError('region outside image')
        allowed = set(pixels.tolist())
        if len(allowed)!=len(pixels) or claimed & allowed:
            raise ValueError('regions overlap or contain duplicate pixels')
        claimed.update(allowed)
        candidates = region['candidates']
        ids = [c['candidate_id'] for c in candidates]
        if len(candidates)>3 or len(ids)!=len(set(ids)) or KEEP in ids:
            raise ValueError('invalid shared candidate table')
        if choice==KEEP:
            continue
        if choice not in ids:
            raise ValueError('choice outside shared candidate table')
        candidate = candidates[ids.index(choice)]
        changes = candidate[arm.lower()+'_changes']
        touched = set()
        for pair in changes:
            if len(pair)!=2 or any(not isinstance(x,(int,np.integer)) or isinstance(x,bool) for x in pair):
                raise ValueError('change must be integer (pixel,class) pair')
            pixel,label = pair
            if pixel not in allowed or pixel in touched or label not in classes or label!=candidate['class_id']:
                raise ValueError('write outside region, duplicate pixel or invalid class')
            if int(y0.reshape(-1)[pixel])==label:
                raise ValueError('change signature contains no-op')
            touched.add(pixel)
            writes.append((pixel,label))
    result = y0.copy()
    flat = result.reshape(-1)
    for pixel,label in writes:
        flat[pixel]=label
    return result


def summarize_tables(tables):
    """Pool diagnostic numerators per dataset; never average image rates.

    Every rate includes its numerator and denominator. Effective/novel action
    counts are summed over regions, where outputs have distinct coordinate
    domains; pixel unions are already unique within each disjoint region.
    """
    groups = {}
    seen = set()
    for table in tables:
        key = (table['dataset'], table['image_id'])
        if key in seen:
            raise ValueError('duplicate image action table')
        seen.add(key)
        groups.setdefault(table['dataset'], []).append(table)
    result = {}
    for dataset, items in sorted(groups.items()):
        items = sorted(items, key=lambda table: table['image_id'])
        regions = [region for table in items for region in table['regions']]
        count_keys = ('geometric_valid_pixels','trigger_pixels','selected_region_pixels','component_count',
            'raw_candidates','double_keep_candidates','pair_duplicates','unique_pair_candidates',
            'truncated_candidates','shared_candidates','zero_candidates','raw_all_low_response_candidates',
            'raw_full_region_masks','response_threshold_equal_pixels','prompt_region_pixel_pairs',
            'selected_full_region_masks','mask_equals_region_outputs','majority_keep',
            'all_low_response_regions','all_cover_region_regions',
            'margin_threshold_equal_pixels','margin_threshold_selected_equal_pixels')
        arm_keys = ('raw_noop_candidates','raw_effective_actions','raw_duplicates','effective_actions',
                    'selected_noop_candidates','selected_duplicates','unique_outputs','modifiable_pixels',
                    'majority_noop','majority_changed_pixels')
        counts = {key:sum(table['stats'][key] for table in items)
                  for key in count_keys + tuple(arm+'_'+key for arm in ('mask','region') for key in arm_keys)}
        rates = {}

        def rate(name, numerator, denominator):
            rates[name] = {'numerator':numerator, 'denominator':denominator,
                           'value':_ratio(numerator, denominator)}

        rate('trigger_coverage',counts['trigger_pixels'],counts['geometric_valid_pixels'])
        rate('selected_region_coverage',counts['selected_region_pixels'],counts['geometric_valid_pixels'])
        rate('zero_candidate_region_rate',counts['zero_candidates'],len(regions))
        rate('double_keep_rate',counts['double_keep_candidates'],counts['raw_candidates'])
        rate('pair_duplicate_rate',counts['pair_duplicates'],counts['raw_candidates']-counts['double_keep_candidates'])
        rate('mask_equals_region_rate',counts['mask_equals_region_outputs'],counts['shared_candidates'])
        rate('raw_full_region_mask_rate',counts['raw_full_region_masks'],counts['raw_candidates'])
        rate('selected_full_region_mask_rate',counts['selected_full_region_masks'],counts['shared_candidates'])
        rate('raw_all_low_response_rate',counts['raw_all_low_response_candidates'],counts['raw_candidates'])
        rate('all_low_response_region_rate',counts['all_low_response_regions'],len(regions))
        rate('all_cover_region_rate',counts['all_cover_region_regions'],len(regions))
        rate('response_threshold_equal_rate',counts['response_threshold_equal_pixels'],counts['prompt_region_pixel_pairs'])
        selector_summary = {}
        for arm in ('mask','region'):
            rate(arm+'_raw_noop_rate',counts[arm+'_raw_noop_candidates'],counts['raw_candidates'])
            rate(arm+'_raw_duplicate_rate',counts[arm+'_raw_duplicates'],counts['raw_candidates']-counts[arm+'_raw_noop_candidates'])
            rate(arm+'_selected_noop_rate',counts[arm+'_selected_noop_candidates'],counts['shared_candidates'])
            rate(arm+'_selected_duplicate_rate',counts[arm+'_selected_duplicates'],counts['shared_candidates']-counts[arm+'_selected_noop_candidates'])
            rate(arm+'_unique_output_rate',counts[arm+'_unique_outputs'],counts[arm+'_effective_actions'])
            rate(arm+'_modifiable_image_coverage',counts[arm+'_modifiable_pixels'],counts['geometric_valid_pixels'])
            rate(arm+'_modifiable_region_coverage',counts[arm+'_modifiable_pixels'],counts['selected_region_pixels'])
            for selector in ('KEEP','majority'):
                all_keep = len(items) if selector=='KEEP' else sum(t['stats']['majority_all_keep'] for t in items)
                noop = len(items) if selector=='KEEP' else sum(t['stats'][arm+'_majority_all_noop'] for t in items)
                changed = 0 if selector=='KEEP' else counts[arm+'_majority_changed_pixels']
                selector_summary[selector+'/'+arm.upper()] = {
                    'all_keep_images':all_keep, 'noop_images':noop, 'changed_pixels':changed,
                    'all_keep_image_rate':{'numerator':all_keep,'denominator':len(items),'value':_ratio(all_keep,len(items))},
                    'noop_image_rate':{'numerator':noop,'denominator':len(items),'value':_ratio(noop,len(items))},
                    'changed_image_coverage':{'numerator':changed,'denominator':counts['geometric_valid_pixels'],
                                              'value':_ratio(changed,counts['geometric_valid_pixels'])},
                }
        result[dataset] = {
            'images':len(items),'regions':len(regions),
            'images_without_regions':sum(not table['regions'] for table in items),
            'zero_candidate_regions':counts['zero_candidates'],
            'zero_candidate_rate':rates['zero_candidate_region_rate']['value'],
            'majority_all_keep_images':sum(table['stats']['majority_all_keep'] for table in items),
            'geometric_claim':('NO_NEW_MASK_OUTPUTS_IN_SELECTED_SAMPLE' if counts['mask_unique_outputs']==0
                               else 'NEW_MASK_OUTPUTS_OBSERVED_NOT_EFFECTIVENESS'),
            'counts':counts,'rates':rates,'selector_summary':selector_summary,
            'per_image_stats':[dict(image_id=table['image_id'],**table['stats']) for table in items],
            'per_region_stats':[dict(image_id=table['image_id'],region_id=region['region_id'],**region['stats'])
                                for table in items for region in table['regions']],
        }
    return result
