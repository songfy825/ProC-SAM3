from dataclasses import dataclass,replace,asdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ..vendor.signatures import SignatureEstimator,Signatures
from ..cache import digest,write_json,sha256


@dataclass(frozen=True)
class CalibrationConfig:
    core_high: float=.5
    core_low: float=.3
    min_pixels: int=2000
    min_scenes: int=5
    sigma: float=.15
    background: int=0
    background_offset: float=1.
    probability_threshold: float=.1
    prior_mode: str='quality'

    def __post_init__(self):
        if not 0<self.core_low<self.core_high<1 or self.min_pixels<1 or self.min_scenes<1 or self.sigma<=0:
            raise ValueError('invalid calibration constants')
        if self.prior_mode not in ('quality','uniform'):raise ValueError('invalid prior')


@dataclass
class SignatureBank:
    global_bank: Signatures
    conditioned: dict
    support: dict
    row_fallback: dict
    rms: dict
    rms_global: torch.Tensor
    class_ids: tuple
    config: CalibrationConfig
    meta: dict


def _factor(total,n):
    rms=(total/max(n,1)).sqrt().float()
    return torch.where(rms==0,torch.ones_like(rms),rms.clamp_min(1e-6))


def fit_rms(samples):
    total=None;n=0;mapping=None
    for sample in samples:
        if mapping is not None and sample.class_ids!=mapping:raise ValueError('class mapping changed')
        mapping=sample.class_ids
        value=sample.scores.double().square().sum((1,2))
        total=value if total is None else total+value;n+=sample.scores.shape[1]*sample.scores.shape[2]
    if total is None:raise ValueError('empty calibration pool')
    return _factor(total,n)


def _pooled(sample):
    h,w=sample.scores.shape[-2:]
    if min(h,w)<4:raise ValueError('official ECD estimation needs dimensions >=4')
    p=F.adaptive_avg_pool2d(sample.scores[None],(h//4,w//4))[0][sample.active]
    return (p.clamp(0,1)*255).round().to(torch.uint8).flatten(1).float()/255.


def fit_signatures(samples,config=CalibrationConfig(),conditions=None):
    """Streaming pool pass; only R rows condition on view, all priors stay global."""
    global_est=None;estimators={};scene_sets={};square_sums={};pixels={};mapping=None
    sample_keys=[];dataset=None;seen=set()
    for i,s in enumerate(samples):
        if s.key in seen:raise ValueError('duplicate calibration scene-view sample')
        seen.add(s.key)
        if mapping is None:
            mapping=s.class_ids;dataset=s.dataset;c=len(mapping)
            global_est=SignatureEstimator(c,config.core_high,config.core_low,config.min_pixels,device='cpu')
        if s.class_ids!=mapping or s.dataset!=dataset:raise ValueError('mixed vocabulary/dataset calibration')
        condition=int(conditions[i]) if conditions is not None else s.view
        if condition not in (504,1008):raise ValueError('unknown condition')
        if condition not in estimators:
            estimators[condition]=SignatureEstimator(c,config.core_high,config.core_low,config.min_pixels,device='cpu')
            scene_sets[condition]=[set() for _ in range(c)]
        if len(s.active):
            global_est.add(s.scores,s.active);estimators[condition].add(s.scores,s.active)
            pooled=_pooled(s);winner=pooled.argmax(0)
            for j,cls in enumerate(s.active.tolist()):
                if ((winner==j)&(pooled[j]>0)).any():scene_sets[condition][cls].add(s.scene_id)
        square=s.scores.double().square().sum((1,2))
        square_sums[s.view]=square_sums.get(s.view,torch.zeros_like(square))+square
        pixels[s.view]=pixels.get(s.view,0)+s.scores.shape[1]*s.scores.shape[2]
        sample_keys.append(list(s.key))
    if global_est is None:raise ValueError('empty calibration pool')
    if conditions is not None and len(conditions)!=len(sample_keys):raise ValueError('condition count mismatch')
    if global_est.num_images==0:raise ValueError('no active calibration responses')
    glob=global_est.signatures(config.background,meta={'adaptation':'ECD-ProC','sample_keys_sha256':digest(sample_keys)})
    banks={};support={};fallback={}
    for condition,est in estimators.items():
        counts=np.array([len(x) for x in scene_sets[condition]],dtype=np.int64)
        use_global=counts<config.min_scenes
        local=est.signatures(config.background) if est.num_images else glob
        # Same global mass, uncalibrated mask, and null statistics for every method.
        banks[condition]=replace(glob,
            R_a=np.where(use_global[:,None],glob.R_a,local.R_a),
            R_e=np.where(use_global[:,None],glob.R_e,local.R_e))
        support[condition]=counts;fallback[condition]=use_global
    return SignatureBank(glob,banks,support,fallback,
                         {v:_factor(square_sums[v],pixels[v]) for v in square_sums},
                         _factor(sum(square_sums.values()),sum(pixels.values())),mapping,config,
                         {'dataset':dataset,'sample_keys':sample_keys,'conditions':conditions,
                          'scene_ids':sorted(set(k[1] for k in sample_keys)),
                          'config':asdict(config)})


def shuffle_conditions(samples,seed):
    """Balanced permutations over canonical records, excluding identity/name swaps."""
    if len(samples)<4:raise ValueError('too few records for a nontrivial condition shuffle')
    keys=[s.key for s in samples]
    if len(set(keys))!=len(keys):raise ValueError('duplicate shuffle key')
    order=sorted(range(len(keys)),key=lambda i:keys[i])
    original=np.array([samples[i].view for i in order])
    if set(original)!={504,1008} or min(np.sum(original==v) for v in (504,1008))<2:
        raise ValueError('both condition buckets need at least two records')
    rng=np.random.default_rng(seed)
    for _ in range(10000):
        values=rng.permutation(original)
        if all(len(set(values[original==v]))==2 for v in (504,1008)):
            result=[0]*len(keys)
            for i,v in zip(order,values):result[i]=int(v)
            return result
    raise ValueError('could not generate valid shuffled control')


def save_bank(directory,bank):
    p=Path(directory);p.mkdir(parents=True,exist_ok=False)
    bank.global_bank.save(p/'global.npz')
    for v,s in bank.conditioned.items():s.save(p/f'{v}.npz')
    write_json(p/'metadata.json',{'config':asdict(bank.config),'class_ids':bank.class_ids,'meta':bank.meta,
        'support':{str(v):x.tolist() for v,x in bank.support.items()},
        'fallback':{str(v):x.tolist() for v,x in bank.row_fallback.items()},
        'rms':{str(v):x.tolist() for v,x in bank.rms.items()},'rms_global':bank.rms_global.tolist()})
    receipt={f.name:sha256(f) for f in p.iterdir()}
    write_json(p/'receipt.json',receipt)
    return receipt


def load_bank(directory,expected):
    p=Path(directory)
    for name,value in expected.items():
        if Path(name).name!=name or sha256(p/name)!=value:raise ValueError('signature bank hash mismatch')
    m=json.loads((p/'metadata.json').read_text());glob=Signatures.load(p/'global.npz')
    return SignatureBank(glob,{int(v):Signatures.load(p/f'{v}.npz') for v in m['support']},
        {int(v):np.array(x) for v,x in m['support'].items()},
        {int(v):np.array(x,dtype=bool) for v,x in m['fallback'].items()},
        {int(v):torch.tensor(x) for v,x in m['rms'].items()},torch.tensor(m['rms_global']),
        tuple(m['class_ids']),CalibrationConfig(**m['config']),m['meta'])
