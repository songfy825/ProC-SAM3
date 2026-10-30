"""Lossless per-view caches with externally pinned metadata/content hashes."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .geometry import validate_encoder

SCHEMA = 1
RESERVE_BYTES = 20 * 1024**3


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


@dataclass
class SampleCache:
    image_id: str
    scene_id: str
    dataset: str
    view: int
    scores: torch.Tensor
    prompt_scores: torch.Tensor
    presence: torch.Tensor
    prompt_presence: torch.Tensor
    active: torch.Tensor
    prompt_to_class: torch.Tensor
    class_ids: tuple
    prediction: torch.Tensor
    geometry: dict
    provenance: dict

    @property
    def key(self):
        return (self.dataset, self.scene_id, self.image_id, self.view)


ARRAY_FIELDS = ('scores', 'prompt_scores', 'presence', 'prompt_presence', 'active',
                'prompt_to_class', 'prediction')


def validate_sample(s):
    c, h, w = s.scores.shape
    q = s.prompt_scores.shape[0]
    for name in ('scores', 'prompt_scores', 'presence', 'prompt_presence'):
        t = getattr(s, name)
        if t.dtype != torch.float32 or t.device.type != 'cpu' or not torch.isfinite(t).all():
            raise ValueError(f'{name} must be finite CPU FP32')
        if (t < 0).any() or (t > 1).any():
            raise ValueError(f'{name} outside probability domain')
    if (tuple(s.prompt_scores.shape) != (q,h,w) or s.presence.shape != (c,)
            or s.prompt_presence.shape != (q,) or s.prompt_to_class.shape != (q,)
            or tuple(s.prediction.shape) != (h,w)):
        raise ValueError('cache shape mismatch')
    if len(s.class_ids) != c or len(set(s.class_ids)) != c:
        raise ValueError('class mapping is not unique')
    for name in ('active', 'prompt_to_class', 'prediction'):
        t = getattr(s,name)
        if t.dtype != torch.int64 or t.device.type != 'cpu' or ((t<0)|(t>=c)).any():
            raise ValueError(f'invalid {name}')
    if s.active.ndim != 1 or s.active.unique().numel() != s.active.numel():
        raise ValueError('invalid active list')
    if s.geometry.get('kind') != 'full_resize' or s.geometry.get('original_hw') != [h,w]:
        raise ValueError('unsupported or mismatched geometry')
    validate_encoder(s.view,s.geometry['encoder_input'])
    if not s.geometry.get('feature_shapes'):
        raise ValueError('encoder feature evidence missing')
    for key in ('image_sha256','weight_sha256','pool_sha256','config_sha256','code_sha256'):
        value=s.provenance.get(key,'')
        if len(value)!=64 or any(ch not in '0123456789abcdef' for ch in value):
            raise ValueError(f'missing/invalid provenance {key}')


def ensure_capacity(directory, needed_bytes, reserve=RESERVE_BYTES):
    p=Path(directory)
    while not p.exists(): p=p.parent
    free=shutil.disk_usage(p).free
    if free-int(needed_bytes)<reserve:
        raise OSError(f'cache capacity: free={free}, need={needed_bytes}, reserve={reserve}')
    return {'free_bytes':free,'estimated_bytes':int(needed_bytes),'reserve_bytes':reserve}


def estimate_cache_bytes(rows, prompts, classes, views=2):
    # Maps + prediction, conservative 5% archive/header allowance.
    return int(sum(r['height']*r['width'] for r in rows) * views *
               (4*(prompts+classes)+8) * 1.05)


def write_sample(directory,s):
    validate_sample(s)
    p=Path(directory)
    arrays={name:getattr(s,name).contiguous().numpy() for name in ARRAY_FIELDS}
    ensure_capacity(p,sum(a.nbytes for a in arrays.values())+1024**2)
    p.mkdir(parents=True,exist_ok=False)
    np.savez(p/'arrays.npz',**arrays)
    meta={k:getattr(s,k) for k in ('image_id','scene_id','dataset','view','class_ids','geometry','provenance')}
    meta.update(schema=SCHEMA,arrays_sha256=sha256(p/'arrays.npz'))
    write_json(p/'meta.json',meta)
    # Receipt is also stored in the parent run index, preventing self-repaired corruption.
    receipt={'meta_sha256':sha256(p/'meta.json'),'arrays_sha256':meta['arrays_sha256']}
    write_json(p/'receipt.json',receipt)
    return receipt


def load_sample(directory,expected=None):
    p=Path(directory)
    if expected is None:
        expected=json.loads((p/'receipt.json').read_text())
    if sha256(p/'meta.json')!=expected['meta_sha256'] or sha256(p/'arrays.npz')!=expected['arrays_sha256']:
        raise ValueError('cache hash corruption')
    meta=json.loads((p/'meta.json').read_text())
    if meta.pop('schema')!=SCHEMA or meta.pop('arrays_sha256')!=expected['arrays_sha256']:
        raise ValueError('cache version or content mismatch')
    with np.load(p/'arrays.npz',allow_pickle=False) as f:
        arrays={name:torch.from_numpy(f[name].copy()) for name in ARRAY_FIELDS}
    meta['class_ids']=tuple(meta['class_ids'])
    sample=SampleCache(**meta,**arrays)
    validate_sample(sample)
    return sample
