"""Explicitly gated real ProC adapter. No model is built by importing this file."""
import os
from pathlib import Path
import runpy
import time
import types
import numpy as np
import torch
from .cache import SampleCache,sha256
from .geometry import align_scores,validate_encoder
from .paths import checkpoint_path

ROOT=Path(__file__).resolve().parents[1]
FROZEN_POOLS={
    'LoveDA':'9c60ef6c64df630042554765854d1d70c84e2b98e7f579409ccebef098c135a7',
    'iSAID':'46b8aae662e85331ecb4af4f762e68889d55c87c9ad4ca55fd6c42697434b5ed',
}


def require_gpu_authorization(allow_gpu):
    if not allow_gpu or os.environ.get('P01_ENABLE_GPU')!='1':
        raise PermissionError('GPU disabled: later user authorization plus P01_ENABLE_GPU=1 and --allow-gpu required')


def model_options(dataset,pool=None):
    filename={'LoveDA':'cfg_loveda.py','iSAID':'cfg_isaid.py'}[dataset]
    cfg=runpy.run_path(str(ROOT/'configs'/filename))['model'].copy()
    cfg.pop('type')
    pool=Path(pool) if pool else ROOT/'prompt_pools'/f'cfg_{"loveda" if dataset=="LoveDA" else "iSAID"}_expanded_prompt_pool_hybrid_strict.pkl'
    if not pool.is_file():raise FileNotFoundError(f'released prompt pool required: {pool}; no generation/fallback allowed')
    if sha256(pool)!=FROZEN_POOLS[dataset]:raise ValueError('published prompt pool differs from frozen base asset')
    cfg.update(classname_path=str(ROOT/cfg['classname_path']),expanded_prompt_pool_path=str(pool),
               device='cuda',image_batch_size=1,execution_mode='per_image',auto_tune_slot_batch=False,
               shared_image_encoder_batch=False,compile_model=False,router_enabled=False,
               inference_dtype='fp32',image_encoder_dtype='bf16',processor_resolution=1008)
    return cfg


def author_source(checkpoint,bpe):
    """Replace only framework wiring and two asset paths; algorithms remain verbatim.

    This supplies torch.nn.Module as the base for direct predict(), with real
    mmengine data carriers. No model methods or outputs are mocked.
    """
    source=(ROOT/'sam3_segmentor_cached.py').read_text()
    replacements={
        'from mmseg.models.segmentors import BaseSegmentor':'from torch.nn import Module as BaseSegmentor',
        'from mmseg.registry import MODELS':'# P01 direct entry: no MMSeg registry needed',
        '@MODELS.register_module()':'',
        'bpe_path="/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz"':f'bpe_path={str(bpe)!r}',
        'checkpoint_path="/data/public/sam3/sam3.pt"':f'checkpoint_path={str(checkpoint)!r}',
    }
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError(f'author adapter source drift: {old}')
        source=source.replace(old,new)
    return source


def load_author_class(checkpoint=None):
    checkpoint=checkpoint or checkpoint_path()
    module=types.ModuleType('p01_author_segmentor')
    source=author_source(checkpoint,ROOT/'sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    exec(compile(source,str(ROOT/'sam3_segmentor_cached.py'),'exec'),module.__dict__)
    return module.CachedSAM3OpenSegmentor


class ProCAdapter:
    def __init__(self,dataset,provenance,allow_gpu=False,checkpoint=None):
        require_gpu_authorization(allow_gpu)
        checkpoint=checkpoint or checkpoint_path()
        options=model_options(dataset)  # pool existence before constructing anything
        if sha256(checkpoint)!=provenance['weight_sha256']:raise ValueError('weight hash mismatch')
        if sha256(options['expanded_prompt_pool_path'])!=provenance['pool_sha256']:raise ValueError('pool hash mismatch')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.manual_seed(20261030)
        self.model=load_author_class(checkpoint)(**options).eval()
        self.model.processor.model.requires_grad_(False)
        self.dataset=dataset;self.provenance=provenance;self.trace=[];self.current_view=1008
        self.trunk=self.model.processor.model.backbone.vision_backbone.trunk
        self.hooks=[self.trunk.register_forward_pre_hook(self._input),self.trunk.register_forward_hook(self._output)]
        self.total_encodes=0

    def _input(self,module,args):
        shape=list(args[0].shape);validate_encoder(self.current_view,shape)
        self.trace.append({'encoder_input':shape,'feature_shapes':[]});self.total_encodes+=1

    def _output(self,module,args,output):
        def shapes(x):
            if isinstance(x,torch.Tensor):return [list(x.shape)]
            if isinstance(x,(tuple,list)):return [s for y in x for s in shapes(y)]
            if isinstance(x,dict):return [s for y in x.values() for s in shapes(y)]
            return []
        self.trace[-1]['feature_shapes']=shapes(output)

    def set_view(self,view):
        if view not in (504,1008):raise ValueError('unsupported view')
        from sam3.model.sam3_image_processor import Sam3Processor
        self.current_view=view;model=self.model
        model.processor=Sam3Processor(model.processor.model,resolution=view,device=model.device,
                                      confidence_threshold=model.confidence_threshold)
        model.prompt_bank.processor=model.processor
        model.execution_engine.processor=model.processor
        model.processor_resolution=view

    @torch.inference_mode()
    def direct(self,row,view):
        """Original predict() including original max aggregation and threshold."""
        from mmengine.structures import BaseDataElement
        self.set_view(view);self.trace=[]
        meta=dict(img_path=row['image_path'],ori_shape=(row['height'],row['width']))
        data=self.model.predict(None,[BaseDataElement(metainfo=meta)])[0]
        if len(self.trace)!=1:raise ValueError('unexpected number of image encodes')
        return data.seg_logits.data.detach().cpu().float(),data.pred_sem_seg.data.squeeze(0).cpu().long()

    @torch.inference_mode()
    def export(self,row,view):
        from PIL import Image
        if sha256(row['image_path'])!=row['image_sha256']:raise ValueError('input changed')
        self.set_view(view);self.trace=[];model=self.model
        with Image.open(row['image_path']) as source:image=source.convert('RGB')
        if (image.height,image.width)!=(row['height'],row['width']):raise ValueError('manifest dimensions changed')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        components=model.execution_engine.infer_images([image],return_components=True)[0]
        prompt=model._fuse_prompt_outputs(components)
        # Same original aggregation; published pools have >C prompts and full mapping.
        if model.num_queries==model.num_cls and not torch.equal(model.query_idx,torch.arange(model.num_cls,device=model.device)):
            raise ValueError('original identity shortcut incompatible with reordered one-prompt vocabulary')
        scores=model._aggregate_query_logits_to_class_logits(prompt)
        scores=align_scores(scores.float(),(image.height,image.width))
        prediction,_=model._build_prediction_outputs(scores)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        resource={'online_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'encodes':len(self.trace),
                  'encoder_pixels':view*view,'prompt_decodes':model.num_queries,
                  'decoder_dtype':'fp32','encoder_dtype':model.image_encoder_dtype,'compile':False}
        if len(self.trace)!=1:raise ValueError('exactly one real encode required per view')
        prompt_presence=components['presence_scores'].float()
        presence=torch.stack([prompt_presence[model.query_idx==k].max() for k in range(model.num_cls)])
        geometry=dict(kind='full_resize',original_hw=[image.height,image.width],**self.trace[0])
        provenance=dict(self.provenance,image_sha256=row['image_sha256'],prompts=list(model.query_words),
                        prompt_source='released-pool; construction provenance unknown')
        cache=SampleCache(row['image_id'],row['source_scene_id'],self.dataset,view,
            scores.detach().cpu().float(),prompt.detach().cpu().float(),presence.cpu(),prompt_presence.cpu(),
            torch.arange(model.num_cls),model.query_idx.cpu().long(),tuple(range(model.num_cls)),
            prediction.cpu().long(),geometry,provenance)
        return cache,resource

    def benchmark(self,row,warmup=5,repeats=3):
        for _ in range(warmup):
            for view in (1008,504):self.export(row,view)
        timings=[];records=[]
        for _ in range(repeats):
            torch.cuda.synchronize();start=time.perf_counter()
            for view in (1008,504):
                _,resource=self.export(row,view);records.append(resource)
            torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        return {'warmup_pairs':warmup,'timed_pairs':repeats,'online_pair_seconds':timings,
                'mean_seconds':float(np.mean(timings)),'p95_seconds':float(np.percentile(timings,95)),
                'records':records,'includes':'preprocessing, encoder, prompt decode, CPU transfer; excludes cache IO and calibration'}

    def close(self):
        for hook in self.hooks:hook.remove()
        self.hooks=[]
