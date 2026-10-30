"""Continuous discriminants and explicit double-signature/null/fallback policy."""
import numpy as np
import torch

PRIOR_EPS=1e-3


def raw_labels(scores,threshold=.1,background=0):
    labels=scores.argmax(0)
    return torch.where(scores.amax(0)<threshold,background,labels)


def _compatible(views):
    if not views:raise ValueError('no views')
    first=views[0]
    for s in views:
        if (s.class_ids!=first.class_ids or s.scores.shape!=first.scores.shape or
            (s.dataset,s.scene_id,s.image_id)!=(first.dataset,first.scene_id,first.image_id)):
            raise ValueError('unaligned views or class mapping mismatch')
    if len({s.view for s in views})!=len(views):raise ValueError('duplicate view')


def fuse_mean(views):
    _compatible(views)
    return torch.stack([s.scores for s in views]).mean(0)


def rms_labels(views,factors,threshold=.1,background=0):
    _compatible(views)
    if len(views)!=len(factors):raise ValueError('RMS factor count mismatch')
    for s,f in zip(views,factors):
        if f.shape!=(s.scores.shape[0],) or not torch.isfinite(f).all() or (f<=0).any():
            raise ValueError('invalid RMS factors')
    normalized=torch.stack([s.scores/f[:,None,None] for s,f in zip(views,factors)]).mean(0)
    thresholds=torch.stack([threshold/f for f in factors]).mean(0)
    labels=normalized.argmax(0)
    return torch.where(normalized.amax(0)<thresholds[labels],background,labels)


def discriminants(scores,log_prior,signatures,sigma):
    """Full Gaussian log score, same absolute domain across views. No softmax."""
    if sigma<=0:raise ValueError('sigma must be positive')
    return log_prior[:,None] + (signatures@scores - .5*signatures.square().sum(1,keepdim=True)
                               -.5*scores.square().sum(0,keepdim=True))/sigma**2


def _nearest(scores,log_prior,signatures,sigma):
    # Preserve upstream operation order for faithful single-view oracle parity.
    scores=scores.clamp(0,1);valid=scores>0
    fit=(signatures@scores-.5*signatures.square().sum(1,keepdim=True))/sigma**2
    logits=(log_prior[:,None]+fit).masked_fill(~valid,torch.finfo(fit.dtype).min)
    return torch.where(valid.any(0),logits.argmax(0),scores.argmax(0))


def faithful_ecd(class_maps,presence,active,bank,sigma=.15,background=None,background_offset=1.):
    """Independent translation of upstream single-view decoding; no ProC threshold."""
    if len(active)==0:
        return torch.full(class_maps.shape[-2:],background if background is not None else 0,dtype=torch.long)
    scores=class_maps[active].flatten(1)
    mass=torch.as_tensor(bank.log_mass,dtype=torch.float32)[active].exp()
    ra=torch.as_tensor(bank.R_a,dtype=torch.float32)[active][:,active]
    re=torch.as_tensor(bank.R_e,dtype=torch.float32)[active][:,active]
    uncal=torch.as_tensor(bank.uncalibrated)[active]
    original=scores.argmax(0);prior=torch.log(presence[active]*mass+PRIOR_EPS)
    a=_nearest(scores,prior,ra,sigma);e=_nearest(scores,prior,re,sigma)
    chosen=torch.where(a==e,a,original)
    keep=uncal[original]|uncal[chosen];chosen=torch.where(keep,original,chosen)
    labels=active[chosen]
    if background is not None:
        prior=torch.log(presence[active].clamp_min(PRIOR_EPS)*mass)
        residual=scores-ra[chosen].T
        label_fit=prior[chosen]-residual.square().sum(0)/(2*sigma**2)
        null=float(np.log(max(bank.m_bg,1e-6))+background_offset)-scores.square().sum(0)/(2*sigma**2)
        labels=torch.where((null>label_fit)&~keep,background,labels)
    return labels.reshape(class_maps.shape[-2:])


def view_discriminants(sample,signature,config,prior_mode='quality'):
    """Both signature scores, null-comparison scores, null score and eligibility."""
    c,h,w=sample.scores.shape;p=h*w;active=sample.active
    # Finite floor permits equal averaging when a concept is absent in one view.
    d=[torch.full((c,p),-1e30) for _ in range(2)];null_label=[x.clone() for x in d]
    eligible=torch.zeros((c,p),dtype=torch.bool)
    if not len(active):return d,null_label,torch.zeros(p),eligible
    s=sample.scores[active].flatten(1).clamp(0,1)
    mass=torch.as_tensor(signature.log_mass,dtype=torch.float32).exp()[active]
    bgmass=signature.m_bg
    if prior_mode=='uniform':mass=torch.ones_like(mass);bgmass=1/c
    prior=torch.log(sample.presence[active]*mass+PRIOR_EPS)
    null_prior=torch.log(sample.presence[active].clamp_min(PRIOR_EPS)*mass)
    # Compute a finite score even where a class has zero evidence in this view;
    # eligibility is applied after fusion, so one missing view cannot dominate.
    eligible[active]=s>0
    for i,r in enumerate((signature.R_a,signature.R_e)):
        r=torch.as_tensor(r,dtype=torch.float32)[active][:,active]
        d[i][active]=discriminants(s,prior,r,config.sigma)
        null_label[i][active]=discriminants(s,null_prior,r,config.sigma)
    null=float(np.log(max(bgmass,1e-6))+config.background_offset)-s.square().sum(0)/(2*config.sigma**2)
    return d,null_label,null,eligible


def signature_fallback_mask(rows,winners):
    return torch.stack([row[winner] for row in rows for winner in winners]).any(0)


def decode_multiview(views,bank,conditional=False,conditions=None,prior_mode=None):
    _compatible(views)
    if len(views)!=2 or {s.view for s in views}!={504,1008}:raise ValueError('exactly 504 and 1008 required')
    if not torch.equal(views[0].active.sort().values,views[1].active.sort().values):
        raise ValueError('P01 requires the same active vocabulary in both views; routing is disabled')
    if views[0].class_ids!=bank.class_ids:raise ValueError('signature class mapping mismatch')
    cfg=bank.config;h,w=views[0].scores.shape[-2:];c=len(bank.class_ids)
    raw=fuse_mean(views);fallback=raw_labels(raw,cfg.probability_threshold,cfg.background).flatten()
    conditions=[s.view for s in views] if conditions is None else conditions
    if len(conditions)!=2:raise ValueError('condition count mismatch')
    pieces=[];low_support=[]
    for s,condition in zip(views,conditions):
        sig=bank.conditioned.get(condition,bank.global_bank) if conditional else bank.global_bank
        pieces.append(view_discriminants(s,sig,cfg,prior_mode or cfg.prior_mode))
        low_support.append(torch.as_tensor(bank.row_fallback.get(condition,np.ones(c,dtype=bool))))
    eligible=pieces[0][3]|pieces[1][3];no_evidence=~eligible.any(0)
    null=(pieces[0][2]+pieces[1][2])/2
    choices=[];pre_null=[]
    for t in range(2):
        scores=(pieces[0][0][t]+pieces[1][0][t])/2
        scores=scores.masked_fill(~eligible,-1e30)
        winner=scores.argmax(0);pre_null.append(winner)
        null_score=(pieces[0][1][t]+pieces[1][1][t])/2
        selected=null_score.gather(0,winner[None])[0]
        choices.append(torch.where(null>selected,cfg.background,winner))
    uncal=torch.as_tensor(bank.global_bank.uncalibrated)
    disagreement=choices[0]!=choices[1]
    # All involved winners must be calibrated, even if null competition masks them.
    unknown=uncal[fallback]|uncal[pre_null[0]]|uncal[pre_null[1]]|uncal[choices[0]]|uncal[choices[1]]
    keep=disagreement|unknown|no_evidence
    result=torch.where(keep,fallback,choices[0])
    result=torch.where(raw.amax(0).flatten()<cfg.probability_threshold,cfg.background,result)
    row_fallback=signature_fallback_mask(low_support,pre_null) if conditional else torch.zeros_like(keep)
    stats={'pixels':h*w,'fallback_pixels':int(keep.sum()),'fallback_rate':float(keep.float().mean()),
           'disagreement_pixels':int(disagreement.sum()),'uncalibrated_pixels':int(unknown.sum()),
           'no_evidence_pixels':int(no_evidence.sum()),'signature_row_fallback_pixels':int(row_fallback.sum())}
    return result.reshape(h,w),stats
