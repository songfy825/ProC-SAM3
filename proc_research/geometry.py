"""Continuous response geometry. Labels are never interpolated here."""
import torch
import torch.nn.functional as F


def align_scores(scores, original_hw):
    if scores.dtype != torch.float32 or scores.ndim != 3:
        raise ValueError('geometry requires FP32 [C,H,W] responses, not labels')
    if len(original_hw) != 2 or min(original_hw) <= 0:
        raise ValueError('invalid original dimensions')
    if tuple(scores.shape[-2:]) == tuple(original_hw):
        return scores
    return F.interpolate(scores[None], size=tuple(original_hw), mode='bilinear',
                         align_corners=False)[0]


def validate_encoder(requested, actual_shape):
    if requested not in (504, 1008) or list(actual_shape) != [1, 3, requested, requested]:
        raise ValueError(f'encoder size mismatch: requested={requested}, actual={actual_shape}')
