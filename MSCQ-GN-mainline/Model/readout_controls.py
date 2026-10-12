"""Isolated training controls for C's existing class-specific readout.

C and QK_NODECAY have exactly the original model forward and parameter layout.
The latter changes only Adam's coupled L2 term for class-query/Q/K parameters;
the packed V rows retain their original decay after task-gradient clipping.
UNIFORM retains each class's V/output/FFN and attention-probability dropout.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .relation_sheaf import RelationSheafModel
from Utils.relation_optimizer import (graph_only_parameter, relation_optimizer,
                                      optimizer_manifest)


class UniformPerLabelPool(nn.Module):
    """Fixed uniform weights BEFORE the original attention dropout.

The stochastic training weights are Bernoulli-masked 1/M/(1-p), without
renormalization, independently per patient, attention head, and modality.
At evaluation this is exact uniform pooling of the class-specific values.
Query and Q/K parameters are removed rather than counted as active capacity.
"""
    def __init__(self, original_pool):
        super().__init__()
        attention = original_pool.attn
        if (not attention._qkv_same_embed_dim or attention.in_proj_bias is None
                or attention.bias_k is not None or attention.bias_v is not None
                or attention.add_zero_attn):
            raise ValueError('Uniform control expects the unmodified C class pool')
        self.hidden = attention.embed_dim
        self.num_heads = attention.num_heads
        self.attention_dropout = float(attention.dropout)
        self.norm_kv = original_pool.norm_kv
        self.norm_out = original_pool.norm_out
        self.ffn = original_pool.ffn
        self.out_proj = attention.out_proj
        # Deep-copying an existing Linear consumes no initialization RNG.
        self.v_proj = copy.deepcopy(attention.out_proj)
        with torch.no_grad():
            self.v_proj.weight.copy_(attention.in_proj_weight[2*self.hidden:])
            self.v_proj.bias.copy_(attention.in_proj_bias[2*self.hidden:])
        self.train(original_pool.training)

    def forward(self, modal_tokens):
        batch, modalities, hidden = modal_tokens.shape
        if hidden != self.hidden or modalities < 1:
            raise ValueError('Expected [patients, modalities, hidden] tokens')
        head_size = hidden // self.num_heads
        # Preserve the original MHA sequence-first projection and SDPA axes.
        values = self.v_proj(self.norm_kv(modal_tokens).transpose(0, 1))
        values = values.view(modalities, batch*self.num_heads, head_size).transpose(0, 1)
        values = values.view(batch, self.num_heads, modalities, head_size)
        queries = modal_tokens.new_zeros((1, batch, hidden))
        queries = queries.view(1, batch*self.num_heads, head_size).transpose(0, 1)
        queries = queries.view(batch, self.num_heads, 1, head_size)
        keys = modal_tokens.new_zeros((modalities, batch, hidden))
        keys = keys.view(modalities, batch*self.num_heads, head_size).transpose(0, 1)
        keys = keys.view(batch, self.num_heads, modalities, head_size)
        pooled = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0., is_causal=False)
        pooled = pooled.permute(2, 0, 1, 3).contiguous().view(batch, hidden)
        pooled = self.out_proj(pooled)
        return pooled + self.ffn(self.norm_out(pooled))


class ReadoutControlModel(RelationSheafModel):
    """Original C with an optimizer-only or strictly uniform readout control."""
    def __init__(self, dd, cfg, graphs, mode, training_seed=0):
        if mode not in ('C', 'QK_NODECAY', 'UNIFORM'):
            raise ValueError(mode)
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        self.readout_control_mode = mode
        if mode == 'UNIFORM':
            self.label_pools = nn.ModuleList(UniformPerLabelPool(pool) for pool in self.label_pools)
        # The inherited forward is intentionally unchanged in all three modes.


class _ReadoutSliceAdam(torch.optim.Adam):
    """Native Adam with temporary post-clip L2 on selected packed V rows.

The caller must clip task gradients BEFORE step(), just as original C does.
The whole packed parameter's optimizer decay is zero. Only its V rows receive
lambda*p before Adam updates BOTH moments. Original task-gradient objects are
restored even on failure, so repeated steps cannot accumulate injected L2.
Closures are rejected because they could recalculate gradients after injection.
"""
    def __init__(self, params, *, slice_decay, **kwargs):
        super().__init__(params, **kwargs)
        self.slice_decay = tuple(slice_decay)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Readout slice decay requires explicit backward/clip/step, without closure')
        originals = []
        try:
            for parameter, start, stop, coefficient in self.slice_decay:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError('Packed attention parameters require dense gradients')
                original = parameter.grad
                originals.append((parameter, original))
                parameter.grad = original.clone(memory_format=torch.preserve_format)
                parameter.grad[start:stop].add_(parameter[start:stop], alpha=coefficient)
            return super().step()
        finally:
            for parameter, original in originals:
                parameter.grad = original


def _selection_parameters(model):
    result = []
    for pool in model.label_pools:
        result.extend((pool.query, pool.attn.in_proj_weight, pool.attn.in_proj_bias))
    return result


def readout_optimizer(model, cfg):
    """Keep graph decay zero and the original Adam defaults/ordinary decay.

QK_NODECAY removes decay from class-pool query, Q/K weight and Q/K bias only.
V, output projections, LayerNorm, FFN, auxiliary heads, and all other patient
parameters retain cfg.weight_decay. Shared Transformer attention is untouched.
"""
    mode = model.readout_control_mode
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    if mode != 'QK_NODECAY':
        return relation_optimizer(model, cfg, 0.)
    selection = _selection_parameters(model)
    selected_ids = {id(parameter) for parameter in selection}
    ordinary, graph = [], []
    for name, parameter in model.named_parameters():
        if id(parameter) in selected_ids:
            continue
        (graph if graph_only_parameter(name) else ordinary).append(parameter)
    if not ordinary or not graph or len(selection) != 3*len(model.label_pools):
        raise ValueError('Unexpected C parameter structure')
    slices = []
    for pool in model.label_pools:
        hidden = pool.attn.embed_dim
        for parameter in (pool.attn.in_proj_weight, pool.attn.in_proj_bias):
            slices.append((parameter, 2*hidden, 3*hidden, float(cfg.weight_decay)))
    return _ReadoutSliceAdam([
        dict(params=ordinary, weight_decay=cfg.weight_decay),
        dict(params=graph, weight_decay=0.),
        dict(params=selection, weight_decay=0.),
    ], lr=cfg.lr, weight_decay=cfg.weight_decay, slice_decay=slices)


def readout_optimizer_manifest(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    overrides = [dict(name=names[id(parameter)], axis=0, start=start, stop=stop,
                      weight_decay=coefficient)
                 for parameter, start, stop, coefficient in getattr(optimizer, 'slice_decay', ())]
    selection = []
    if model.readout_control_mode == 'QK_NODECAY':
        for index, pool in enumerate(model.label_pools):
            prefix = f'label_pools.{index}'
            selection.append(dict(name=f'{prefix}.query', scope='whole parameter'))
            for suffix in ('weight', 'bias'):
                selection.append(dict(name=f'{prefix}.attn.in_proj_{suffix}', axis=0,
                                      start=0, stop=2*pool.attn.embed_dim))
    return dict(
        mode=model.readout_control_mode, optimizer='Adam',
        weight_decay_semantics='Coupled L2 added after task-gradient clipping, before both Adam moments; not AdamW',
        groups=optimizer_manifest(model, optimizer), slice_overrides=overrides,
        no_decay_selection=selection,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                 if parameter.requires_grad))


def _require_eval(model):
    if any(module.training for module in model.modules()):
        raise ValueError('Readout diagnostics/interventions require model.eval(), including all submodules')


def _pool_diagnostics(pool, tokens, class_index):
    batch, modalities, hidden = tokens.shape
    if isinstance(pool, UniformPerLabelPool):
        return dict(class_index=class_index, heads=pool.num_heads, modalities=modalities,
                    attention_max_abs_deviation=0., attention_mean_abs_deviation=0.,
                    normalized_entropy_mean=1. if modalities > 1 else 0.,
                    query_norm=None, effective_query_norm=None, q_weight_norm=None,
                    k_weight_norm=None, v_weight_norm=float(pool.v_proj.weight.norm()),
                    out_weight_norm=float(pool.out_proj.weight.norm()))
    attention = pool.attn
    normalized = pool.norm_kv(tokens)
    query = pool.query.expand(batch, -1, -1)
    # Match native MHA's packed Q / joined K,V projection layout. Diagnostic
    # probabilities are pre-dropout math probabilities, not head-averaged maps.
    projected_query = F.linear(query, attention.in_proj_weight[:hidden],
                               attention.in_proj_bias[:hidden])
    projected_kv = F.linear(normalized, attention.in_proj_weight[hidden:],
                            attention.in_proj_bias[hidden:])
    projected_key = projected_kv[..., :hidden]
    heads = attention.num_heads
    head_size = hidden // heads
    q = projected_query.view(batch, 1, heads, head_size).transpose(1, 2)
    k = projected_key.reshape(batch, modalities, heads, head_size).transpose(1, 2)
    probability = ((q @ k.transpose(-1, -2))/math.sqrt(head_size)).softmax(-1)
    deviations = (probability - 1/modalities).abs()
    entropy = -(probability*probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(-1)
    normalized_entropy = entropy/math.log(modalities) if modalities > 1 else torch.zeros_like(entropy)
    return dict(class_index=class_index, heads=heads, modalities=modalities,
                attention_max_abs_deviation=float(deviations.max()),
                attention_mean_abs_deviation=float(deviations.mean()),
                normalized_entropy_mean=float(normalized_entropy.mean()),
                query_norm=float(pool.query.norm()),
                effective_query_norm=float(projected_query[0].norm()),
                q_weight_norm=float(attention.in_proj_weight[:hidden].norm()),
                k_weight_norm=float(attention.in_proj_weight[hidden:2*hidden].norm()),
                v_weight_norm=float(attention.in_proj_weight[2*hidden:].norm()),
                out_weight_norm=float(attention.out_proj.weight.norm()))


@torch.no_grad()
def readout_diagnostics(model, x):
    """Eval-only JSON diagnostics; no state/gradient/RNG changes or retained hooks."""
    _require_eval(model)
    rows = {}
    handles = []
    try:
        for index, pool in enumerate(model.label_pools):
            def record(module, args, index=index):
                rows[index] = _pool_diagnostics(module, args[0], index)
            handles.append(pool.register_forward_pre_hook(record))
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    layers = [rows[index] for index in range(len(model.label_pools))]
    summary = dict(
        attention_max_abs_deviation=max(row['attention_max_abs_deviation'] for row in layers),
        attention_mean_abs_deviation=sum(row['attention_mean_abs_deviation'] for row in layers)/len(layers),
        normalized_entropy_mean=sum(row['normalized_entropy_mean'] for row in layers)/len(layers))
    for key in ('query_norm', 'effective_query_norm', 'q_weight_norm', 'k_weight_norm',
                'v_weight_norm', 'out_weight_norm'):
        available = [row[key] for row in layers if row[key] is not None]
        summary[key] = sum(available)/len(available) if available else None
    return dict(mode=model.readout_control_mode, layers=layers, summary=summary,
                scope='Full supplied cohort; per-head pre-dropout math attention; no head averaging before deviations')


@torch.no_grad()
def uniform_intervention(model, x):
    """Return (normal_logits, uniform_logits), restoring original pool objects.

Fixed-weight evaluation only. It preserves class-specific V/output/FFN,
embeddings and all downstream paths, without retraining or checkpoint selection.
"""
    _require_eval(model)
    normal = model(x)[0]
    if model.readout_control_mode == 'UNIFORM':
        return normal, normal.clone()
    original = model.label_pools
    try:
        replacements = nn.ModuleList(UniformPerLabelPool(pool) for pool in original)
        replacements.eval()
        model.label_pools = replacements
        uniform = model(x)[0]
    finally:
        model.label_pools = original
    return normal, uniform
