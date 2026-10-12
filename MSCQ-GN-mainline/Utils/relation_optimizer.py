"""A single-factor graph-only weight-decay ablation for the relation runner."""
import math
import torch


def graph_only_parameter(name):
    # output_norm remains in PatientOnlyPropagation, so it keeps original decay.
    return name.startswith('propagation.') and not name.startswith('propagation.output_norm.')


def relation_optimizer(model, cfg, graph_weight_decay=None):
    if graph_weight_decay is None:
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if not math.isfinite(graph_weight_decay) or graph_weight_decay < 0:
        raise ValueError('Graph weight decay must be finite and nonnegative')
    patient, graph = [], []
    for name, parameter in model.named_parameters():
        (graph if graph_only_parameter(name) else patient).append(parameter)
    if not graph or not patient:
        raise ValueError('Graph decay override requires both patient and graph parameters')
    return torch.optim.Adam([
        dict(params=patient, weight_decay=cfg.weight_decay),
        dict(params=graph, weight_decay=graph_weight_decay),
    ], lr=cfg.lr, weight_decay=cfg.weight_decay)


def optimizer_manifest(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(group=i, weight_decay=group['weight_decay'], lr=group['lr'],
                 parameters=sum(p.numel() for p in group['params']),
                 names=[names[id(p)] for p in group['params']])
            for i, group in enumerate(optimizer.param_groups)]
