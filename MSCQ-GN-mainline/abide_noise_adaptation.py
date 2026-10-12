"""Noise-only ABIDE controls. No changes to C's network or random draw count."""
import torch
from Model.relation_sheaf import RelationSheafModel

ARMS=('N1','N2','N3')


def configure_noise(model, modal_names, arm):
    if arm not in ('W',)+ARMS:
        raise ValueError(arm)
    if model.noise_scale<=0:
        raise ValueError('This experiment requires the nonzero W noise scale')
    names=list(modal_names)
    for name in ('PHENO','Correlation'):
        if names.count(name)!=1:
            raise ValueError(f'Expected one {name} input group')
    # Base forward still draws one full randn_like(x), even for zero-noise columns.
    with torch.no_grad():
        if arm in ('N1','N3'):
            model._modal_noise_std[names.index('Correlation')]=.03/model.noise_scale
        if arm in ('N2','N3'):
            model._modal_noise_std[names.index('PHENO')]=0.
    return model


def make_model(dd,cfg,graphs,mode,training_seed=0,arm='W'):
    if mode!='content':
        raise ValueError('Noise experiment preserves C content mode')
    model=RelationSheafModel(dd,cfg,graphs,mode,training_seed=training_seed)
    return configure_noise(model,dd['Modal_Name'],arm)
