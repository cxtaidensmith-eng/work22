"""Fixed-checkpoint interventions on C/self-only disagreements.

All conditions are prespecified and applied to every patient. Labels identify
groups and score outputs only; no fitting or checkpoint/parameter selection.
These checkpoints were historically TEST-ACC selected, so all results remain
exploratory. Label-informed 64-mask ceilings are not deployable performance.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT, setup, dump, sha, timestamp
from reference_selection_experiment import read, model_hash
from reference_selection_space import analyze_arrays
from Utils.reference_selection import coalition_masks
from Model.reference_selection import ReferenceSelectionModel, composed_self_part
from Model.relation_sheaf import conformal_maps, cooperative_operator


def norm_match(reference, replacement):
    """Match recipient/content L2 size without inventing a zero-vector direction."""
    old = reference.norm(dim=-1, keepdim=True)
    new = replacement.norm(dim=-1, keepdim=True)
    result = replacement * (old/new.clamp_min(1e-12))
    return result, (new.squeeze(-1) <= 1e-12) & (old.squeeze(-1) > 1e-12)


@torch.no_grad()
def overridden_logits(model, x, tokens):
    handle = model.propagation.register_forward_hook(lambda module, inputs, output: tokens)
    try:
        return model(x)[0]
    finally:
        handle.remove()


class FrozenFlow:
    """Freeze graph evidence and own diagonal while replacing foreign values."""
    @torch.no_grad()
    def __init__(self, model, x):
        if model.training or model.propagation.mode != 'content':
            raise ValueError('Requires eval-mode C with content-only maps')
        p = self.p = model.propagation
        cache = {}
        def capture(module, inputs):
            cache['tokens'] = inputs[0].detach().clone()
        handle = p.register_forward_pre_hook(capture)
        try:
            self.logits = model(x)[0]
        finally:
            handle.remove()
        self.tokens = cache['tokens']
        self.count, self.m, self.h = self.tokens.shape
        z = p.input_norm(self.tokens)
        q = F.normalize(p.edge_query(z), dim=-1).permute(1, 0, 2)
        q = q[None].expand(self.m, -1, -1, -1).reshape(self.m*self.m*self.count, -1)
        i, j = p.edge
        self.weights = torch.sigmoid((q[i]*q[j]).sum(-1)
            + p.similarity*p.edge_similarity.flatten()[p.edge_group] + p.edge_bias.flatten()[p.edge_group])
        self.maps = conformal_maps(p.map_values(z).reshape(-1, 4))
        self.values = p.value(z).permute(1, 0, 2)[None].expand(self.m, -1, -1, -1).reshape(-1, 2, self.h//2)
        full = cooperative_operator(self.values, p.edge, self.weights, *self.maps)
        own = composed_self_part(self.values, p.edge, self.weights, *self.maps, constant_maps=True)
        self.delta, self.own = self.collapse(full), self.collapse(own)
        self.neighbor = self.delta-self.own
        torch.testing.assert_close(self.restore(self.delta), p(self.tokens), rtol=2e-5, atol=2e-5)

    def collapse(self, values):
        return values.reshape(self.m, self.m, self.count, self.h).mean(0).permute(1, 0, 2)

    @torch.no_grad()
    def foreign_values(self, values):
        full = cooperative_operator(values, self.p.edge, self.weights, *self.maps)
        own = composed_self_part(values, self.p.edge, self.weights, *self.maps, constant_maps=True)
        return self.collapse(full)-self.collapse(own)

    @torch.no_grad()
    def restore(self, delta):
        p = self.p
        return p.output_norm(self.tokens-p.step_logit.sigmoid()*p.output(F.gelu(delta)))

    @torch.no_grad()
    def replace_foreign(self, foreign):
        # Exact original delta when the foreign message is unchanged.
        return self.restore(self.delta+(foreign-self.neighbor))

    @torch.no_grad()
    def replacement_values(self, permutation=None):
        values = self.values.reshape(self.m, self.m, self.count, 2, self.h//2)
        changed = (values.mean(2, keepdim=True).expand_as(values) if permutation is None else
                   values[:, :, torch.as_tensor(permutation, device=values.device)])
        return changed.reshape_as(self.values)


def group_ids(base, self_pred, labels):
    bc, sc = base == labels, self_pred == labels
    result = np.full(len(labels), 'both_wrong', dtype='<U24')
    result[~bc & sc] = 'c_wrong_self_right'
    result[bc & ~sc] = 'c_right_self_wrong'
    result[bc & sc] = 'both_right'
    return result


@torch.no_grad()
def run_checkpoint(cfg, dd, data, graphs, entry, folder):
    model = ReferenceSelectionModel(dd, cfg, graphs, entry['seed'], learned=False).to(cfg.Device).eval()
    checkpoint = torch.load(entry['checkpoint'], map_location=cfg.Device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    initial_hash = model_hash(model)
    x, y = data['Feature'], data['Label']
    test = data['Mask'][entry['fold']-1][1]
    with np.load(entry['c_predictions']) as saved, np.load(entry['self_predictions']) as own:
        np.testing.assert_array_equal(saved['row_index'], own['row_index'])
        np.testing.assert_array_equal(saved['labels'], own['labels'])
        np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
        np.testing.assert_array_equal(saved['labels'], y[test].cpu().numpy())
        self_pred = own['logits'].argmax(-1)
        source_row, labels = saved['row_index'].copy(), saved['labels'].copy()
        flow = FrozenFlow(model, x)
        np.testing.assert_allclose(flow.logits[test].cpu().numpy(), saved['logits'], atol=2e-5, rtol=2e-5)
    masks = coalition_masks(len(dd['Modal_Name']))
    mask_logits = np.stack([model(x, mask[None])[0][test].cpu().numpy() for mask in masks])
    np.testing.assert_allclose(mask_logits[-1], flow.logits[test].cpu().numpy(), atol=2e-5, rtol=2e-5)
    conditions = dict(all_open=mask_logits[-1], all_closed=mask_logits[0],
        constant_half=model(x, np.full((1, len(dd['Modal_Name'])), .5, dtype=np.float32))[0][test].cpu().numpy())
    for n, name in enumerate(dd['Modal_Name']):
        conditions[f'close_{name}'] = mask_logits[len(masks)-1-(1 << n)]
    # Direct identity-update control keeps C's learned output norm and readout.
    conditions['bypass_graph_update'] = overridden_logits(model, x, model.propagation.output_norm(flow.tokens))[test].cpu().numpy()
    identity = overridden_logits(model, x, flow.replace_foreign(flow.foreign_values(flow.values)))[test].cpu().numpy()
    np.testing.assert_allclose(identity, mask_logits[-1], atol=2e-5, rtol=2e-5)
    closed = overridden_logits(model, x, flow.replace_foreign(torch.zeros_like(flow.neighbor)))[test].cpu().numpy()
    np.testing.assert_allclose(closed, mask_logits[0], atol=2e-5, rtol=2e-5)
    rng = np.random.default_rng(44019)
    variants = [('cohort_mean', None)] + [(f'permutation_{k}', rng.permutation(len(x))) for k in range(3)]
    degenerate = {}
    for name, permutation in variants:
        foreign = flow.foreign_values(flow.replacement_values(permutation))
        matched, invalid = norm_match(flow.neighbor, foreign)
        degenerate[name] = int(invalid[test].sum())
        for condition, message in ((name, foreign), (name+'_norm_matched', matched)):
            conditions[condition] = overridden_logits(model, x, flow.replace_foreign(message))[test].cpu().numpy()
    assert initial_hash == model_hash(model), 'Diagnostic changed model state'
    condition_names = list(conditions)
    condition_logits = np.stack(list(conditions.values()))
    assert np.isfinite(condition_logits).all() and np.isfinite(mask_logits).all()
    mask_losses = F.cross_entropy(torch.from_numpy(mask_logits).reshape(-1, dd['Class_Num']),
        torch.from_numpy(np.tile(labels, len(masks))), reduction='none').reshape(len(masks), -1).numpy()
    oracle, _ = analyze_arrays(mask_logits, mask_losses, labels, masks)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'predictions.npz', condition_names=np.asarray(condition_names),
        condition_logits=condition_logits, mask_logits=mask_logits, mask_losses=mask_losses, masks=masks,
        labels=labels, source_row=source_row, self_predictions=self_pred,
        group=group_ids(mask_logits[-1].argmax(-1), self_pred, labels))
    record = dict(seed=entry['seed'], fold=entry['fold'], epoch=checkpoint['best']['epoch'],
        checkpoint_sha256=entry['checkpoint_sha256'], predictions_sha256=sha(folder/'predictions.npz'),
        unchanged_model_state=True, identity_and_closed_replay_verified=True,
        norm_match_degenerate_test_patient_modalities=degenerate, oracle=oracle)
    dump(folder/'complete.json', record)
    return record


def main():
    out = ROOT/'Result/c_disagreement_diagnosis'
    if out.exists():
        raise FileExistsError(out)
    baseline = ROOT/'Result/reference_selection'
    protocol = read(baseline/'PROTOCOL.json')
    for path, digest in protocol['code_sha256'].items():
        assert sha(ROOT/path) == digest, path
    cfg, dd, data, graphs, paths = setup(k=protocol['candidate_k'])
    assert {Path(p).name: sha(p) for p in paths} == protocol['data_sha256']
    assert hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest() == protocol['candidate_hash']
    entries = []
    for seed in (0, 1, 2):
        self_base = ROOT/'Result/relation_graph_audit'/f'self_seed_{seed}'
        self_protocol = read(self_base/'PROTOCOL.json')
        for field in ('original_config', 'data_sha256', 'checkpoint_selection', 'folds', 'epochs', 'split_seed'):
            assert self_protocol[field] == protocol[field]
        for fold in range(1, 11):
            folder = baseline/f'seed_{seed}'/'c_real'/f'fold_{fold:02d}'
            reference = read(folder/'baseline_reference.json')
            cp = Path(reference['source'])/'best.pt'
            assert sha(cp) == reference['best_checkpoint_sha256']
            pred, self_pred = folder/'predictions.npz', self_base/'self_only'/f'fold_{fold:02d}'/'predictions.npz'
            entries.append(dict(seed=seed, fold=fold, checkpoint=str(cp), checkpoint_sha256=sha(cp),
                c_predictions=str(pred), c_predictions_sha256=sha(pred),
                self_predictions=str(self_pred), self_predictions_sha256=sha(self_pred)))
    plan = dict(created_at=timestamp(), diagnostic_source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(baseline/'PROTOCOL.json'), entries=entries,
        target_group='Saved self-only correct, saved C wrong; all other patients are prespecified controls',
        fixed='Selected C parameters, features, candidate edges, original edge weights/maps, patient own values and readout',
        masks='All 64 binary post-aggregation content masks; all closed, half, each single modality closed reported separately',
        bypass='Skip whole graph residual but retain C output LayerNorm and readout; not retrained self-only model',
        value_replacements='Cohort mean and three patient permutations; fixed graph/weights/maps, original own diagonal restored',
        norm_controls='Repeat each replacement matching original foreign-message L2 per recipient/content; report degenerate vectors',
        permutation_seed=44019, labels='Original outer-test labels for descriptive diagnosis only; no fitting or checkpoint/tuning selection',
        output='Per-case recovery and harm, group/seed/class counts, label-informed 64-mask oracle ceilings',
        caveats=['Historically test-selected checkpoints; exploratory analysis, not independent validation',
            'Repeated patient evaluations across seeds are not independent patients',
            'Interventions change inference conditions; failures to recover do not prove errors originate only in own information',
            'Even successful replacement is functional sensitivity, not proof that a particular neighbor is incorrect',
            'Best masks selected with labels are diagnostic ceilings, not learned selector accuracy'])
    out.mkdir(parents=True)
    dump(out/'PLAN.json', plan)
    try:
        records = []
        for entry in entries:
            folder = out/f"seed_{entry['seed']}"/f"fold_{entry['fold']:02d}"
            records.append(run_checkpoint(cfg, dd, data, graphs, entry, folder))
            dump(out/'status.json', dict(state='running', completed=len(records), planned=30,
                seed=entry['seed'], fold=entry['fold'], updated_at=timestamp()))
            print(json.dumps(dict(event='checkpoint_complete', completed=len(records), seed=entry['seed'], fold=entry['fold'])), flush=True)
        dump(out/'records.json', records)
        dump(out/'audit.json', dict(passed=True, checkpoints_replayed=30, original_sources_data_candidates_verified=True,
            saved_c_and_self_rows_labels_matched=True, models_unchanged=True,
            identity_value_replacement_and_all_closed_verified=True,
            norm_match_degenerate_records=sum(sum(r['norm_match_degenerate_test_patient_modalities'].values()) for r in records)))
        dump(out/'COMPLETE.json', dict(completed_at=timestamp(), checkpoints=30))
        dump(out/'status.json', dict(state='complete', updated_at=timestamp(), completed=30))
    except BaseException as error:
        dump(out/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp()))
        raise


if __name__ == '__main__':
    main()
