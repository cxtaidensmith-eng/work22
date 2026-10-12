"""Independent audit of the completed ABIDE-5 C hyperparameter search.

The tuning driver, its metrics, and its ranking functions are never imported.
After completion and lock removal, replay each candidate's first three training
steps and all saved best/last weights. The historical baseline has no last.pt.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import configparser
import copy
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from Loss import MSCQGNLoss
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import setup


ROOT = Path(__file__).resolve().parent
CONFIG = 'Config/c_abide5_transfer.ini'
DEFAULT_OUT = ROOT/'Result/abide5_fine_tune_20260928'
DEFAULT_REF = ROOT/'Result/c_abide5_seed0_20260924'
BASE = dict(lr=.006, weight_decay=.0007, Drop_rate=.64,
            input_noise_std=.1, aux_loss_weight=1., Loss_rate=.0001,
            graph_weight_decay=0.)
COARSE = {
    'L003': dict(lr=.003),
    'L0015': dict(lr=.0015),
    'D050': dict(Drop_rate=.5),
    'N005': dict(input_noise_std=.05),
    'N015': dict(input_noise_std=.15),
    'W002': dict(weight_decay=.002),
    'A050': dict(aux_loss_weight=.5),
    'O00001': dict(Loss_rate=.00001),
    'G005': dict(graph_weight_decay=.0005),
}
REFINEMENT_RULES = [
    dict(name='R_L080', key='lr', operation='multiply', value=.8),
    dict(name='R_L120', key='lr', operation='multiply', value=1.2),
    dict(name='R_DMINUS', key='Drop_rate', operation='add_clamp', value=-.08, lower=.2, upper=.8),
    dict(name='R_DPLUS', key='Drop_rate', operation='add_clamp', value=.08, lower=.2, upper=.8),
    dict(name='R_N075', key='input_noise_std', operation='multiply', value=.75),
    dict(name='R_N125', key='input_noise_std', operation='multiply', value=1.25),
]
SCREEN = [1, 3, 6, 8]
REMAINING = [2, 4, 5, 7, 9, 10]
FOLDS = list(range(1, 11))
MEAN_KEYS = ('acc', 'macro_f1', 'balanced_acc', 'ads_auc', 'ads_ap',
             'ads_recall', 'cn_recall')
PARAMETERS = 768348
SAMPLES = 864
# Only first-three-step loss comparisons use this scale-aware float32 policy.
# Original failure and 19 bounded replays are preserved in audit_diagnostics.
# Exact initialization/RNG/identity/count checks and checkpoint tolerances stay
# unchanged. Repeated CUDA sparse reductions are not bitwise deterministic.
FIRST_THREE_LOSS_ATOL = 2e-5
FIRST_THREE_LOSS_RTOL = 1e-5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, context, atol=1e-12):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                f'{context}: dictionary keys differ')
        for key in expected:
            same(actual[key], expected[key], f'{context}.{key}', atol)
    elif isinstance(expected, (tuple, list)):
        require(len(actual) == len(expected), f'{context}: lengths differ')
        for index, value in enumerate(expected):
            same(actual[index], value, f'{context}[{index}]', atol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
                f'{context}: nonfinite value')
        require(abs(float(actual)-float(expected)) <= atol,
                f'{context}: {actual!r} != {expected!r}')
    else:
        require(actual == expected, f'{context}: {actual!r} != {expected!r}')


def verify_manifest(manifest):
    for filename, expected in manifest.items():
        require(digest(ROOT/filename) == expected, f'Hash mismatch: {filename}')


def compare_first_three_loss(actual, expected, context):
    require(math.isfinite(actual) and math.isfinite(expected), f'{context}: nonfinite loss')
    error = abs(actual-expected)
    limit = FIRST_THREE_LOSS_ATOL+FIRST_THREE_LOSS_RTOL*abs(expected)
    result = dict(actual=actual, expected=expected, absolute_error=error,
        relative_error=error/max(abs(expected), float(np.finfo(np.float32).tiny)),
        float32_ulp_error=error/abs(float(np.spacing(np.float32(expected)))),
        atol=FIRST_THREE_LOSS_ATOL, rtol=FIRST_THREE_LOSS_RTOL, allowed_error=limit,
        passed=error <= limit)
    require(result['passed'], f'{context}: error {error:.9g} exceeds '
            f'{FIRST_THREE_LOSS_ATOL:g}+{FIRST_THREE_LOSS_RTOL:g}*abs({expected:.9g})={limit:.9g}')
    return result


def calculate(logits, labels):
    """Derive metrics from integer confusion counts; ADS is index zero."""
    z, y = np.asarray(logits), np.asarray(labels)
    require(z.shape == (len(y), 2) and np.isfinite(z).all(), 'Invalid binary logits')
    require(np.array_equal(np.unique(y), [0, 1]), 'Both classes are required')
    pred = z.argmax(1)
    matrix = np.bincount(2*y.astype(np.int64)+pred, minlength=4).reshape(2, 2)
    support = matrix.sum(1)
    true_positive = matrix.diagonal().astype(float)
    recall = true_positive/support
    denominator = support+matrix.sum(0)
    f1 = np.divide(2*true_positive, denominator, out=np.zeros(2), where=denominator != 0)
    exponential = np.exp(z-z.max(1, keepdims=True))
    ads_probability = exponential[:, 0]/exponential.sum(1)
    ads_truth = (y == 0).astype(int)
    metrics = dict(acc=float(true_positive.sum()/len(y)), macro_f1=float(f1.mean()),
        balanced_acc=float(recall.mean()), ads_auc=float(roc_auc_score(ads_truth, ads_probability)),
        ads_ap=float(average_precision_score(ads_truth, ads_probability)),
        ads_recall=float(recall[0]), cn_recall=float(recall[1]),
        confusion_matrix=matrix.tolist())
    legacy = dict(acc=metrics['acc'], macro_f1=metrics['macro_f1'],
        f1=float(np.sum(f1*support)/support.sum()),
        auc=float((roc_auc_score(1-y, z[:, 0])+roc_auc_score(y, z[:, 1]))/2),
        per_class_recall=recall.tolist(), confusion_matrix=matrix.tolist())
    return metrics, legacy


def aggregate(rows, folds):
    chosen = [rows[f] for f in folds]
    result = {stage: {key: float(np.mean([r[stage][key] for r in chosen]))
                     for key in MEAN_KEYS} for stage in ('best', 'last')}
    result.update(folds=list(folds),
        late100_acc=float(np.mean([r['late100_acc'] for r in chosen])),
        acc_std=float(np.std([r['best']['acc'] for r in chosen])))
    return result


def ordered(names, rows, folds):
    def key(name):
        value = aggregate(rows[name], folds)
        return tuple(round(v, 12) for v in (
            value['best']['acc'], value['best']['macro_f1'],
            value['best']['balanced_acc'], value['late100_acc']))
    return sorted(names, key=key, reverse=True)


def refine(leader, initial):
    """Independently reconstruct prespecified local search and exact aliases."""
    specs, candidates, aliases = copy.deepcopy(initial), {}, {}
    rules = [('R_L080', 'lr', initial[leader]['lr']*.8),
             ('R_L120', 'lr', initial[leader]['lr']*1.2),
             ('R_DMINUS', 'Drop_rate', max(.2, min(.8, initial[leader]['Drop_rate']-.08))),
             ('R_DPLUS', 'Drop_rate', max(.2, min(.8, initial[leader]['Drop_rate']+.08))),
             ('R_N075', 'input_noise_std', initial[leader]['input_noise_std']*.75),
             ('R_N125', 'input_noise_std', initial[leader]['input_noise_std']*1.25)]
    for name, key, value in rules:
        candidate = {**initial[leader], key: round(value, 12)}
        duplicate = next((old for old, setting in specs.items() if setting == candidate), None)
        if duplicate is None:
            specs[name] = candidate
            candidates[name] = candidate
        else:
            aliases[name] = duplicate
    return specs, candidates, aliases


def check_config(path, spec):
    actual, original = configparser.ConfigParser(), configparser.ConfigParser()
    require(actual.read(path, encoding='utf-8'), f'Missing configuration: {path}')
    original.read(ROOT/CONFIG, encoding='utf-8')
    allowed = {('Modal' if k in ('Drop_rate', 'input_noise_std') else 'Optim', k.lower()): v
               for k, v in spec.items()}
    same(actual.sections(), original.sections(), f'{path}: sections')
    for section in original.sections():
        keys = set(original[section]) | {key for sec, key in allowed if sec == section}
        same(set(actual[section]), keys, f'{path}: options')
        for key in keys:
            if (section, key) in allowed:
                same(actual.getfloat(section, key), allowed[section, key], f'{path}: {key}')
            else:
                same(actual[section][key], original[section][key], f'{path}: {key}')


def configure(cfg, spec):
    conf = copy.copy(cfg)
    for key, value in spec.items():
        setattr(conf, key, value)
    return conf


def is_graph(name):
    return name.startswith('propagation.') and not name.startswith('propagation.output_norm.')


def optimizer_groups(model, spec):
    named = list(model.named_parameters())
    result = []
    for index, graph in enumerate((False, True)):
        selected = [(name, param) for name, param in named if is_graph(name) == graph]
        result.append(dict(group=index, lr=spec['lr'],
            weight_decay=spec['graph_weight_decay'] if graph else spec['weight_decay'],
            parameters=sum(param.numel() for _, param in selected),
            names=[name for name, _ in selected]))
    return result


def make_optimizer(model, spec):
    named = list(model.named_parameters())
    return torch.optim.Adam([
        dict(params=[param for name, param in named if not is_graph(name)], weight_decay=spec['weight_decay']),
        dict(params=[param for name, param in named if is_graph(name)], weight_decay=spec['graph_weight_decay']),
    ], lr=spec['lr'], weight_decay=spec['weight_decay'])


def reset_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameter_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu()
        result.update(name.encode())
        result.update(str((str(value.dtype), tuple(value.shape))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def curve_for(folder):
    with (folder/'epochs.csv').open(encoding='utf-8', newline='') as stream:
        curve = list(csv.DictReader(stream))
    same([int(row['epoch']) for row in curve], list(range(1, 1001)),
         f'{folder}: consecutive 1000 epochs')
    require(np.isfinite([[float(v) for v in row.values()] for row in curve]).all(),
            f'{folder}: nonfinite training trace')
    return curve


def first_three(name, conf, dd, data, graphs, spec, folder, preflight):
    reset_seed()
    model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(conf.Device)
    same(sum(p.numel() for p in model.parameters()), PARAMETERS, f'{name}: parameter count')
    same(model.noise_scale, spec['input_noise_std']/.1, f'{name}: actual noise multiplier')
    torch.testing.assert_close(model._modal_noise_std,
        torch.full((5,), .3, dtype=torch.float32, device=conf.Device), atol=0, rtol=0)
    torch.testing.assert_close(model.modal_gate_logit,
        torch.tensor([-2., -2., -2., -1., -1.], device=conf.Device), atol=0, rtol=0)
    initialization_hash = parameter_digest(model)
    cpu_rng_hash = tensor_digest(torch.get_rng_state())
    cuda_rng_hash = tensor_digest(torch.cuda.get_rng_state(conf.Device))
    initial = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
    model.eval()
    with torch.no_grad():
        initial_logits = model(data['Feature'])[0].detach().cpu().clone()
    same(tensor_digest(torch.get_rng_state()), cpu_rng_hash, f'{name}: evaluation retains CPU RNG')
    same(tensor_digest(torch.cuda.get_rng_state(conf.Device)), cuda_rng_hash, f'{name}: evaluation retains CUDA RNG')
    optimizer = make_optimizer(model, spec)
    expected_groups = optimizer_groups(model, spec)
    same(read(folder/'optimizer_groups.json'), expected_groups, f'{name}: optimizer manifest')
    criterion = MSCQGNLoss(dd, conf.Device, rate=spec['Loss_rate'], aux_loss_weight=spec['aux_loss_weight'])
    same(criterion.CE_loss.label_smoothing, .05, f'{name}: retained label smoothing')
    torch.testing.assert_close(criterion.weight, data['Feature'].new_tensor([467/864, 397/864]),
                               atol=1e-7, rtol=0)
    same(criterion.rate, spec['Loss_rate'], f'{name}: effective orthogonality weight')
    same(criterion.aux_loss_weight, spec['aux_loss_weight'], f'{name}: effective auxiliary weight')
    curve = curve_for(folder)
    fit, test = data['Mask'][0]
    replay = []
    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, embeddings, auxiliary = model(data['Feature'])
        loss = criterion(logits, data['Label'], fit, embeddings, auxiliary)
        require(torch.isfinite(loss).item(), f'{name}: nonfinite first-three loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            z = model(data['Feature'])[0]
            test_acc = int((z[test].argmax(1) == data['Label'][test]).sum())/int(test.sum())
        loss_key = 'train_loss' if name == 'BASE' else 'training_loss'
        error = abs(float(loss.detach())-float(curve[step][loss_key]))
        comparison = compare_first_three_loss(float(loss.detach()), float(curve[step][loss_key]),
                                              f'{name}: first-three training loss at epoch {step+1}')
        same(test_acc, float(curve[step]['test_acc']), f'{name}: first-three test ACC')
        correct_count = int((z[test].argmax(1) == data['Label'][test]).sum())
        same(correct_count, round(float(curve[step]['test_acc'])*int(test.sum())),
             f'{name}: first-three integer correct count')
        same(optimizer.param_groups[0]['lr'], float(curve[step]['lr']), f'{name}: first-three LR')
        if 'gradient_norm' in curve[step]:
            np.testing.assert_allclose(float(norm), float(curve[step]['gradient_norm']), atol=2e-5, rtol=2e-5)
        replay.append(dict(epoch=step+1, loss=float(loss.detach()), test_acc=test_acc,
                           correct_count=correct_count, lr=optimizer.param_groups[0]['lr'],
                           loss_absolute_difference=error, training_loss_comparison=comparison))
    if preflight is not None:
        same(preflight['spec'], spec, f'{name}: recorded preflight settings')
        same(preflight['parameters'], PARAMETERS, f'{name}: recorded preflight size')
        require(preflight['passed'] is True, f'{name}: preflight failed')
        same(preflight['init_parameter_sha256'], initialization_hash, f'{name}: initialization hash')
        same(preflight['cpu_rng_sha256'], cpu_rng_hash, f'{name}: post-initialization CPU RNG')
        same(preflight['cuda_rng_sha256'], cuda_rng_hash, f'{name}: post-initialization CUDA RNG')
        same(preflight['initialization_and_eval_equal_BASE'], True, f'{name}: initialization/eval check')
        same(preflight['optimizer_groups'], expected_groups, f'{name}: preflight optimizer')
        np.testing.assert_allclose(preflight['effective_noise_sd'],
            (model._modal_noise_std*model.noise_scale).detach().cpu().numpy(), atol=1e-7, rtol=0)
        for actual, expected in zip(replay, preflight['first3']):
            for key in ('epoch', 'test_acc', 'lr'):
                same(actual[key], expected[key], f'{name}: preflight step {actual["epoch"]} {key}')
            actual['preflight_loss_comparison'] = compare_first_three_loss(actual['loss'], expected['loss'],
                f'{name}: preflight loss at epoch {actual["epoch"]}')
            same(actual['correct_count'], round(expected['test_acc']*int(test.sum())),
                 f'{name}: preflight integer correct count')
        same(len(preflight['first3']), 3, f'{name}: preflight length')
    result = dict(passed=True, first_three=replay, effective_noise_sd=(model._modal_noise_std*model.noise_scale).detach().cpu().tolist(),
                  optimizer_groups=expected_groups, init_parameter_sha256=initialization_hash,
                  cpu_rng_sha256=cpu_rng_hash, cuda_rng_sha256=cuda_rng_hash)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss, z
    torch.cuda.empty_cache()
    return result, initial, initial_logits


def audit_fold(name, fold, folder, model, spec, data, source_rows):
    baseline = name == 'BASE'
    record = read(folder/'complete.json')
    same(record['parameters'], PARAMETERS, f'{folder}: parameter count')
    same(record['fold'], fold, f'{folder}: fold')
    if baseline:
        same(record['mode'], 'content', f'{folder}: model mode')
        same(record['seed'], 0, f'{folder}: seed')
    else:
        same(record['arm'], name, f'{folder}: arm')
        require(record['audit_passed'] is True, f'{folder}: training audit failed')
        same(read(folder/'effective_settings.json'), spec, f'{folder}: effective settings')
    same(read(folder/'optimizer_groups.json'), optimizer_groups(model, spec), f'{folder}: actual optimizer groups')
    curve = curve_for(folder)
    acc = np.asarray([float(row['test_acc']) for row in curve])
    best_epoch = int(acc.argmax())+1
    same(record['best']['epoch'], best_epoch, f'{folder}: first maximum test ACC')
    for row in curve:
        tick = int(row['epoch'])-1
        lr = spec['lr'] if tick < 20 else .0001+(spec['lr']-.0001)*(1+math.cos(math.pi*(tick-20)/980))/2
        same(float(row['lr']), lr, f'{folder}: epoch {tick+1} scheduler')
    test = data['Mask'][fold-1][1]
    labels = data['Label'][test].cpu().numpy()
    indices = source_rows[test.cpu().numpy()]
    require(len(labels) in (86, 87), f'{folder}: incorrect fold size')
    np.testing.assert_allclose(acc*len(labels), np.rint(acc*len(labels)), atol=1e-10, rtol=0)
    result = dict(name=name, fold=fold, best_epoch=best_epoch, parameters=PARAMETERS,
                  late100_acc=float(acc[-100:].mean()))
    if not baseline:
        same(record['late100_mean_acc'], result['late100_acc'], f'{folder}: late100')
    predictions, details = {}, {}
    expected_noise = model._modal_noise_std.detach().clone()
    fine = None if baseline else read(folder/'fine_audit.json')
    if fine is not None:
        same(fine['settings'], spec, f'{folder}: fine audit settings')
        for key, expected in dict(name=name, fold=fold, parameters=PARAMETERS, epoch_count=1000,
                                  first_best_epoch=best_epoch, passed=True).items():
            same(fine[key], expected, f'{folder}: fine audit {key}')
        np.testing.assert_allclose(fine['effective_noise_sd'],
            (model._modal_noise_std*model.noise_scale).detach().cpu().numpy(), atol=1e-7, rtol=0)
        same([item['stage'] for item in fine['checks']], ['best', 'last'], f'{folder}: fine audit stages')
    for stage in ('best', 'last'):
        filename = ('predictions.npz' if baseline else 'best_predictions.npz') if stage == 'best' else 'last_predictions.npz'
        with np.load(folder/filename) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        np.testing.assert_array_equal(saved['labels'], labels)
        explicit_ids = 'row_index' in saved
        if explicit_ids:
            np.testing.assert_array_equal(saved['row_index'], indices)
        else:
            require(not baseline and stage == 'last', f'{folder}: missing prediction row identity')
        metrics, legacy = calculate(saved['logits'], labels)
        same(metrics['acc'], float(acc[best_epoch-1 if stage == 'best' else -1]), f'{folder}: {stage} curve match')
        if baseline or stage == 'best':
            for key, value in legacy.items():
                same(record[stage][key], value, f'{folder}: {stage} {key}')
        else:
            same(record['last']['test_acc'], metrics['acc'], f'{folder}: last record')
            same(record['last']['epoch'], 1000, f'{folder}: last epoch')
        replayed = not (baseline and stage == 'last')
        error = None
        if replayed:
            cp = torch.load(folder/f'{stage}.pt', map_location=data['Feature'].device, weights_only=False)
            same(cp['fold'], fold, f'{folder}: {stage} checkpoint fold')
            same(cp['seed'], 0, f'{folder}: {stage} checkpoint seed')
            same(cp['mode'] if baseline else cp['arm'], 'content' if baseline else name,
                 f'{folder}: {stage} checkpoint identity')
            if stage == 'best':
                same(cp['best'], record['best'], f'{folder}: checkpoint best metadata')
            else:
                same(cp['epoch'], 1000, f'{folder}: checkpoint final epoch')
            model.load_state_dict(cp['state_dict'], strict=True)
            torch.testing.assert_close(model._modal_noise_std, expected_noise, atol=0, rtol=0)
            model.eval()
            with torch.inference_mode():
                z = model(data['Feature'])[0][test].cpu().numpy()
            np.testing.assert_allclose(z, saved['logits'], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(z.argmax(1), saved['logits'].argmax(1))
            error = float(np.max(np.abs(z-saved['logits'])))
            if fine is not None:
                check = fine['checks'][0 if stage == 'best' else 1]
                same(check['passed'], True, f'{folder}: {stage} fine audit passed')
                same(check['sha256'], digest(folder/f'{stage}.pt'), f'{folder}: {stage} recorded weight hash')
                same(check['metrics'], metrics, f'{folder}: {stage} fine audit metrics')
            del cp, z
        result[stage] = metrics
        predictions[stage] = dict(logits=saved['logits'], labels=labels, row_index=indices,
                                  fold=np.full(len(labels), fold, dtype=int))
        details[stage] = dict(checkpoint_replayed=replayed, maximum_logit_absolute_error=error,
            explicit_row_index=explicit_ids,
            limitation='Historical BASE has no last.pt; its saved final predictions were checked without weight replay' if not replayed else None)
    return result, predictions, details


def check_archive(archive_path, dd, data, graphs):
    with np.load(archive_path) as archive:
        np.testing.assert_array_equal(archive['source_row'], dd['Index'])
        np.testing.assert_array_equal(archive['labels'], data['Label'].cpu().numpy())
        np.testing.assert_array_equal(archive['test_masks'], torch.stack([m for _, m in data['Mask']]).cpu().numpy())
        for i, graph in enumerate(graphs):
            np.testing.assert_array_equal(archive[f'edge_{i}'], graph['edge'].numpy())
            np.testing.assert_array_equal(archive[f'neighbors_{i}'], graph['neighbors'].numpy())


def check_oof(path, expected):
    with np.load(path) as saved:
        same(set(saved.files), set(expected), f'{path}: OOF fields')
        for key, value in expected.items():
            np.testing.assert_array_equal(saved[key], value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REF)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    out, ref = args.output.resolve(), args.reference.resolve()
    require((out/'COMPLETE.json').is_file(), 'Training is incomplete; GPU audit is not permitted yet')
    require(not (out/'RUNNING.lock').exists(), 'Training lock remains; wait for training exit')
    completion = read(out/'COMPLETE.json')
    require(completion['all_audits_passed'] is True, 'Training completion indicates failed audits')
    diagnostics = out/'audit_diagnostics'
    evidence_path = diagnostics/'float32_first3_diagnosis.json'
    failed_manifest_path = diagnostics/'failed_attempt_001_float32/manifest.json'
    evidence = read(evidence_path)
    require(evidence['exact_initialization_and_initial_rng'] is True
            and evidence['exact_recorded_correct_prediction_counts'] is True,
            'Numerical tolerance diagnostic does not establish unchanged initialization/RNG/prediction counts')
    for item in evidence['repeated_L0015']:
        require(item['exact_random_states'] is True and item['exact_integer_predictions'] is True,
                'Repeated numerical diagnostic changes RNG or integer predictions')
    failed_manifest = read(failed_manifest_path)
    for item in failed_manifest['files'].values():
        same(digest(ROOT/item['backup']), item['sha256'], 'Preserved first failed audit evidence')
    plan, specs, analysis = (read(out/filename) for filename in ('PLAN.json', 'all_settings.json', 'analysis.json'))
    refinement, selection = read(out/'refinement.json'), read(out/'selection.json')
    for key, expected in dict(base=BASE, coarse=COARSE, seed=0, epochs=1000,
            screen_folds=SCREEN, maximum_new_runs=150, samples=SAMPLES,
            parameters=PARAMETERS, ranking_round_decimals=12,
            refinement_rules=REFINEMENT_RULES, remaining_folds=REMAINING).items():
        same(plan[key], expected, f'Frozen plan: {key}')
    same(list(plan['coarse']), list(COARSE), 'Frozen coarse insertion order')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    required_sources = ('abide5_fine_tune.py', 'diagnose_abide_capacity.py', 'relation_experiment.py',
        'Model/relation_sheaf.py', 'Model/network.py', 'Model/models.py', 'Loss/loss_fn.py',
        'Utils/utils.py', 'Utils/data_load.py', 'Utils/graph_load.py', 'Utils/relation_optimizer.py', CONFIG)
    for filename in required_sources:
        require(filename in plan['code_sha256'], f'Unpinned source dependency: {filename}')
    for filename, expected in plan['code_sha256'].items():
        same(digest(out/'source_snapshot'/filename), expected, f'Source snapshot: {filename}')
    required_references = [ref/name for name in ('PLAN.json', 'PROTOCOL.json', 'splits_and_graphs.npz')]
    for fold in FOLDS:
        required_references.extend(ref/'content'/f'fold_{fold:02d}'/name for name in
            ('best.pt', 'epochs.csv', 'complete.json', 'optimizer_groups.json', 'predictions.npz', 'last_predictions.npz'))
    for path in required_references:
        require(str(path.relative_to(ROOT)) in plan['reference_sha256'], f'Unpinned reused artifact: {path}')
    same(digest(out/'data_splits_graphs.npz'), plan['split_archive_sha256'], 'Frozen split archive')
    historical = read(ref/'PLAN.json')
    same(plan['baseline_protocol'], historical['protocol'], 'Original baseline protocol')

    # The first GPU access happens only after completion/lock/manifest checks.
    cfg, dd, data, graphs, paths = setup(args.cuda, CONFIG, 10)
    require(cfg.Device.type == 'cuda', 'Expected CUDA checkpoint replay')
    same(dd['Sample_Num'], SAMPLES, 'Samples')
    same(dd['Class_Names'], ['ADS', 'CN'], 'Label order')
    same(dd['Modal_Name'], ['PHENO', 'ANAT', 'FUNC', 'MRI', 'fMRI'], 'Modality identity')
    same([len(v) for v in dd['Modal_Index']], [48, 6, 10, 150, 256], 'Modality dimensions')
    same(dd['Feature_Num'], 470, 'Feature count')
    same(torch.bincount(data['Label']).cpu().tolist(), [397, 467], 'Class counts')
    for path in paths:
        same(digest(path), historical['protocol']['data_sha256'][Path(path).name], f'Data hash: {path}')
    for key, expected in historical['protocol']['original_config'].items():
        same(getattr(cfg, key), expected, f'Unmodified original config: {key}')
    source_rows = np.asarray(dd['Index'])
    require(len(source_rows) == len(np.unique(source_rows)) == SAMPLES, 'Source row identity is not unique')
    raw = pd.read_csv(ROOT/'DATASET/ABIDE-5/ABIDE_processed_data_modal.csv')
    shuffled = raw.sample(frac=1, random_state=0)
    np.testing.assert_array_equal(source_rows, shuffled.index.to_numpy())
    np.testing.assert_array_equal(data['Label'].cpu().numpy(), shuffled.iloc[:, -1].to_numpy()-1)
    np.testing.assert_array_equal(data['Feature'].cpu().numpy(), shuffled.iloc[:, :-1].to_numpy(dtype=np.float32))
    masks = torch.stack([m for _, m in data['Mask']]).cpu().numpy()
    independent_masks = []
    for train_indices, test_indices in StratifiedKFold(10, shuffle=True, random_state=0).split(
            shuffled.iloc[:, :-1], shuffled.iloc[:, -1]-1):
        mask = np.zeros(SAMPLES, dtype=bool)
        mask[test_indices] = True
        independent_masks.append(mask)
    np.testing.assert_array_equal(independent_masks, masks)
    np.testing.assert_array_equal(masks.sum(0), np.ones(SAMPLES, dtype=int))
    for train, test in data['Mask']:
        require(not (train & test).any().item() and (train | test).all().item(), 'Invalid train/test masks')
    check_archive(ref/'splits_and_graphs.npz', dd, data, graphs)
    check_archive(out/'data_splits_graphs.npz', dd, data, graphs)
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    same(graph_hash, historical['protocol']['candidate_hash'], 'Original fixed graph support')

    preflight = read(out/'preflight.json')
    same(preflight['passed'], True, 'Preflight passed')
    same(preflight['verified_reference_checkpoints'], 10, 'Preflight reference replay count')
    same([item['fold'] for item in preflight['baseline_checkpoints']], FOLDS, 'Preflight reference folds')
    for item in preflight['baseline_checkpoints']:
        folder = ref/'content'/f'fold_{item["fold"]:02d}'
        same(item['passed'], True, 'Preflight reference passed')
        same(item['best_epoch'], read(folder/'complete.json')['best']['epoch'], 'Preflight reference epoch')
        same(item['checkpoint_sha256'], digest(folder/'best.pt'), 'Preflight reference checkpoint hash')
    preflight_checks = preflight['checks']
    require(isinstance(preflight_checks, dict), 'Expected per-candidate preflight dictionary')
    same(set(preflight_checks), set(specs), 'Preflight covers all realized settings')
    config_hashes = read(out/'config_sha256.json')
    same(set(config_hashes), set(specs), 'Config hashes cover all settings')
    for name, expected_hash in config_hashes.items():
        same(digest(out/'configs'/f'{name}.ini'), expected_hash, f'{name}: frozen config hash')
    rows, predictions, details, replay3 = {}, {}, {}, {}
    baseline_initial = None
    baseline_initial_logits = None
    new_runs = 0
    same({folder.name for folder in (out/'runs').iterdir() if folder.is_dir()},
         set(specs)-{'BASE'}, 'Recognized candidate directories')
    for name, spec in specs.items():
        same(set(spec), set(BASE), f'{name}: tunable fields')
        require(all(math.isfinite(float(v)) and float(v) >= 0 for v in spec.values()), f'{name}: invalid setting')
        config_path = out/'configs'/f'{name}.ini'
        same(digest(config_path), config_hashes[name], f'{name}: config hash')
        check_config(config_path, spec)
        conf = configure(cfg, spec)
        parent = ref/'content' if name == 'BASE' else out/'runs'/name
        replay3[name], initial, initial_logits = first_three(name, conf, dd, data, graphs, spec,
                                                            parent/'fold_01', preflight_checks[name])
        if name == 'BASE':
            baseline_initial = initial
            baseline_initial_logits = initial_logits
        else:
            same(set(initial), set(baseline_initial), f'{name}: unchanged trainable structure')
            for key in initial:
                torch.testing.assert_close(initial[key], baseline_initial[key], atol=0, rtol=0)
            torch.testing.assert_close(initial_logits, baseline_initial_logits, atol=2e-7, rtol=1e-6)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
        same(sum(p.numel() for p in model.parameters()), PARAMETERS, f'{name}: model size')
        rows[name], predictions[name], details[name] = {}, {}, {}
        for fold_folder in sorted(parent.glob('fold_*')):
            require((fold_folder/'complete.json').is_file(), f'Incomplete fold: {fold_folder}')
            fold = int(fold_folder.name.split('_')[-1])
            require(fold in FOLDS, f'Unknown fold: {fold}')
            row, prediction, detail = audit_fold(name, fold, fold_folder, model, spec, data, source_rows)
            rows[name][fold], predictions[name][fold], details[name][fold] = row, prediction, detail
            new_runs += int(name != 'BASE')
            print(json.dumps(dict(audit=name, fold=fold, passed=True)), flush=True)
        require(set(SCREEN).issubset(rows[name]), f'{name}: incomplete screen')
        del model, initial, initial_logits
        torch.cuda.empty_cache()

    initial_specs = {'BASE': dict(BASE), **{name: {**BASE, **change} for name, change in COARSE.items()}}
    coarse_order = ordered(list(initial_specs), rows, SCREEN)
    leader = coarse_order[0]
    expected_specs, expected_candidates, expected_aliases = refine(leader, initial_specs)
    same(specs, expected_specs, 'Independently reconstructed candidate settings')
    same(list(specs), list(expected_specs), 'Stable settings insertion order')
    same(refinement, dict(leader=leader, coarse_order=coarse_order,
                         candidates=expected_candidates, aliases=expected_aliases), 'Adaptive refinement')
    candidate_order = ordered([name for name in specs if name != 'BASE'], rows, SCREEN)
    screen_order = ordered(list(specs), rows, SCREEN)
    cutoff = aggregate(rows[candidate_order[1]], SCREEN)['best']['acc']
    finalists = [name for name in candidate_order
                 if name in candidate_order[:2] or aggregate(rows[name], SCREEN)['best']['acc'] >= cutoff-1e-12]
    expected_runs = 4*(len(specs)-1)+6*len(finalists)
    same(selection, dict(screen_order=screen_order, candidate_order=candidate_order, cutoff_acc=cutoff,
                        finalists=finalists, screen_folds=SCREEN, remaining_folds=REMAINING,
                        expected_new_runs=expected_runs), 'Fourfold selection including all ACC ties')
    same(new_runs, expected_runs, 'Training budget')
    require(new_runs <= 150, 'Search exceeds prespecified bound')
    completed_paths = list((out/'runs').glob('*/fold_*/complete.json'))
    same(new_runs, len(completed_paths), 'No unrecognized new runs')
    actual_replays = sum(int(stage['checkpoint_replayed'])
        for candidate in details.values() for fold_detail in candidate.values() for stage in fold_detail.values())
    same(actual_replays, 2*new_runs+10, 'Actual checkpoint replay count')
    for name in specs:
        same(sorted(rows[name]), FOLDS if name == 'BASE' or name in finalists else SCREEN,
             f'{name}: exact completed fold set')
    full_names = [name for name in specs if name == 'BASE' or name in finalists]
    full = {name: aggregate(rows[name], FOLDS) for name in full_names}
    full_order = ordered(full_names, rows, FOLDS)
    retained = full_order[0]
    stages = {name: dict(screen=aggregate(rows[name], SCREEN), remaining=aggregate(rows[name], REMAINING),
        screen_delta_acc=aggregate(rows[name], SCREEN)['best']['acc']-aggregate(rows['BASE'], SCREEN)['best']['acc'],
        remaining_delta_acc=aggregate(rows[name], REMAINING)['best']['acc']-aggregate(rows['BASE'], REMAINING)['best']['acc'])
        for name in full_names}
    same(read(out/'screen_vs_remaining.json'), stages, 'Separate screen/remaining artifact')
    pairs = {}
    for name in full_names:
        for stage in ('best', 'last'):
            ids = np.concatenate([predictions[name][fold][stage]['row_index'] for fold in FOLDS])
            require(len(ids) == len(np.unique(ids)) == SAMPLES, f'{name}/{stage}: invalid OOF coverage')
            np.testing.assert_array_equal(np.sort(ids), np.sort(source_rows))
        oof = {key: np.concatenate([predictions[name][fold]['best'][key] for fold in FOLDS])
               for key in ('labels', 'logits', 'row_index', 'fold')}
        baseline_logits = np.concatenate([predictions['BASE'][fold]['best']['logits'] for fold in FOLDS])
        correct = oof['logits'].argmax(1) == oof['labels']
        old_correct = baseline_logits.argmax(1) == oof['labels']
        pooled, _ = calculate(oof['logits'], oof['labels'])
        fold_deltas = [int(np.trace(rows[name][fold]['best']['confusion_matrix']))
                       -int(np.trace(rows['BASE'][fold]['best']['confusion_matrix'])) for fold in FOLDS]
        pairs[name] = dict(corrected=int(np.sum(correct & ~old_correct)),
            harmed=int(np.sum(~correct & old_correct)), wins=sum(delta > 0 for delta in fold_deltas),
            ties=sum(delta == 0 for delta in fold_deltas), losses=sum(delta < 0 for delta in fold_deltas), pooled=pooled)
        check_oof(out/'oof'/f'{name}.npz', oof)
        if name == retained:
            check_oof(out/'retained_oof_predictions.npz', oof)
    expected_analysis = dict(retained=retained, full_order=full_order, full=full, pairs=pairs,
        records=[rows[name][fold] for name in full_names for fold in FOLDS], settings=specs[retained],
        selection=selection, new_runs=new_runs, verified_new_checkpoints=2*new_runs, screen_vs_remaining=stages)
    same(analysis, expected_analysis, 'Complete analysis')
    for key, expected in dict(retained=retained, new_runs=new_runs, all_audits_passed=True,
                              verified_new_checkpoints=2*new_runs).items():
        same(completion[key], expected, f'Completion: {key}')
    summary = read(out/'summary.json')['candidates']
    same(set(summary), set(specs), 'Summary coverage')
    for name in specs:
        same(summary[name], aggregate(rows[name], sorted(rows[name])), f'{name}: summary')
    exported = read(out/'best_configuration.json')
    for key, expected in dict(name=retained, overrides=specs[retained],
            model='RelationSheafModel(mode=content)', graph_weight_decay=specs[retained]['graph_weight_decay'],
            k=10, seed=0, epochs=1000, parameters=PARAMETERS).items():
        same(exported[key], expected, f'Exported best configuration: {key}')
    require(Path(exported['config']).resolve() == out/'configs'/f'{retained}.ini', 'Exported config path')
    require((ROOT/exported['driver']).resolve() == ROOT/'abide5_fine_tune.py', 'Exported driver path')
    same(exported['base_config'], CONFIG, 'Exported base configuration')
    np.testing.assert_allclose(exported['effective_noise_sd'], [3*specs[retained]['input_noise_std']]*5,
                               atol=1e-12, rtol=0)
    status = read(out/'status.json')
    for key, expected in dict(state='complete', retained=retained, new_runs=new_runs).items():
        same(status[key], expected, f'Final status: {key}')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    for name, expected_hash in config_hashes.items():
        same(digest(out/'configs'/f'{name}.ini'), expected_hash, f'{name}: final config hash')
    artifacts = ('PLAN.json', 'all_settings.json', 'refinement.json', 'selection.json', 'analysis.json',
        'COMPLETE.json', 'data_splits_graphs.npz', 'preflight.json', 'config_sha256.json',
        'retained_oof_predictions.npz', 'best_configuration.json', 'summary.json', 'screen_vs_remaining.json')
    report = dict(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=digest(__file__), independent_of_tuning_metrics_and_ranking=True,
        first_three_loss_numeric_policy=dict(atol=FIRST_THREE_LOSS_ATOL, rtol=FIRST_THREE_LOSS_RTOL,
            rule='abs(actual-reference) <= atol + rtol*abs(reference); only first-three training/preflight losses',
            reason='Bounded float32 CUDA replay shows identical initialization, random states and integer predictions with reduction-order loss variation.',
            observed_replay_count=len(evidence['records']),
            evidence_path=str(evidence_path), evidence_sha256=digest(evidence_path),
            failed_attempt_manifest=str(failed_manifest_path), failed_attempt_manifest_sha256=digest(failed_manifest_path),
            unchanged_checks='Exact initializer/RNG, source/row/label identity, integer predictions and counts, ranking, scheduler, and original saved-weight/logit replay tolerances.'),
        device=str(cfg.Device), new_runs=new_runs, replayed_new_checkpoints=2*new_runs,
        replayed_baseline_best_checkpoints=10, baseline_last_prediction_files_verified=10,
        first_three_training_replays=replay3, prediction_files_verified=2*(new_runs+10),
        parameters=PARAMETERS, samples=SAMPLES, class_counts=[397, 467], graph_candidate_hash=graph_hash,
        finalists=finalists, retained=retained, full=full, pairs=pairs, screen_vs_remaining=stages,
        checkpoint_details=details, artifact_sha256={filename: digest(out/filename) for filename in artifacts},
        limitations=['Historical BASE has no last.pt; its final predictions are verified without replaying final weights.',
            'New last prediction files omit row_index; inference replay verifies fixed test-mask order.',
            'Unique CSV row indices do not establish clinical patient independence.',
            'ABIDE-5 largely overlaps ABIDE; this is not an independent external cohort.',
            'Historical test-selected epochs and adaptive reuse of test folds remain exploratory, not independent validation or a global optimum.'])
    target = out/'independent_audit.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(dict(passed=True, retained=retained, new_runs=new_runs,
        replayed_checkpoints=2*new_runs+10, mean_acc=full[retained]['best']['acc'],
        output=str(target))), flush=True)


if __name__ == '__main__':
    main()
