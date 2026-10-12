"""Passive G2 training observer; no changes to forward values, losses or RNG."""
import csv
import math
import torch


def tensor_norm(value):
    return float(value.detach().double().norm())


def combined_norm(tensors):
    tensors = list(tensors)
    if not tensors:
        return 0.0
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


class GraphDiagnostics:
    def __init__(self, model, path, weight_decay, optimizer=None):
        self.model, self.weight_decay = model, weight_decay
        self.optimizer = optimizer
        self.enabled, self.phase = False, 'train'
        self.row, self.snapshots, self.rows = {}, {}, []
        self.stream = open(path, 'w', newline='', encoding='utf-8')
        self.writer = None
        self.groups = {}
        for name, parameter in model.named_parameters():
            if name.startswith('propagation.value.'):
                group = 'value'
            elif name.startswith('propagation.output.'):
                group = 'output'
            elif name.startswith(('propagation.relation_id.', 'propagation.content_id.', 'propagation.conditioner.')):
                group = 'maps'
            elif name.startswith(('propagation.edge_query.', 'propagation.edge_similarity', 'propagation.edge_bias')):
                group = 'edges'
            elif name == 'propagation.step_logit':
                group = 'step'
            elif name.startswith('propagation.input_norm.'):
                group = 'graph_input_norm'
            else:
                # Includes the output LayerNorm retained in the no-graph model.
                group = 'patient_path'
            self.groups.setdefault(group, []).append(parameter)
        prop = model.propagation
        if not hasattr(prop, 'step_logit'):
            self.stream.close()
            raise ValueError('Graph diagnostics requires an active propagation module')
        self.hooks = [prop.register_forward_pre_hook(self._input),
                      prop.value.register_forward_hook(self._value),
                      prop.output.register_forward_pre_hook(self._diffused),
                      prop.output.register_forward_hook(self._output)]

    def _input(self, module, args):
        if self.enabled:
            self.row[f'{self.phase}_token_norm'] = tensor_norm(args[0])

    def _value(self, module, args, output):
        if self.enabled:
            self.row[f'{self.phase}_projected_value_norm'] = tensor_norm(output)

    def _diffused(self, module, args):
        if self.enabled:
            self.row[f'{self.phase}_diffused_gelu_norm'] = tensor_norm(args[0])

    def _output(self, module, args, output):
        if self.enabled:
            step = float(self.model.propagation.step_logit.detach().sigmoid())
            update_norm = step * tensor_norm(output)
            self.row[f'{self.phase}_step'] = step
            self.row[f'{self.phase}_update_norm'] = update_norm
            self.row[f'{self.phase}_update_ratio'] = update_norm / max(self.row[f'{self.phase}_token_norm'], 1e-30)

    def begin(self, epoch, last_epoch, lr):
        self.enabled = epoch <= 20 or epoch % 10 == 0 or epoch == last_epoch
        self.phase = 'train'
        self.row = dict(epoch=epoch, lr=lr)
        self.snapshots = {}

    def after_backward(self):
        if self.enabled:
            for group, params in self.groups.items():
                self.row[f'{group}_weight_before'] = combined_norm(params)
                self.row[f'{group}_loss_grad_before_clip'] = combined_norm(p.grad for p in params if p.grad is not None)

    def before_step(self, global_grad_norm):
        if not self.enabled:
            return
        self.row['global_loss_grad_before_clip'] = float(global_grad_norm)
        decay_by_parameter = None if self.optimizer is None else {
            id(p): group['weight_decay'] for group in self.optimizer.param_groups for p in group['params']}
        for group, params in self.groups.items():
            loss_grad = combined_norm(p.grad for p in params if p.grad is not None)
            # Adam's coupled weight decay is added after explicit gradient clipping.
            if decay_by_parameter is None:
                decay_grad = self.weight_decay * combined_norm(p for p in params if p.grad is not None)
            else:
                decay_grad = math.sqrt(sum(
                    decay_by_parameter[id(p)] ** 2 * float(p.detach().double().square().sum())
                    for p in params if p.grad is not None))
            self.row[f'{group}_loss_grad_after_clip'] = loss_grad
            self.row[f'{group}_decay_grad'] = decay_grad
            self.row[f'{group}_decay_to_loss_grad_ratio'] = decay_grad / max(loss_grad, 1e-30)
            self.snapshots[group] = [p.detach().clone() for p in params]

    def after_step(self):
        if self.enabled:
            for group, params in self.groups.items():
                self.row[f'{group}_weight_after'] = combined_norm(params)
                self.row[f'{group}_actual_update_norm'] = combined_norm(
                    p.detach()-before for p, before in zip(params, self.snapshots[group]))
        self.snapshots = {}
        self.phase = 'eval'

    def finish(self, train_loss, test_loss, test_acc):
        if not self.enabled:
            return
        self.row.update(train_loss=float(train_loss), test_loss=float(test_loss), test_acc=test_acc)
        if not all(math.isfinite(value) for value in self.row.values()):
            raise FloatingPointError('Nonfinite diagnostic value')
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(self.row))
            self.writer.writeheader()
        self.writer.writerow(self.row)
        self.stream.flush()
        self.rows.append(dict(self.row))

    def summary(self):
        if not self.rows:
            return dict(observations=0)
        thresholds = {}
        for threshold in (.01, .001, .0001):
            hits = [r for r in self.rows if r['eval_update_ratio'] < threshold]
            thresholds[str(threshold)] = hits[0]['epoch'] if hits else None
        return dict(observations=len(self.rows), first=self.rows[0], last=self.rows[-1],
                    first_observed_epoch_below_update_ratio=thresholds,
                    note='Observed epochs only; threshold crossings need not be permanent. Norm ratios do not identify an Adam-causal effect.')

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.stream.close()
