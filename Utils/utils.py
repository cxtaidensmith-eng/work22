import os
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import Optimizer
import torch.nn.functional as F
import math
from configparser import ConfigParser
from datetime import datetime
from sklearn.metrics import confusion_matrix, recall_score, f1_score


def SET_Random(seed):
    """Set the seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_auc(y_true, y_pred):
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()
    auc_scores = roc_auc_score(y_true, y_pred)

    return auc_scores


def load_path(Root_path, DATA_SET, Task):

    if DATA_SET == 'ABIDE':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_processed_data_modal.csv')
        Feature_dict_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_modal_feat_dict.npy')
        Save_History_path = [os.path.join(Root_path, f'RESULT/{DATA_SET}_History{Task}.npy'),
                             os.path.join(Root_path, f'logs/{DATA_SET}_History{Task}.json')]

        if Task == 'ADS_CN':
            class_names = ['ADS', 'CN']
        else:
            print('Task Error')


    elif DATA_SET == 'ABIDE_CC200':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_processed_data_modal.csv')
        Feature_dict_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_modal_feat_dict.npy')
        Save_History_path = [os.path.join(Root_path, f'RESULT/{DATA_SET}_History{Task}.npy'),
                             os.path.join(Root_path, f'logs/{DATA_SET}_History{Task}.json')]

        if Task == 'ADS_CN':
            class_names = ['ADS', 'CN']
        else:
            print('Task Error')


    elif DATA_SET == 'ABIDE_ho':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_processed_data_modal.csv')
        Feature_dict_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_modal_feat_dict.npy')
        Save_History_path = [os.path.join(Root_path, f'RESULT/{DATA_SET}_History{Task}.npy'),
                             os.path.join(Root_path, f'logs/{DATA_SET}_History{Task}.json')]

        if Task == 'ADS_CN':
            class_names = ['ADS', 'CN']
        else:
            print('Task Error')


    elif DATA_SET == 'ABIDE-5':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_processed_data_modal.csv')
        Feature_dict_path = os.path.join(Root_path, f'DATASET/{DATA_SET}/ABIDE_modal_feat_dict.npy')
        Save_History_path = [os.path.join(Root_path, f'RESULT/{DATA_SET}_History{Task}.npy'),
                             os.path.join(Root_path, f'logs/{DATA_SET}_History{Task}.json')]

        if Task == 'ADS_CN':
            class_names = ['ADS', 'CN']
        else:
            print('Task Error')


    elif DATA_SET == 'TADPOLE':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/TADPOLE/{}_ADNI_processed_standard_data.csv'.format(Task))
        Feature_dict_path = os.path.join(Root_path, 'DATASET/TADPOLE/{}_ADNI_modal_feat_dict.npy'.format(Task))
        Save_History_path = [os.path.join(Root_path, 'RESULT/ADNI_History{}.npy'.format(Task)),
                             os.path.join(Root_path, 'logs/ADNI_History{}.json'.format(Task))]

        if Task == 'SMCI_PMCI':
            class_names = ['SMCI', 'PMCI']
        elif Task == 'AD_CN_SMCI':
            class_names = ['AD', 'CN', 'SMCI']
        elif Task == 'AD_CN_SMCI_PMCI':
            class_names = ['AD', 'CN', 'SMCI', 'PMCI']
        else:
            print('Task Error')

    elif DATA_SET == 'ADNI':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/ADNI/Select_NEW_{}__ADNI_data.csv'.format(Task))
        Feature_dict_path = os.path.join(Root_path, 'DATASET/ADNI/Select_NEW_{}__ADNI_modal_feat_dict.npy'.format(Task))
        Save_History_path = [os.path.join(Root_path, 'RESULT/ADNI_NEW_History{}.npy'.format(Task)),
                             os.path.join(Root_path, 'logs/ADNI_NEW_History{}.json'.format(Task))]

        if Task == 'AD_CN':
            class_names = ['AD', 'CN']
        elif Task == 'SMC_EMCI_LMCI':
            class_names = ['SMC', 'EMCI', 'LMCI']
        elif Task == 'AD_CN_EMCI_LMCI':
            class_names = ['AD', 'CN', 'EMCI', 'LMCI']
        else:
            print('Task Error')

    return Feature_Data_path, Feature_dict_path, Save_History_path, class_names


def run_epoch(model, fold, criterion, optimizer, data, data_dict, eval_model=None,
              grad_clip=None, ema=None, mixup_alpha=0.0, gate_sparsity_lambda=0.0,
              label_graph_reg_lambda=0.0, logit_adjust_tau=0.0):
    X, Y = data['Feature'], data['Label']
    train_mask, test_mask = data['Mask'][fold]
    train_num = data['Train_Num'][fold]
    test_num = data['Test_Num'][fold]
    Class_num = data_dict['Class_Num']

    model.train()
    optimizer.zero_grad()

    if mixup_alpha > 0.0:
        # 只在 train 样本之间做 mixup，避免 test 信息通过 Adj_Learning / Global_Message 泄漏
        train_idx = torch.where(train_mask)[0]
        perm_train = train_idx[torch.randperm(train_idx.size(0), device=X.device)]
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))

        X_in = X.clone()
        X_in[train_idx] = lam * X[train_idx] + (1.0 - lam) * X[perm_train]

        Y_b = Y.clone()
        Y_b[train_idx] = Y[perm_train]

        output, _Label_embedding, _Auxi_classifier_output = model(X_in)
        loss_a = criterion(output, Y, train_mask, _Label_embedding, _Auxi_classifier_output)
        loss_b = criterion(output, Y_b, train_mask, _Label_embedding, _Auxi_classifier_output)
        loss = lam * loss_a + (1.0 - lam) * loss_b
    else:
        output, _Label_embedding, _Auxi_classifier_output = model(X)
        loss = criterion(output, Y, train_mask, _Label_embedding, _Auxi_classifier_output)

    if gate_sparsity_lambda > 0 and hasattr(model, 'gate_sparsity_loss'):
        loss = loss + gate_sparsity_lambda * model.gate_sparsity_loss()
    if label_graph_reg_lambda > 0 and hasattr(model, 'label_relation_loss'):
        loss = loss + label_graph_reg_lambda * model.label_relation_loss(train_mask)

    loss_train = loss.item()

    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    pred = torch.argmax(output[train_mask], dim=1)
    correct = torch.sum(pred == Y[train_mask])
    acc_train = correct.item() / train_num

    if ema is not None:
        ema.update(model)
        ema.swap_in(model)

    eval_net = eval_model if eval_model is not None else model
    eval_net.eval()
    try:
        with torch.no_grad():
            output, _Label_embedding, _Auxi_classifier_output = eval_net(X)
            loss = criterion(output, Y, test_mask, _Label_embedding, _Auxi_classifier_output)
            loss_test = loss.item()
            metric_output = output
            if logit_adjust_tau != 0.0:
                class_weight = data_dict['Label_Weight'].to(output).clamp_min(1e-8)
                metric_output = metric_output - float(logit_adjust_tau) * class_weight.log().view(1, -1)
            pred = torch.argmax(metric_output[test_mask], dim=1)
            correct = torch.sum(pred == Y[test_mask])
            acc_test = correct.item() / test_num
            auc_test = get_auc(y_true=F.one_hot(Y[test_mask].cpu(), num_classes=Class_num),
                                   y_pred=metric_output[test_mask].cpu())

            if Class_num == 2:
                Y_pred = pred.cpu().numpy()
                Y_true = Y[test_mask].cpu().numpy()
                tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()

                f1_test = f1_score(Y_true, Y_pred, average='weighted')
                sensitivity_test = tp / (tp + fn)
                specificity_test = tn / (tn + fp)
            else:
                Y_pred = pred.cpu().numpy()
                Y_true = Y[test_mask].cpu().numpy()
                f1_test = f1_score(Y_true, Y_pred, average='weighted')
                sensitivity_test = recall_score(Y_true, Y_pred, average='weighted')
                specificity_test = 0.0

            Feature_1 = eval_net.GCN.GCN_feature_1.cpu().numpy()
            Feature_2 = eval_net.GCN.GCN_feature_2.cpu().numpy()
            Y_full_pred = torch.argmax(metric_output, dim=1).cpu().numpy()
            test_logit = metric_output[test_mask].cpu().numpy()
            raw_test_logit = output[test_mask].cpu().numpy()
    finally:
        if ema is not None:
            ema.swap_out(model)

    return (acc_train, loss_train, acc_test, loss_test, auc_test,
            sensitivity_test, specificity_test, f1_test, Feature_1, Feature_2,
            Y_full_pred, test_logit, raw_test_logit)


class ModelEMA:
    """Exponential Moving Average of model weights (swap-style, deepcopy-free).
    维护参数字典的滑动平均；eval 时通过 .swap_in(model) 把 EMA 权重临时装入 student，
    eval 完调用 .swap_out(model) 还原 student 当前训练权重。
    适合包含非叶子 tensor 的模型（如 graph_learning.Ws_Parameter）。
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone()
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow:
            v = msd[k]
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    @torch.no_grad()
    def swap_in(self, model):
        """把 EMA 权重装入 model，原 student 权重保存到 self._backup。"""
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)

    @torch.no_grad()
    def swap_out(self, model):
        """恢复 student 训练权重。"""
        if self._backup is None:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = None


class CustomCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.01, last_epoch: int = -1):
        self.T_max = T_max  # 总的 epoch 数
        self.eta_min = eta_min  # 最小学习率
        self.hold_epoch = 20  # 前 20 个 epoch 保持学习率不变
        self.initial_lr = optimizer.param_groups[0]['lr']  # 初始学习率
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.hold_epoch:
            # 前 20 个 epoch 学习率保持为 0.1
            return [self.initial_lr for _ in self.optimizer.param_groups]
        else:
            # 后 80 个 epoch 按余弦下降
            T_cur = self.last_epoch - self.hold_epoch
            return [self.eta_min + (self.initial_lr - self.eta_min) *
                    (1 + math.cos(math.pi * T_cur / (self.T_max - self.hold_epoch))) / 2
                    for _ in self.optimizer.param_groups]


class Config_(object):
    def __init__(self, Root_path, Config_name, cuda_device):
        super().__init__()
        config = ConfigParser()
        # config.read(os.path.join(Root_path, f'Config/{Config_name}'), encoding='UTF-8')
        config.read(f'{Config_name}', encoding='UTF-8')
        self.Root_path = Root_path

        self.DATA_SET = config.get('DataSet', 'DATA_SET')
        self.Task = config.get('DataSet', 'Task')
        self.Shuffle = config.getboolean('DataSet', 'Shuffle')
        self.train_size = config.getfloat('DataSet', 'train_size')

        self.Cuda_id = config.getint('Hyper_Opt', 'Cuda_id')
        self.seed = config.getint('Hyper_Opt', 'seed')

        self.G_base  = config.get('GRAPH', 'G_base')
        self.G_distance = config.get('GRAPH', 'G_distance')
        self.G_use  = config.get('GRAPH', 'G_use')


        self.Drop_rate = config.getfloat('Modal', 'Drop_rate')
        self.ChebGCN_K = config.getint('Modal', 'ChebGCN_K')
        self.Herter_k = config.getint('Modal', 'Herter_k')
        self.Hidden_size = config.getint('Modal', 'Hidden_size')


        self.epochs = config.getint('Optim', 'epochs')
        self.lr = config.getfloat('Optim', 'lr')
        self.weight_decay = config.getfloat('Optim', 'weight_decay')
        self.Loss_rate = config.getfloat('Optim', 'Loss_rate')


        self.Use_scheduler = config.getboolean('Scheduler', 'Use_scheduler')
        self.T_max = config.getint('Scheduler', 'T_max')
        self.Lr_Min = config.getfloat('Scheduler', 'Lr_Min')

        self.use_ema = config.getboolean('Optim', 'use_ema', fallback=False)
        self.ema_decay = config.getfloat('Optim', 'ema_decay', fallback=0.999)
        self.grad_clip = config.getfloat('Optim', 'grad_clip', fallback=0.0)
        self.n_seeds = config.getint('Optim', 'n_seeds', fallback=1)
        self.mixup_alpha = config.getfloat('Optim', 'mixup_alpha', fallback=0.0)
        self.num_layers = config.getint('Modal', 'num_layers', fallback=2)
        self.num_heads = config.getint('Modal', 'num_heads', fallback=4)
        self.input_noise_std = config.getfloat('Modal', 'input_noise_std', fallback=0.05)
        self.drop_path = config.getfloat('Modal', 'drop_path', fallback=0.05)
        self.Graph_head = config.get('Modal', 'Graph_head', fallback='cheb')
        self.graph_layers = config.getint('Modal', 'graph_layers', fallback=1)
        self.graph_heads = config.getint('Modal', 'graph_heads', fallback=2)
        self.graph_beta = config.getfloat('Modal', 'graph_beta', fallback=0.5)
        self.graph_k_order = config.getint('Modal', 'graph_k_order', fallback=3)
        self.graph_alpha = config.getfloat('Modal', 'graph_alpha', fallback=0.5)
        self.graph_kernel = config.get('Modal', 'graph_kernel', fallback='simple')
        self.graph_use_graph = config.getboolean('Modal', 'graph_use_graph', fallback=True)
        self.graph_dropout = config.getfloat('Modal', 'graph_dropout', fallback=self.Drop_rate)
        self.graph_hidden = config.getint('Modal', 'graph_hidden', fallback=max(1, self.Hidden_size // 2))
        self.global_word_emb = config.getint('Modal', 'global_word_emb', fallback=self.Hidden_size)
        self.semantic_branch = config.get('Modal', 'semantic_branch', fallback='both')
        self.semantic_fusion = config.get('Modal', 'semantic_fusion', fallback='add')
        self.adj_mode = config.get('Modal', 'adj_mode', fallback='learned')
        self.label_graph_alpha = config.getfloat('Modal', 'label_graph_alpha', fallback=0.0)
        self.label_graph_topk = config.getint('Modal', 'label_graph_topk', fallback=0)
        self.label_graph_reg_lambda = config.getfloat('Optim', 'label_graph_reg_lambda', fallback=0.0)
        self.gate_sparsity_lambda = config.getfloat('Optim', 'gate_sparsity_lambda', fallback=0.0)
        self.logit_adjust_tau = config.getfloat('Optim', 'logit_adjust_tau', fallback=0.0)
        self.save_raw_logits = config.getboolean('SAVE', 'save_raw_logits', fallback=False)

        self.SAVE_GAPH = config.getboolean('SAVE', 'SAVE_GAPH')
        


        self.remove_repeat = config.getboolean('Herter_Graph', 'remove_repeat')
        self.remove_self_loop = config.getboolean('Herter_Graph', 'remove_self_loop')

        if self.Cuda_id == -1:
            self.Device = torch.device('cpu')
        else:
            if cuda_device is not None:
                self.Device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
            else:
                self.Device = torch.device(f'cuda:{self.Cuda_id}' if torch.cuda.is_available() else 'cpu')

        Config_name = os.path.basename(Config_name)
        self.config_name = Config_name
        self.config_stem = os.path.splitext(Config_name)[0]

        self.result_dir = os.path.join(Root_path, 'Result', self.DATA_SET, self.Task)
        os.makedirs(self.result_dir, exist_ok=True)

        time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_stem = f'{time_str}_{self.config_stem}'
        collision_index = 1
        while os.path.exists(os.path.join(self.result_dir, f'{run_stem}.json')):
            collision_index += 1
            run_stem = f'{time_str}_{self.config_stem}_{collision_index}'
        self.history_name = run_stem

        self.Save_History_Path = os.path.join(self.result_dir, f'{self.history_name}.json')
        self.Epoch_CSV_Path = os.path.join(self.result_dir, f'{self.history_name}_epochs.csv')
        self.Split_CSV_Path = os.path.join(self.result_dir, f'{self.history_name}_splits.csv')
        self.Raw_Logit_Path = os.path.join(self.result_dir, f'{self.history_name}_logits.npz')

        self.Graph_dir = os.path.join(Root_path, 'Graph', self.DATA_SET, self.Task, self.history_name)

        if not os.path.exists(self.Graph_dir) and self.SAVE_GAPH:
            os.makedirs(self.Graph_dir)

        self.Save_Graph_1_Path = os.path.join(self.Graph_dir, "graph_1.npy")
        self.Save_Graph_2_Path = os.path.join(self.Graph_dir, "graph_2.npy")

        self.Save_Y_Pred_Path = os.path.join(self.Graph_dir, "Y_Pred.npy")
        self.Save_Y_True_Path = os.path.join(self.Graph_dir, "Y_True.npy")
        self.Save_Y_Mask_Path = os.path.join(self.Graph_dir, "Y_Mask.npy")

