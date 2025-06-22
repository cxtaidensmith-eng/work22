import os
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import Optimizer
import torch.nn.functional as F
import math
from configparser import ConfigParser
from loguru import logger
from datetime import datetime
import pytz
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


def run_epoch(model, fold, criterion, optimizer, data, data_dict):
    X, Y = data['Feature'], data['Label']
    train_mask, test_mask = data['Mask'][fold]
    train_num = data['Train_Num'][fold]
    test_num = data['Test_Num'][fold]
    Class_num = data_dict['Class_Num']

    model.train()
    optimizer.zero_grad()
    output, _Label_embedding, _Auxi_classifier_output = model(X)
    loss = criterion(output, Y, train_mask, _Label_embedding, _Auxi_classifier_output)
    loss_train = loss.item()

    loss.backward()
    optimizer.step()

    pred = torch.argmax(output[train_mask], dim=1)
    correct = torch.sum(pred == Y[train_mask])
    acc_train = correct.item() / train_num

    model.eval()
    with torch.no_grad():
        output, _Label_embedding, _Auxi_classifier_output  = model(X)
        loss = criterion(output, Y, test_mask, _Label_embedding, _Auxi_classifier_output)
        loss_test = loss.item()
        pred = torch.argmax(output[test_mask], dim=1)
        correct = torch.sum(pred == Y[test_mask])
        acc_test = correct.item() / test_num
        auc_test = get_auc(y_true=F.one_hot(Y[test_mask].cpu(), num_classes=Class_num),
                               y_pred=output[test_mask].cpu())

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

        Feature_1 = model.GCN.GCN_feature_1.cpu().numpy()
        Feature_2 = model.GCN.GCN_feature_2.cpu().numpy()

    return acc_train, loss_train, acc_test, loss_test, auc_test, sensitivity_test, specificity_test, f1_test, Feature_1, Feature_2, torch.argmax(output, dim=1).cpu().numpy()


class CustomCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.01, last_epoch: int = -1, verbose: bool = False):
        self.T_max = T_max  # 总的 epoch 数
        self.eta_min = eta_min  # 最小学习率
        self.hold_epoch = 20  # 前 20 个 epoch 保持学习率不变
        self.initial_lr = optimizer.param_groups[0]['lr']  # 初始学习率
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

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

        shanghai_tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(shanghai_tz).strftime('%Y-%m-%d_%H-%M')
        # self.history_name = f'{self.DATA_SET}_{self.Task}_{current_time}'
        Config_name = os.path.basename(Config_name)
        self.history_name = f'{self.DATA_SET}_{self.Task}_{Config_name[:-4]}'

        self.logger = logger
        self.logger.remove()
        self.logger.add(os.path.join(Root_path, f"logs/{self.history_name}.log"), format="{time} | {message}")

        self.Save_History_Path = os.path.join(Root_path, f"Result/{self.history_name}.npy")


        self.result_dir = os.path.join(Root_path, 'Result')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.Graph_dir = os.path.join(Root_path, 'Graph', self.DATA_SET, self.Task, self.history_name)

        if not os.path.exists(self.Graph_dir) and self.SAVE_GAPH:
            os.makedirs(self.Graph_dir)

        self.Save_Graph_1_Path = os.path.join(self.Graph_dir, "graph_1.npy")
        self.Save_Graph_2_Path = os.path.join(self.Graph_dir, "graph_2.npy")

        self.Save_Y_Pred_Path = os.path.join(self.Graph_dir, "Y_Pred.npy")
        self.Save_Y_True_Path = os.path.join(self.Graph_dir, "Y_True.npy")
        self.Save_Y_Mask_Path = os.path.join(self.Graph_dir, "Y_Mask.npy")

