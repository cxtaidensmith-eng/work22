"""诊断 v5 modal gate 是否真在学：训练 fold 0，每若干 epoch 打印 gate 值。"""
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
import torch
import random
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch import optim
from Utils import load_path, load_dataset, split_hertergraph, run_epoch, SET_Random, CustomCosineAnnealingLR, Config_
from Model import HeterGraph_Model_Kmeans
from Loss import criterion_lossv2

CONFIG = Config_('.', './Config/T_ADNI3_v5.ini', 0)
CONFIG.epochs = 200

Feature_Data_path, Feature_dict_path, _, Class_names_list = load_path(CONFIG.Root_path, CONFIG.DATA_SET, CONFIG.Task)
DATASET_Dict, DATASET_DATA = load_dataset(Feature_Data_path, Feature_dict_path, CONFIG.Device, Class_names_list, CONFIG.Shuffle, CONFIG.seed, train_size=CONFIG.train_size)
split_hertergraph_list = split_hertergraph(DATASET_Dict, DATASET_DATA, CONFIG, K=CONFIG.Herter_k, remove_self_loop=CONFIG.remove_self_loop, remove_repeat=CONFIG.remove_repeat)

modal_names = DATASET_Dict['Modal_Name']
print('Modal names:', modal_names)

fold = 0
SET_Random(0)

model = HeterGraph_Model_Kmeans(
    DATASET_Dict, split_hertergraph_list[fold], CONFIG.Hidden_size, CONFIG.Drop_rate,
    CONFIG.ChebGCN_K,
    num_layers=CONFIG.num_layers, num_heads=CONFIG.num_heads,
    input_noise_std=CONFIG.input_noise_std, drop_path=CONFIG.drop_path,
).to(CONFIG.Device)
criterion = criterion_lossv2(DATASET_Dict, CONFIG.Device, rate=CONFIG.Loss_rate)
optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
scheduler = CustomCosineAnnealingLR(optimizer, T_max=CONFIG.T_max, eta_min=CONFIG.Lr_Min)

print(f'\n=== Initial gate (sigmoid of logit) ===')
gate = torch.sigmoid(model.modal_gate_logit).detach().cpu().numpy()
for n, g in zip(modal_names, gate):
    print(f'  {n:20s} = {g:.4f}')

acc_list = []
for step in range(CONFIG.epochs):
    res = run_epoch(model, fold, criterion, optimizer, DATASET_DATA, DATASET_Dict, grad_clip=CONFIG.grad_clip)
    acc_train = res[0]
    acc_test = res[2]
    acc_list.append(acc_test)
    if CONFIG.Use_scheduler:
        scheduler.step()
    if step % 20 == 19 or step == CONFIG.epochs - 1:
        gate = torch.sigmoid(model.modal_gate_logit).detach().cpu().numpy()
        gate_str = ' '.join([f'{n[:4]}={g:.3f}' for n, g in zip(modal_names, gate)])
        print(f'  ep {step+1:4d}  train={acc_train:.4f}  test={acc_test:.4f}  best={max(acc_list):.4f}  gate: {gate_str}')

print(f'\n=== Final gate ===')
gate = torch.sigmoid(model.modal_gate_logit).detach().cpu().numpy()
for n, g in zip(modal_names, gate):
    print(f'  {n:20s} = {g:.4f}')

print(f'\nFold 0 best test acc: {max(acc_list):.4f}')
