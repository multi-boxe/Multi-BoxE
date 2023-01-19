from config import MyTrainer, MyTester
from model import *
from process import MyTrainDataLoader, MyTestDataLoader
import os
import numpy as np
import faulthandler

faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dim = 100
train_times = 500
gamma = 12
alpha = 0.0001  # learning rate

nbatches = None
batch_size = 1024
score_ord = 1  # 得分的范数
bern_flag = 0  # negative sampling mode: 0->unif; 1->bern
neg_ent = 100
neg_rel = 0


sampling_mode = "normal"  # normal: 不固定替换头或尾
loss = "BoxLoss()"
opt_method = "adam"
filter_flag = 1  # 生成的负样本如果已经出现在train中则剔除，默认也是如此
uniform_init_args = [-0.5 / np.sqrt(dim), 0.5 / np.sqrt(dim)]
norm_flag = True  # 是否应用Tanh函数
bound_norm = True
w = 1. / neg_ent

tmp = 'semantic'

kn = 5
if bern_flag:
    b = 'b'
else:
    b = 'u'

dataset = f'./data/FB15K237/{tmp}/k{kn}/'
ckpt = f'./ckpt/multi-fb237-{tmp[:3]}-k{kn}-{dim}-gam{gamma}-ord{score_ord}-lr1e4-ep1k.ckpt'
res_path = f'./res/'

test_file = "test2id.txt"
config_file = res_path + f"config-multi-fb237-{tmp[:3]}-k{kn}-{dim}-gam{gamma}-ord{score_ord}-lr1e4-ep1k.txt"

if os.path.exists(config_file):
    os.remove(config_file)

with open(config_file, "w", encoding="utf-8") as f:
    line = f'dataset={dataset}\nckpt={ckpt}\n' + \
           f'dim = {dim}\nnbatches = {nbatches}\nbatch_size = {batch_size}\ntrain_times = {train_times}\n' + \
           f'score ord = {score_ord}\nloss = {loss}\nopt_method = {opt_method}\n' + \
           f'learning rate = {alpha}\nneg_ent = {neg_ent}\nbern_flag = {bern_flag}\n' + \
           f'norm_flag = {norm_flag}\nbound_norm = {bound_norm}\ngamma = {gamma}\n' + \
           f'w = {w}\nuniform_init_args = {uniform_init_args}\n'
    f.write(line)

# dataloader for training
train_dataloader = MyTrainDataLoader(
    in_path=dataset,
    nbatches=nbatches,
    batch_size=batch_size,
    threads=8,
    sampling_mode=sampling_mode,
    bern_flag=bern_flag,
    filter_flag=filter_flag,
    neg_ent=neg_ent,
    neg_rel=neg_rel)

box_tot = np.loadtxt(f'{dataset}box_tot.txt', dtype=int)
start_idx = np.loadtxt(f'{dataset}start_idx.txt', dtype=int)
box_num = np.loadtxt(f'{dataset}box_num.txt', dtype=int)
boxe = MultiBoxE(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    box_tot=box_tot,
    box_num=box_num,
    start_idx=start_idx,
    embedding_dim=dim,
    weight_init_args=uniform_init_args,
    norm_flag=norm_flag,
    use_gpu=1,
    score_ord=score_ord
)

# define the loss function
model = NegativeSampling(
    model=boxe,
    loss=BoxELoss(gamma, w),
    batch_size=train_dataloader.get_batch_size()
)

# train the model
trainer = MyTrainer(model=model, data_loader=train_dataloader, train_times=train_times, alpha=alpha, use_gpu=True,
                    opt_method=opt_method)
trainer.run()
boxe.save_checkpoint(ckpt)

# test the model
# print("!-------start to test-------!")
test_dataloader = MyTestDataLoader(dataset, test_file, "link", type_constrain=False)

boxe.load_checkpoint(ckpt)
tester = MyTester(model=boxe, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction5(type_constrain=False)