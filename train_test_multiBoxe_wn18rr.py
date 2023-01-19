from config import MyTrainer, MyTester
from model import *
from process import MyTrainDataLoader, MyTestDataLoader
import os
import numpy as np
import faulthandler

faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dim = 100
train_times = 500
gamma = 5
alpha = 0.001  # learning rate

nbatches = None
batch_size = 512
bern_flag = 0  # negative sampling mode: 0->unif; 1->bern
neg_ent = 150
neg_rel = 0

score_ord = 2  # 得分的范数
distance_func = 2   # 距离函数 0-boxe 1-query2box 3-newEmb
dis_arg = None # 当distance_func=1时，需要一个额外参数
if distance_func == 1:
    dis_arg = 0.2


sampling_mode = "normal"  # normal: 不固定替换头或尾
loss = "BoxLoss()"
opt_method = "adam"
filter_flag = 1  # 生成的负样本如果已经出现在train中则剔除，默认也是如此
uniform_init_args = [-0.5 / np.sqrt(dim), 0.5 / np.sqrt(dim)]
norm_flag = True  # 是否应用Tanh函数
bound_norm = True
w = 1. / neg_ent

tmp = 'semantic'
dd = 'wn18rr2'
kn = 1000
if bern_flag:
    b = 'b'
else:
    b = 'u'

dataset = f"/home/zhouyq/dataset/{dd}/{tmp}/k{kn}/"
ckpt = f"./ckpt/{dd}/multi-{tmp[:3]}-k{kn}-{dim}-ep{train_times}-dis{distance_func}.ckpt"
if dis_arg != None:
    ckpt = f"./ckpt/{dd}/multi-{tmp[:3]}-k{kn}-{dim}-ep{train_times}-dis{distance_func}-arg{str(dis_arg).replace('.','')}.ckpt"    
res_path = f'./res/{dd}/'

test_file = "test2id.txt"
config_file = res_path + f"config-multi-{tmp[:3]}-k{kn}-{dim}-ep{train_times}-dis{distance_func}.txt"
if dis_arg != None:
    config_file = res_path + f"config-multi-{tmp[:3]}-k{kn}-{dim}-ep{train_times}-dis{distance_func}-arg{str(dis_arg).replace('.','')}.txt"

if os.path.exists(config_file):
    os.remove(config_file)

# 日志写入
with open(config_file, "w", encoding="utf-8") as f:
    line = f'dataset={dataset}\nckpt={ckpt}\n' + \
           f'dim = {dim}\nnbatches = {nbatches}\nbatch_size = {batch_size}\ntrain_times = {train_times}\n' + \
           f'score ord = {score_ord}\nloss = {loss}\nopt_method = {opt_method}\n' + \
           f'learning rate = {alpha}\nneg_ent = {neg_ent}\nbern_flag = {bern_flag}\n' + \
           f'norm_flag = {norm_flag}\nbound_norm = {bound_norm}\ngamma = {gamma}\n' + \
           f'w = {w}\nuniform_init_args = {uniform_init_args}\ndistance_func={distance_func}\ndis_arg={dis_arg}\n'
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
    score_ord=score_ord,
    distance=distance_func,
    dis_arg=dis_arg
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
print("!-------start to train-------!")
trainer.run()
boxe.save_checkpoint(ckpt)

# test the model
print("!-------start to test-------!")
test_dataloader = MyTestDataLoader(dataset, test_file, "link", type_constrain=False)

boxe.load_checkpoint(ckpt)
tester = MyTester(model=boxe, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction5(type_constrain=False)