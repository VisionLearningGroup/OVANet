from __future__ import print_function
import yaml
import easydict
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import get_model_mme, log_set
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders
from eval import test, feat_get
from models.basenet import ResClassifier_MME
import neptune
# Training settings

import argparse

parser = argparse.ArgumentParser(description='Pytorch DA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

parser.add_argument('--source_path', type=str, default='./utils/source_list.txt', metavar='B',
                    help='path to source list')
parser.add_argument('--target_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='path to target list')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp_name', type=str, default='office', help='/path/to/config/file')
parser.add_argument('--net', type=str, default='resnet50', help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--multi', type=float, default=0.1, metavar='N',
                    help='weight factor for adaptation')
parser.add_argument('--neptune', dest='neptune',
                    help='whether use neptune logging',
                    action='store_true')
args = parser.parse_args()

config_file = args.config
conf = yaml.load(open(config_file))
save_config = yaml.load(open(config_file))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

source_data = args.source_path
target_data = args.target_path
evaluation_data = args.target_path
network = args.net
source_loader, target_loader, \
test_loader, target_folder = get_dataloaders(source_data,
                                             target_data,
                                             evaluation_data,
                                             conf)
script_name = os.path.basename(__file__)
logname = log_set(source_data,
                  target_data,
                  network,
                  args,
                  script_name,
                  config_file,
                  gpu_devices)
use_gpu = torch.cuda.is_available()

n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total

open = n_total-n_share-n_source_private > 0

num_class = n_share + n_source_private

G, _, _ = get_model_mme(network, num_class=num_class,
                      temp=conf.model.temp)
dim = 2048
if args.net != 'resnet50':
    dim = 512
C2 = ResClassifier_MME(num_classes=2*num_class,
                       norm=False, input_size=dim)
C1 = ResClassifier_MME(num_classes=num_class,
                       norm=False,input_size=dim)

device = torch.device("cuda")
G.to(device)
C1.to(device)
C2.to(device)
ndata = target_folder.__len__()

params = []
for key, value in dict(G.named_parameters()).items():
    if 'bias' in key:
        params += [{'params': [value], 'lr': conf.train.multi,
                    'weight_decay': conf.train.weight_decay}]
    else:
        params += [{'params': [value], 'lr': conf.train.multi,
                    'weight_decay': conf.train.weight_decay}]

criterion = torch.nn.CrossEntropyLoss().cuda()
opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                  weight_decay=0.0005, nesterov=True)
opt_c1 = optim.SGD(list(C1.parameters())+list(C2.parameters()), lr=1.0,
                   momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                   nesterov=True)
[G, C1, C2], [opt_g, opt_c1] = amp.initialize([G, C1, C2],
                                          [opt_g, opt_c1],
                                          opt_level="O1")
G = nn.DataParallel(G)
C1 = nn.DataParallel(C1)
C2 = nn.DataParallel(C2)
param_lr_g = []
for param_group in opt_g.param_groups:
    param_lr_g.append(param_group["lr"])
param_lr_f = []
for param_group in opt_c1.param_groups:
    param_lr_f.append(param_group["lr"])


def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    weight = torch.ones(2).float().cuda()
    weight[0] = 1. / (num_class-1)
    criterion_open = nn.CrossEntropyLoss(weight=weight).cuda()
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_f, opt_c1, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        opt_g.zero_grad()
        opt_c1.zero_grad()
        C2.module.weight_norm()
        ## Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)
        out_open = C2(feat)
        out_open = out_open.view(out_s.size(0), 2, -1)
        label_s_sp = torch.zeros((out_open.size(0),
                                  out_open.size(2))).long().cuda()
        label_range = torch.range(0, out_open.size(0)-1).long()
        label_s_sp[label_range, label_s] = 1
        loss_s = criterion(out_s, label_s)
        loss_st = criterion_open(out_open, label_s_sp)
        feat_t = G(img_t)
        out_open_t = C2(feat_t)
        out_open_t = out_open_t.view(img_t.size(0), 2, -1)
        out_open_t = F.softmax(out_open_t, 1)
        ent_open = torch.mean(torch.mean(torch.sum(-out_open_t *
                                                   torch.log(out_open_t+1e-8),1),1))
        all = loss_s + loss_st + args.multi * ent_open
        with amp.scale_loss(all, [opt_g, opt_c1]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c1.step()
        opt_g.zero_grad()
        opt_c1.zero_grad()
        if step % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '
                  'Loss Open: {:.6f} Loss Open Target: {:.6f}\t'.format(
                step, conf.train.min_step,
                100 * float(step / conf.train.min_step),
                loss_s.item(), loss_st.item(), ent_open.item()))
        if step > 0 and step % conf.test.test_interval == 0:
            print("with open")
            acc_o, h_score = test(step, test_loader, logname, n_share, G,
                                  [C1, C2], open=open)
            acc_so, h_score_so = test(step, source_loader, logname, n_share, G,
                                      [C1, C2], open=False)
            print("acc source %s"%acc_so)
            print("with known")
            if args.neptune:
                neptune.log_metric('accuracy open', acc_o)
                neptune.log_metric('H Score', h_score)
            G.train()
            C1.train()


train()
