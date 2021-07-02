from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from resnet_triplet import *
from sklearn.mixture import GaussianMixture
from torch.nn.functional import kl_div, softmax, log_softmax
import dataloader_cifar_MPL as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--noisy_use_epoch', default=50, type=int)
parser.add_argument('--lr_student', default=0.001, type=int)
parser.add_argument('--lr_teacher', default=0.001, type=int)
parser.add_argument('--ratio_unsup', default=1, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training
def trainMPL(epoch,net_teacher,net_student,optimizer_teacher,optimizer_student,labeled_trainloader,unlabeled_trainloader):
    net_teacher.train()
    net_student.train()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, _, labels_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, _ = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)

        inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            pseudo_label = net_teacher(inputs_u2) # original implementation uses augmented version
            pseudo_label = torch.softmax(pseudo_label, dim=1)
            pseudo_label = pseudo_label.detach()
        
        output_s_on_u_old = net_student(inputs_u2)
        loss_s_on_u = loss_fn(output_s_on_u_old, pseudo_label)

        optimizer_student.zero_grad()
        loss_s_on_u.backward()

        ## second term of h
        grad_s_on_u = []
        for name,params in net_student.named_parameters():
            if 'encoder' in name or 'fc' in name:
                grad_s_on_u.append(params.grad.view(-1))
        grad_s_on_u = torch.cat(grad_s_on_u)

        optimizer_student.step()
        # now new student updated.

        output_s_on_l_new = net_student(inputs_x)
        loss_s_on_l_new = loss_fn(output_s_on_l_new, labels_x)

        optimizer_student.zero_grad()
        loss_s_on_l_new.backward()

        grad_s_on_l = []
        for name,params in net_student.named_parameters():
            if 'encoder' in name or 'fc' in name:
                grad_s_on_l.append(params.grad.view(-1))
        grad_s_on_l = torch.cat(grad_s_on_l)

        # teacher entropy for unlabeled data
        dot_product = grad_s_on_u * grad_s_on_l
        dot_product = dot_product.detach()

        output_t_on_u = net_teacher(inputs_u2)
        loss_t_on_u_aug = loss_fn(output_t_on_u, pseudo_label)

        optimizer_teacher.zero_grad()
        loss_t_on_u_aug.backward()

        grad_t_on_u = []
        for name,params in net_teacher.named_parameters():
            if 'encoder' in name or 'fc' in name:
                grad_t_on_u.append(params.grad.view(-1))
        grad_t_on_u = torch.cat(grad_t_on_u)

        # Total Meta Grad for teacher
        grad_t_on_u = grad_t_on_u * dot_product

        # teacher labeled data loss (CE)
        output_t_on_l = net_teacher(inputs_x)
        loss_t_on_l = loss_fn(output_t_on_l, labels_x)

        # teacher unlabeled data loss (KL)
        teacher_output = net_teacher(inputs_u)
        preds1 = softmax(teacher_output, dim=1).detach()
        preds2 = net_teacher(inputs_u2)
        preds2 = log_softmax(preds2, dim=1)
        
        loss_kldiv = kl_div(preds2, preds1, reduction='none')    # UDA loss
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        
        loss_t_uda = args.ratio_unsup * torch.mean(loss_kldiv)

        loss = loss_t_on_l + loss_t_uda
        optimizer_teacher.zero_grad()
        loss.backward()

        # add Meta grad
        for name,params in net_teacher.named_parameters():
            if 'encoder' in name or 'fc' in name:
                grad_size = params.grad.view(-1).size(0)
                grad_shape = params.grad.shape
                meta_grad = grad_t_on_u[:grad_size]
                meta_grad = meta_grad.reshape(grad_shape)
                params.grad += meta_grad

                grad_t_on_u = grad_t_on_u[grad_size:]

        optimizer_teacher.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t Teacher loss: %.2f  Student loss: %.2f'
                %(args.dataset, args.r, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item(), loss_s_on_u.item()))
        sys.stdout.flush()

def test(epoch,net1):
    net1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class CrossEntropyLoss_soft(object):
    def __call__(self, input, target):
        return -torch.mean(torch.sum(F.log_softmax(input, dim=1) * target, dim=1))

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    if args.model == 'resnet18':
        model = SwavResNet(name='resnet18', num_class=args.num_class).cuda()

    return model

def load_weights(model, path):
    pretrained_model = SwavResNet(name='resnet18', num_class=args.num_class)
    pretrained_model.load_state_dict(torch.load(path))
    model = SwavResNet(name='resnet18', num_class=args.num_class).cuda()

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def load_module_weights(model,path):
    pretrained_model = SwavResNet(name='resnet18', num_class=args.num_class)
    pretrained_dict = torch.load(path)['model']
    new_dict = dict()
    for k, v in pretrained_dict.items():
        k1 = k.replace('module.', '')
        new_dict[k1] = pretrained_dict[k]
    pretrained_model.load_state_dict(new_dict)
    model = SwavResNet(name='resnet18', num_class=args.num_class).cuda()

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def freeze_fc(net):
    for name, child in net.named_children():
        for name2, params in child.named_parameters():
            if name in ['encoder', 'head']:
                params.requires_grad = False

def unfreeze_fc(net):
    for name, child in net.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = True

stats_log=open('./checkpoint/%s_%.1f_%s_MPL_UDA_'%(args.dataset,args.r,args.model)+str(args.noisy_use_epoch)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s_MPL_UDA_'%(args.dataset,args.r,args.model)+str(args.noisy_use_epoch)+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log)

print('| Building net')
net_teacher = create_model()
net_student = create_model()
#net1 = load_module_weights(net1, path)
#net1.load_state_dict(torch.load(path))
#net1 = torch.nn.DataParallel(net1)
#cudnn.benchmark = True

optimizer_teacher = optim.SGD(filter(lambda p:p.requires_grad, net_teacher.parameters()), lr=args.lr_teacher, momentum=0.9, weight_decay=5e-4)
optimizer_student = optim.SGD(filter(lambda p:p.requires_grad, net_student.parameters()), lr=args.lr_student, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
loss_fn = CrossEntropyLoss_soft()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [] # save the history of losses from network

test_loader = loader.run('test')
labeled_trainloader, unlabeled_trainloader = loader.run('train')

for epoch in range(args.num_epochs+1):
    if epoch == 150:
        for param_group in optimizer_teacher.param_groups:
            param_group['lr'] = args.lr_teacher / 10
        for param_group in optimizer_student.param_groups:
            param_group['lr'] = args.lr_student / 10
        
    print('MPL Train Net1')
    trainMPL(epoch,net_teacher,net_student,optimizer_teacher,optimizer_student,labeled_trainloader, unlabeled_trainloader)

    test(epoch,net_teacher)
    test(epoch,net_student)
