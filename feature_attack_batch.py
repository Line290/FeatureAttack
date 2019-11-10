from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import datetime
import ot
import random

from tqdm import tqdm
from models import *


parser = argparse.ArgumentParser(
    description='Feature Scattering Adversarial Training')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--store_adv_path', default='x_adv.pt', type=str,
                    help='adv. images store path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass')
parser.add_argument('--log_step', default=1, type=int, help='log_step')

# dataset dependent
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--batch_size_test',
                    default=100,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')

args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'cifar100':
    print('----------cifar100---------')
    args.num_classes = 100
    args.image_size = 32
if args.dataset == 'svhn':
    print('------------svhn10---------')
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'mnist':
    print('----------mnist---------')
    args.num_classes = 10
    args.image_size = 28

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print('==> Preparing data..')

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])
elif args.dataset == 'svhn':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1 1]
    ])

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

elif args.dataset == 'svhn':
    testset = torchvision.datasets.SVHN(root='./data',
                                        split='test',
                                        download=True,
                                        transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=10000,
                                         shuffle=False,
                                         num_workers=20)

print('==> Building model..')
if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
    print('---wide resenet-----')
    basic_net = WideResNet(depth=28,
                           num_classes=args.num_classes,
                           widen_factor=10)

net = basic_net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume and args.init_model_pass != '-1':
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                          ('checkpoint-%s' % args.init_model_pass))
    if not os.path.isdir(args.model_dir):
        print('train from scratch: no checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(
            f_path_latest):
        checkpoint = torch.load(f_path_latest)
        from collections import OrderedDict
        #
        new_ckpt = OrderedDict()
        for k, v in checkpoint['net'].items():
            # 'module.basic_net.fc.bias'
            new_k = k[:len('module.')] + k[len('module.basic_net.'):]
            new_ckpt[new_k] = v
        try:
            net.load_state_dict(new_ckpt)
        except:
            net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('resuming from epoch %s in latest' % start_epoch)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('resuming from epoch %s' % start_epoch)
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print('train from scratch: no checkpoint directory or file found')

criterion = nn.CrossEntropyLoss()

config_feature_attack = {
    'train': False,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 50,
    'step_size': 1.0 / 255 * 2,
    'random_start': True,
    'early_stop': True,
    'num_total_target_images': 500
}


def attack(model, inputs, target_inputs, y, config):
    step_size = config['step_size']
    epsilon = config['epsilon']
    num_steps = config['num_steps']
    random_start = config['random_start']
    early_stop = config['early_stop']
    model.eval()

    x = inputs.detach()
    if random_start:
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)

    target_logits, target_feat = model(target_inputs)
    target_feat = target_feat.detach()

    for i in range(num_steps):
        x.requires_grad_()
        zero_gradients(x)
        if x.grad is not None:
            x.grad.data.fill_(0)
        logits_pred, feat = model(x)
        preds = logits_pred.argmax(1)
        if early_stop:
            num_not_corr = (preds != y).sum().item()
            if num_not_corr > 0:
                break
        inver_loss = ot.pair_cos_dist(feat, target_feat)
        adv_loss = inver_loss.mean()
        adv_loss.backward()
        x_adv = x.data - step_size * torch.sign(x.grad.data)
        x_adv = torch.min(torch.max(x_adv, inputs - epsilon), inputs + epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)
        x = Variable(x_adv)
    return x.detach(), preds


target_images_size = args.batch_size_test
print('target batch size is: ', target_images_size)
num_total_target_images = config_feature_attack['num_total_target_images']
x_adv = []  # adv accumulator

net.eval()

untarget_success_count = 0
target_success_count = 0
total = 0
# load all test data
iterator = tqdm(testloader, ncols=0, leave=False)
all_test_data, all_test_label = None, None
for i, (test_data, test_label) in enumerate(iterator):
    all_test_data, all_test_label = test_data, test_label
print(all_test_data.size())

num_eval_imgs = all_test_data.size(0)
for clean_idx in tqdm(range(num_eval_imgs)):
    input, label_cpu = all_test_data[clean_idx].unsqueeze(0), all_test_label[clean_idx].unsqueeze(0)
    # print(inputs.size(), labels_cpu.size())
    start_time = time.time()
    batch_idx_list = {}
    other_label_test_idx = (all_test_label != label_cpu[0])
    other_label_test_data = all_test_data[other_label_test_idx]
    other_label_test_label = all_test_label[other_label_test_idx]
    num_other_label_img = other_label_test_data.size(0)

    # Setting candidate targeted images
    candidate_indices = torch.zeros(num_total_target_images).long().random_(0, num_other_label_img)
    num_batches = int(math.ceil(num_total_target_images / target_images_size))
    # print(other_label_test_idx.size(), other_label_test_data.size(), other_label_test_label.size())

    # Init index of image which be attacked successfully
    adv_idx = 0
    for i in range(num_batches):
        bstart = i * target_images_size
        bend = min(bstart + target_images_size, num_total_target_images)

        target_inputs = other_label_test_data[candidate_indices[bstart:bend]]
        target_labels_cpu = other_label_test_label[candidate_indices[bstart:bend]]
        target_inputs, target_labels = target_inputs.to(device), target_labels_cpu.to(device)

        input, label = input.to(device), label_cpu.to(device)
        inputs = input.repeat(target_images_size, 1, 1, 1)
        labels = label.repeat(target_images_size)


        # print(inputs.size(), labels)
        # print(target_inputs.size(), target_labels)
        x_batch_adv, predicted = attack(net, inputs, target_inputs, labels, config_feature_attack)

        # print(predicted.size())
        not_correct_idices = (predicted != labels).nonzero().view(-1)
        not_corrent_num = not_correct_idices.size(0)
        attack_success_num = predicted.eq(target_labels).sum().item()

        # At least one misclassified
        if not_corrent_num != 0:
            untarget_success_count += 1
            if attack_success_num != 0:
                target_success_count += 1
            adv_idx = not_correct_idices[0]
            break

    total += 1
    duration = time.time() - start_time
    x_adv.append(x_batch_adv[adv_idx].unsqueeze(0).cpu())

    if clean_idx % args.log_step == 0:
        print(
            "step %d, duration %.2f, aver untargeted attack success %.2f, aver targeted attack success %.2f"
            % (clean_idx, duration, 100. * untarget_success_count / total, 100.*target_success_count / total))
        sys.stdout.flush()
acc = 100. * untarget_success_count / total
print('Val acc:', acc)
print('Storing examples')
path = args.store_adv_path
x_adv = torch.cat(x_adv, dim=0)
torch.save({'x_adv': x_adv}, path)
