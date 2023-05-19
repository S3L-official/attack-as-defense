'''
This is the test code of poisoned training under PhysicalBA.
Using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST and torchvision.datasets.CIFAR10.
Default physical transformations is Compose([RandomHorizontalFlip(),ColorJitter(), RandomAffine()])
Choose other transformations from torchvsion.transforms if you need
'''
import sys
sys.path.append('./BackdoorBox/')
sys.path.append('./scripts/')
print(sys.path)

import os
import numpy as np

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ColorJitter, RandomAffine
import torchvision.transforms as transforms
import core
import argparse
from tqdm import tqdm

from sklearn import metrics

# s3l: attack as defense import
import foolbox
import attacks_method.saliency as saliency
from attacks_method.iterative_projected_gradient import BIM, L2BasicIterativeAttack

import gl_var
from aasd_utils import get_auc
from aasd_utils import data_preprocessing
from aasd_utils import pytorch_bim_attack

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
poisoned_train_idx = None

parser = argparse.ArgumentParser(description='PyTorch ISSBA Backdoor Attack Training')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['gtsrb', 'mnist', 'cifar10'], help='dataset name')
parser.add_argument('--function', default='acc', type=str, help='function name')
parser.add_argument('--backdoor', default='TUAP', type=str, help='backdoor attack method')
args = parser.parse_args()


if args.dataset == 'gtsrb':
    # ============== GTSRB ==============
    dataset = torchvision.datasets.DatasetFolder

    transform_train = Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        ToTensor()
    ])
    trainset = dataset(
        root='../Datasets/GTSRB/Train', # change to your own path
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    transform_test = Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        ToTensor()
    ])
    testset = dataset(
        root='../Datasets/GTSRB/Test', # change to your own path
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)

    if bd_method == 'BadNets':
        pattern = torch.zeros((32, 32), dtype=torch.uint8)
        pattern[-3:, -3:] = 255
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[-3:, -3:] = 1.0

        badnets = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18, 43),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            poisoned_transform_train_index=2,
            poisoned_transform_test_index=2,
            seed=global_seed,
            deterministic=deterministic
        )
        poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
        model_file_name = 'change to your own path'
    else:
        raise NotImplementedError

elif args.dataset == 'mnist':

    # ============== mnist ==============
    dataset = torchvision.datasets.MNIST
    transform_train = Compose([
        ToTensor(),
    ])
    trainset = dataset('../Datasets', train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset('../Datasets', train=False, transform=transform_test, download=True)

    if bd_method == 'BadNets':
        pattern = torch.zeros((28, 28), dtype=torch.uint8)
        pattern[-3:, -3:] = 255
        weight = torch.zeros((28, 28), dtype=torch.float32)
        weight[-3:, -3:] = 1.0

        badnets = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.BaselineMNISTNetwork(),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            seed=global_seed,
            deterministic=deterministic
        )
        poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
        model_file_name = 'change to your own path'
    else:
        raise NotImplementedError

elif args.dataset == 'cifar10':

    # ============== cifar10 ==============

    dataset = torchvision.datasets.CIFAR10

    transform_train = Compose([
        ToTensor(),
    ])
    trainset = dataset('../Datasets', train=True, transform=transform_train, download=False)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset('../Datasets', train=False, transform=transform_test, download=False)

    if args.backdoor == 'PhysicalBA':
        PhysicalBA = core.PhysicalBA(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            seed=global_seed,
            deterministic=deterministic,
            # modify other transformations from torchvsion.transforms if you want 
            physical_transformations = Compose([
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.2,contrast=0.2), 
                RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))
            ])
        )

        poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()
        model_file_name = 'change to your own path'

    elif bd_method == 'TUAP':
        CUDA_VISIBLE_DEVICES = '0'
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
            'GPU_num': 1,

            'benign_training': False, # Train Attacked Model
            'batch_size': 128,
            'num_workers': 8,

            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180],

            'epochs': 200,

            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 10,

            'save_dir': 'experiments',
            'experiment_name': 'ResNet-18_CIFAR-10_TUAP'
        }

        UAP_benign_model = core.models.ResNet(18)
        UAP_benign_PATH = 'change to your own path'
        checkpoint = torch.load(UAP_benign_PATH)
        UAP_benign_model.load_state_dict(checkpoint)
        poisoned_rate = 0.25
        # epsilon = 10
        epsilon = 0.031
        delta = 0.2
        max_iter_uni = np.inf
        p_norm = np.inf
        num_classes = 10
        overshoot = 0.02
        max_iter_df = 50
        p_samples = 0.01
        mask = np.ones((3, 32, 32))

        tuap = core.TUAP(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            loss=nn.CrossEntropyLoss(),

            benign_model=UAP_benign_model,
            y_target=2,
            poisoned_rate=poisoned_rate,
            epsilon = epsilon,
            delta=delta,
            max_iter_uni=max_iter_uni,
            p_norm=p_norm,
            num_classes=num_classes,
            overshoot=overshoot,
            max_iter_df=max_iter_df,
            p_samples=p_samples,
            mask=mask,

            poisoned_transform_train_index=0,
            poisoned_transform_test_index=0,
            poisoned_target_transform_index=0,
            schedule=schedule,
            seed=global_seed,
            deterministic=True
        )

        model_file_name = 'change to your own path'
        poisoned_train_dataset, poisoned_test_dataset = tuap.get_poisoned_dataset()

    elif bd_method == 'BadNets':
        pattern = torch.zeros((32, 32), dtype=torch.uint8)
        pattern[-3:, -3:] = 255
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[-3:, -3:] = 1.0

        badnets = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.BaselineMNISTNetwork(),
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            seed=global_seed,
            deterministic=deterministic
        )
        poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
        model_file_name = 'change to your own path'
    else:
        raise NotImplementedError

else:
    raise ValueError('No such dataset!')


# load backdoor model
if args.dataset == 'gtsrb':
    model = core.models.ResNet(18, 43)
elif args.dataset == 'cifar10':
    model = core.models.ResNet(18)
elif args.dataset == 'mnist':
    model = core.models.BaselineMNISTNetwork()
else:
    raise ValueError('No such dataset!')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_file_name))
model.eval()
model.to(device)

# check model summary
print(model)

# select the function to run
if args.function == 'acc':
    ## check clean and poisoned model accuracy
    correct = 0
    total_clean = 0
    bd_correct = 0
    bd_total = 0
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    bd_test_loader = DataLoader(poisoned_test_dataset, batch_size=128, shuffle=False, num_workers=4)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_clean += target.size(0)
        
        for data, target in tqdm(bd_test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            bd_correct += pred.eq(target.view_as(pred)).sum().item()
            bd_total += target.size(0)

    print('clean accuracy: {}/{} ({:.0f}%)'.format(correct, total_clean, 100. * correct / total_clean))
    print('bd accuracy: {}/{} ({:.0f}%)'.format(bd_correct, bd_total, 100. * bd_correct / bd_total))

elif args.function == 'a2d':
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False, # True
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )

    bd_test_loader = DataLoader(
        poisoned_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )

    # define the hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # activation[name] = output.detach()
            activation[name] = input[0].detach()
        return hook

    # register the hook
    if args.dataset == 'mnist':
        model.fc1.register_forward_hook(get_activation('linear'))
        # model.fc2.register_forward_hook(get_activation('fc2'))
    else:
        model.linear.register_forward_hook(get_activation('linear'))

    # only need first N correct data
    clean_a2d_metrics = []
    poisoned_a2d_metrics = []

    counter = 0
    print('Start analyze process...')
    for data, target in tqdm(test_loader):
        data = data.to(device)
        target = target.to(device)
        
        pred_logits = model(data)
        max_logits = torch.max(pred_logits)
        pred_label = torch.argmax(pred_logits)
        pred_prob = torch.exp(pred_logits) / torch.sum(torch.exp(pred_logits))
        max_prob = torch.max(pred_prob)

        # check the accuracy of the clean data set
        if pred_label == target:
            counter += 1
            
            # use hook to get the middle output
            feature_value = activation['linear'].cpu().numpy()
            # flatten the feature value
            feature_value = feature_value.flatten()
            # get the argmax index of the feature value
            top_index = np.argmax(feature_value)
            top_feature_value = feature_value[top_index]

            # bim attack
            atk_image = pytorch_bim_attack(model, data, target, device)
            pred_logits = model(atk_image)

            #* use hook to get the middle output
            feature_value_scale = activation['linear'].cpu().numpy()
            feature_value_scale = feature_value_scale.flatten()
            feature_value_scale = feature_value_scale[top_index]
            feature_value_scale = np.mean(feature_value_scale)
            clean_a2d_metrics.append(feature_value_scale)

        if counter == 1000:
            break
    
    counter = 0
    print('Start poisoned samples analyze process...')
    for data, target in tqdm(bd_test_loader):
        # pytorch version
        data = data.to(device)
        target = target.to(device)

        pred_logits = model(data)
        max_logits = torch.max(pred_logits)
        pred_label = torch.argmax(pred_logits)
        pred_prob = torch.exp(pred_logits) / torch.sum(torch.exp(pred_logits))
        max_prob = torch.max(pred_prob)
        
        if pred_label == target:
            counter += 1
            
            # use hook to get the middle output
            feature_value = activation['linear'].cpu().numpy()
            # flatten the feature value
            feature_value = feature_value.flatten()
            top_index = np.argmax(feature_value)
            
            # bim attack
            atk_image = pytorch_bim_attack(model, data, target, device)
            pred_logits = model(atk_image)
            
            #* use hook to get the middle output
            feature_value_scale = activation['linear'].cpu().numpy()
            feature_value_scale = feature_value_scale.flatten()

            feature_value_scale = feature_value_scale[top_index]
            feature_value_scale = np.mean(feature_value_scale)
            poisoned_a2d_metrics.append(feature_value_scale)

        else:
            continue

        if counter == 1000:
            break
    
    # get auc
    poisoned_scale_up_test = np.array(poisoned_a2d_metrics)
    clean_scale_up_test = np.array(clean_a2d_metrics)
    print(poisoned_scale_up_test.shape, clean_scale_up_test.shape)
    auroc = get_auc(poisoned_scale_up_test, clean_scale_up_test)
    print('AUROC of test is:', auroc)


else:
    raise NotImplementedError