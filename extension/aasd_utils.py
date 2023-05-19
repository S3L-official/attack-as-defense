'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is ISSBA.
'''
import sys
sys.path.append('/storage/data/zhaozhe1/backup/root/root/ISSTA-21-open-source/BackdoorBox')
sys.path.append('/storage/data/zhaozhe1/backup/root/root/ISSTA-21-open-source/attacks_method')
sys.path.append('/storage/data/zhaozhe1/backup/root/root/ISSTA-21-open-source/scripts')
sys.path.append('/storage/data/zhaozhe1/backup/root/root/ISSTA-21-open-source/backdoor')

import faulthandler
faulthandler.enable()

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import argparse

import numpy as np
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from torchvision import transforms
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
from tqdm import tqdm

import foolbox
import saliency as saliency
from iterative_projected_gradient import BIM, L2BasicIterativeAttack
import gl_var

import core
from core.attacks.ISSBA import ISSBA

import keras
import utils_backdoor
import collections
from itertools import repeat
from sklearn import metrics

print(sys.path)

parser = argparse.ArgumentParser(description='PyTorch ISSBA Backdoor Attack Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--train', default=False, type=bool, help='train or test')
parser.add_argument('--function', default='acc', type=str, help='function name')

global_seed = 666
deterministic = False
torch.manual_seed(global_seed)
# secret_size = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(global_seed)
np.random.seed(global_seed)

def get_auc(neg_score, pos_score):
    """get the auroc resule

    Args:
        neg_score [np array]: benign samples attack costs
        pos_score [np array]: adversarial examples attack costs

    Returns:
        auc [float]: the auroc result
    """    
    pos_label = np.ones(pos_score.size, dtype=int)
    neg_label = np.zeros(neg_score.size, dtype=int)

    y_score = np.concatenate((pos_score, neg_score))
    y_label = np.concatenate((pos_label, neg_label))

    fpr, tpr, _ = metrics.roc_curve(y_label, y_score)
    auc = 1 - metrics.auc(fpr, tpr)

    return auc


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``

    Args:
        n (int): Number of repetitions x.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse

_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/6

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_,
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class StegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=32, width=32, in_channel=3):
        super(StegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up9 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(Conv2dSame(in_channels=64+in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=in_channel, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))
        inputs = torch.cat([secret, image], axis=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv4,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv6))
        merge7 = torch.cat([conv3,up7], axis=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv7))
        merge8 = torch.cat([conv2,up8], axis=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv8))
        merge9 = torch.cat([conv1,up9,inputs], axis=1)

        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)

        return residual


class StegaStampDecoder(nn.Module):
    """The image steganography decoder to assist the training of the image steganography encoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size, height, width, in_channel):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2)*(width//2//2//2), out_features=128), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([128, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=128, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2//2//2)*(width//2//2//2//2//2), out_features=512), nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)

        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)
        return secret


class Discriminator(nn.Module):
    """The image steganography discriminator to assist the training of the image steganography encoder and decoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        in_channel (int): Channel of the input image.
    """
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=8, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=16, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=1, kernel_size=3), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output


class MNISTStegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger (Customized for MNIST dataset).

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=28, width=28, in_channel=1):
        super(MNISTStegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=66, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=1, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))

        inputs = torch.cat([secret, image], axis=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        up5 = self.up5(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv4))
        merge5 = torch.cat([conv3,up5], axis=1)
        conv5 = self.conv5(merge5)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv2,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv1,up7,inputs], axis=1)
        conv7 = self.conv7(merge7)

        residual = self.residual(conv7)

        return residual


class MNISTStegaStampDecoder(nn.Module):
    """The image steganography decoder to assist the training of the image steganography encoder (Customized for MNIST dataset).

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size, height, width, in_channel):
        super(MNISTStegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=64*(height//2//2)*(width//2//2), out_features=64), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([64, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=64*(height//2//2)*(width//2//2), out_features=256), nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)

        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)

        return secret


class MNISTDiscriminator(nn.Module):
    """The image steganography discriminator to assist the training of the image steganography encoder and decoder (Customized for MNIST dataset).

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        in_channel (int): Channel of the input image.
    """
    def __init__(self, in_channel=1):
        super(MNISTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=4, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=4, out_channels=8, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=8, out_channels=1, kernel_size=3), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output


class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label


def data_preprocessing(input_x, min, max):
    '''
    input_x: torch tensor
    min: min value of input_x
    max: max value of input_x

    Return: torch tensor
    '''
    # add random noise
    noise_scale = 0.05 * (max - min)
    random_noise = torch.empty_like(input_x).uniform_(-noise_scale, noise_scale)
    input_x = input_x + random_noise
    input_x = torch.clamp(input_x, min, max)
    
    # scaling
    resize_factor = 2.0
    input_x = torch.nn.Upsample(scale_factor=resize_factor, mode='nearest')(input_x)
    input_x = torch.nn.Upsample(scale_factor=1/resize_factor, mode='nearest')(input_x)

    # quantization
    quantization = 2.0
    input_x = torch.clamp(input_x, min, max / quantization)
    input_x = input_x * quantization
    input_x = torch.clamp(input_x, min, max)

    return input_x


def pytorch_bim_attack(model, image, label, device, epsilon=0.03, bounds_min=0.0, bounds_max=1.0):
    # for cifar
    eps = epsilon * (bounds_max - bounds_min)
    alpha = eps / 500
    num_steps = 2

    # if the data is load by testset, need to expand image dim by unsqueeze
    image = image.clone().detach().to(device)
    label = torch.tensor([label]).to(device)
    
    ori_image = image.clone().detach()
    loss = nn.CrossEntropyLoss()
    
    # untargeted BIM attack
    for _ in range(num_steps):
        image.requires_grad = True
        output = model(image)

        # Calculate loss
        cost = loss(output, label)

        # Update adversarial images, code from github.com/Harry24k/adversarial-attacks-pytorch
        grad = torch.autograd.grad(cost, image,
                                   retain_graph=False,
                                   create_graph=False)[0]

        adv_image = image + alpha*grad.sign()
        a = torch.clamp(ori_image - eps, min=bounds_min)
        b = (adv_image >= a).float() * adv_image + (adv_image < a).float() * a
        c = (b > ori_image+eps).float() * (ori_image+eps) + (b <= ori_image + eps).float() * b
        image = torch.clamp(c, max=bounds_max).detach()

    return image 


GTSRB_DATA_DIR = ''  # data folder
GTSRB_DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
def load_gtsrb_dataset(data_file=('%s/%s' % (GTSRB_DATA_DIR, GTSRB_DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    return X_train, Y_train, X_test, Y_test


class Normalize:
    """Normalization of images.

    Args:
        dataset_name (str): the name of the dataset to be normalized.
        expected_values (float): the normalization expected values.
        variance (float): the normalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar10" or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone
