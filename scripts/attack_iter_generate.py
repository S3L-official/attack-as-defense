### This file is used to generate attack costs for analysis.
### Our repo provided models and adv examples, the data were generated by baselines.
### 

import sys
sys.path.append('../')
import numpy as np
import foolbox
import os
import argparse

from keras.models import load_model
from detect.util import get_data
from utils import attack_for_input


def main(args):
    # load the model
    kmodel = load_model('../data/model_%s.h5' % args.dataset)

    root_path = '../results/' 
    folder_name = args.dataset + '_attack_iter_stats/'
    folder_path = os.path.join(root_path, folder_name)
    file_name = args.attack + '_attack_' + args.input

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = folder_path + file_name

    print('Attack %s samples by %s' %(args.input, args.attack))
    # load benign samples
    _, _, x_input, y_input = get_data(args.dataset)
    
    # except wrong label
    preds_test = kmodel.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input.argmax(axis=1))[0]
    
    # load adv samples
    if args.input != 'benign':
        x_input = np.load('../data/Adv_'+ args.dataset +'_' + str(args.input) +'.npy')
    # cw adv file only have 1000 examples, which have been filtered 
    if args.input != 'cw':
        x_input = x_input[inds_correct]

    # only need 1000 samples
    auroc_calc_size = 1000
    x_input = x_input[0: auroc_calc_size]

    # ! get attack cost
    attack_for_input(kmodel, x_input, y_input, method=args.attack, dataset=args.dataset, save_path=save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar'",
        required=False, type=str, default='mnist',
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; recommanded to use JSMA, BIM, BIM2 or DBA. ",
        required=True, type=str
    )
    parser.add_argument(
        '-i', '--input',
        help="The input data to be attacked; either 'fgsm', 'bim', 'jsma', 'cw'. ",
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)
