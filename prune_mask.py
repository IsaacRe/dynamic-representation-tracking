from model import IncrNet
import cv2
from copy import deepcopy
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import time
import numpy as np
import copy
import subprocess
import os
import torch.multiprocessing as mp
import atexit
import sys
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from context_merger import ContextMerger
from dataset_incr_cifar import iCIFAR10, iCIFAR100
from dataset_batch_cifar import CIFAR20
from csv_writer import CSVWriter
from feature_matching import match, between_net_correlation
from feature_vis_2 import PatchTracker
from feature_generalizability import ANOVATracker
from vc_utils.vc_dataset import set_base_dataset, get_vc_dataset, test_vc_accuracy
from vc_utils.activation_tracker import ActivationTracker
set_base_dataset('CIFAR')

parser = argparse.ArgumentParser(description="Incremental learning")

# Saving options
parser.add_argument("--outfile", default="results/temp.csv", type=str,
                    help="Output file name (should have .csv extension)")
parser.add_argument("--save_all", dest="save_all", action="store_true",
                    help="Option to save models after each "
                         "test_freq number of learning exposures")
parser.add_argument("--save_all_dir", dest="save_all_dir", type=str,
                    default=None, help="Directory to store all models in")
parser.add_argument("--resume", dest="resume", action="store_true",
                    help="Resume training from checkpoint at outfile")
parser.add_argument("--resume_outfile", default=None, type=str,
                    help="Output file name after resuming")
parser.add_argument("--explr_start", action='store_true',
                    help="Whether to start from a full exemplar set")
parser.add_argument("--test_saved", action='store_true', help='Whether to test model accuracy of loaded checkpoints')

# Arguments when resuming from exemplar-batch-trained model
parser.add_argument("--batch_coverage", type=str, default=None,
                    help="The coverage file containing class info for batch training (used when pretraining from batch)")
parser.add_argument("--explr_model", type=str, required=False,
                    help="Model to load exemplars from")

# Hyperparameters
parser.add_argument("--init_lr", default=0.002, type=float,
                    help="initial learning rate")
parser.add_argument("--init_lr_ft", default=0.005, type=float,
                    help="Init learning rate for balanced finetuning (for E2E)")
parser.add_argument("--num_epoch", default=1, type=int,
                    help="Number of epochs")
parser.add_argument("--num_epoch_ft", default=10, type=int,
                    help="Number of epochs for balanced finetuning (for E2E)")
parser.add_argument("--lrd", default=5, type=float,
                    help="Learning rate decrease factor")
parser.add_argument("--wd", default=0.00001, type=float,
                    help="Weight decay for SGD")
parser.add_argument("--batch_size", default=64, type=int,
                    help="Mini batch size for training")
parser.add_argument("--llr_freq", default=10, type=int,
                    help="Learning rate lowering frequency for SGD (for E2E)")
parser.add_argument("--batch_size_test", default=100, type=int,
                    help="Mini batch size for testing")
parser.add_argument("--batch_size_ft_fc", type=int, default=100,
                    help="Mini batch size for final fc finetuning")

# Incremental options
parser.add_argument("--lexp_len", default=100, type=int,
                    help="Number of frames in Learning Exposure")
parser.add_argument("--num_exemplars", default=400, type=int,
                    help="number of exemplars")
parser.add_argument("--img_size", default=224, type=int,
                    help="Size of images input to the network")
parser.add_argument("--total_classes", default=20, type=int,
                    help="Total number of classes")
parser.add_argument("--num_classes", default=1, type=int,
                    help="Number of classes for each learning exposure")
parser.add_argument("--num_iters", default=1000, type=int,
                    help="Total number of learning exposures (currently"
                         " only integer multiples of args.total_classes"
                         " each class seen equal number of times)")
parser.add_argument("--fix_explr", action='store_true',
                    help="Fix the number of exemplars per class")
parser.add_argument('--mix_class', action='store_true',
                    help='Whether to split into subtasks after getting permutation (same class potentially '
                         'in multiple subtasks)')

# Model options
parser.add_argument("--algo", default="icarl", type=str,
                    help="Algorithm to run. Options : icarl, e2e, lwf")
parser.add_argument("--dist", dest="dist", action="store_true",
                    help="Option to switch off distillation loss")
parser.add_argument("--no_pt", dest="pretrained", action="store_false",
                    help="Option to start from an ImageNet pretrained model")
parser.add_argument('--sample', default='wg', type=str,
                    help='Sampling mechanism to be performed')
parser.add_argument('--loss', default='BCE', type=str,
                    help='Loss to be used in classification')
parser.add_argument('--file_path', default='', type=str,
                    help='Path to csv file of pretrained model')
parser.add_argument('--fixed_ex', dest='fixed_ex', action='store_true',
                    help='Option to use a fixed set of exemplars')

# Training options
parser.add_argument("--fix_exposure", action='store_true', help="Fix order of class exposures")
parser.add_argument("--diff_order", dest="d_order", action="store_true",
                    help="Use a random order of classes introduced")
parser.add_argument("--subset", dest="subset", action="store_true",
                    help="Use a random subset of total_classes classes, instead of the first total_classes classes")
parser.add_argument("--no_jitter", dest="jitter", action="store_false",
                    help="Option for no color jittering (for iCaRL)")
parser.add_argument("--h_ch", default=0.02, type=float,
                    help="Color jittering : max hue change")
parser.add_argument("--s_ch", default=0.05, type=float,
                    help="Color jittering : max saturation change")
parser.add_argument("--l_ch", default=0.1, type=float,
                    help="Color jittering : max lightness change")
parser.add_argument("--aug", default="icarl", type=str,
                    help="Data augmentation to perform on train data")
parser.add_argument("--s_wo_rep", dest="sample_w_replacement", action="store_false",
                    help="Sampling train data with replacement")

# Finetune final FC options
parser.add_argument('--ft_fc', action='store_true',
                    help='Whether to conduct training on final fc after each lexp')
parser.add_argument('--ft_fc_epochs', type=int, default=10,
                    help='Number of epochs to finetune fc layer on')
parser.add_argument('--ft_fc_lr', type=float, default=0.002,
                    help='Lr for fc layer finetuning')

# Matching Feature Correlation Analysis options
parser.add_argument('--corr_model_type', type=str, choices=['incr, batch'], default='incr',
                    help='Which type of pretrained model to use for correlation analysis')
parser.add_argument('--corr_model_incr', type=str, default='incr_model-cifar20-model.pth.tar',
                    help='Specify IncrNet for feature match correlation')
parser.add_argument('--corr_model_batch', type=str, default='batch_model/batch_model-cifar20-model.pth.tar',
                    help='Specify batch trained model for correlation')
parser.add_argument('--feat_corr', action='store_true', help='Conduct matching feature correlation analysis')
parser.add_argument('--sequential_corr', action='store_true', help='Compute correlation between each sequential pair'
                                                                   'of model features')
parser.add_argument('--corr_feature_idx', type=int, default=7,
                    help='Index of the layer in model.features over which correlations will be computed')
parser.add_argument("--batch_size_corr", type=int, default=50,
                    help="Mini batch size for computing correlations (keep < 64)")
parser.add_argument('--save_matr', action='store_true', help='Save correlation matrices during each iteration')

# Feature Visualization Options
parser.add_argument('--probe', action='store_true',
                    help='Carry out probing of feature map with previously obtained patches')
parser.add_argument('--feat_vis_layer_name', nargs='+', type=str, default=['layer2.0.conv1'],
                    help='Index of the layer at which to conduct visualization')
parser.add_argument('--track_grad', action='store_true',
                    help='Whether to track gradient information as well as activations when probing')
parser.add_argument('--class_anova', action='store_true', help='Track ANOVA of class-specific activation distributions')

# Visual Concept (VC) Tracking Options
parser.add_argument('--track_vc', action='store_true', help='Track VCs corresponding to filter encodings of corr_model')
parser.add_argument('--absent_vc_threshold', type=float, default=0.0,
                    help='Filter activation threshold below which the corresponding visual concept will be'
                         ' considered absent')
parser.add_argument('--present_vc_threshold', type=float, default=1.0,
                    help='Filter activation threshold above which the corresponding visual concept will be'
                         'considered present')
parser.add_argument('--should_prune', action='store_true',
                    help='Actually performs pruning as opposed to just computing the masks')
parser.add_argument('--save_pruned_path', type=str, default='prune_masks.npz',
                    help='Save path for computed prune masks')
parser.add_argument('--load_pruned_path', type=str, default='prune_masks.npz',
                    help='Load path for computed prune masks')
parser.add_argument('--nosave_prune_mask', action='store_false', dest='save_prune_mask',
                    help='Do not save computed prune masks to save path')
parser.add_argument('--noload_prune_mask', action='store_false', dest='load_prune_mask',
                    help='Do not load prune masks from load path')
parser.add_argument('--prune_ratio', type=float, default=0.8,
                    help='Ratio of filters to prune before compiling visual concepts dataset from filter'
                         ' activations')
parser.add_argument('--vc_dataset_size', type=int, default=10000,
                    help='Number of datapoints for the binary classification task of each visual concept')
parser.add_argument('--save_vc_weights', action='store_true', help='Store weights resulting from classification'
                                                                   ' layer training')
parser.add_argument('--lr_vc', type=float, default=0.01, help='Learning rate for VC classification layer')
parser.add_argument('--batch_size_train_vc', type=int, default=100, help='Batch size for training VC classification'
                                                                         ' layer')
parser.add_argument('--batch_size_test_vc', type=int, default=100, help='Batch size for testing VC classification')

# System options
parser.add_argument("--test_freq", default=1, type=int,
                    help="Number of iterations of training after"
                         " which a test is done/model saved")
parser.add_argument("--num_workers", default=8, type=int,
                    help="Maximum number of threads spawned at any" 
                         "stage of execution")
parser.add_argument("--one_gpu", dest="one_gpu", action="store_true",
                    help="Option to run multiprocessing on 1 GPU")
parser.add_argument("--debug", action='store_true',
                    help="Set DataLoaders to num_workers=0 for debugging in data iteration loop")

parser.add_argument('--diff_perm', action='store_true')
parser.add_argument('--seed', type=int, default=1, help='Set torch and numpy seed')

parser.set_defaults(ncm=False)
parser.set_defaults(dist=False)
parser.set_defaults(pretrained=True)
parser.set_defaults(d_order=False)
parser.set_defaults(subset=False)
#parser.set_defaults(jitter=True)
parser.set_defaults(save_all=False)
parser.set_defaults(resume=False)
parser.set_defaults(one_gpu=False)
parser.set_defaults(sample_w_replacement=True)

# Print help if no arguments passed
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
if args.debug:
    args.num_workers = 1

if args.track_grad:
    args.probe = True

torch.manual_seed(args.seed)

num_iters = args.num_iters
test_freq = args.test_freq
load_iters = list(range(0, num_iters, test_freq))

def load_model(i, device=0):
    model = IncrNet(args, device=device, cifar=True)
    model.from_resnet(args.save_all_dir + '-saved_models/model_iter_%d.pth.tar' % i, load_fc=True)
    return model

def get_sad_acc(mask1, mask2, key):
    # print("mask1 shape: ", mask1.shape)
    # print("mask2 shape: ", mask2.shape)
    assert(mask1.shape == mask2.shape)
    length = len(mask1)

    diffs = np.abs(mask1-mask2)
    sum_diff = np.sum(diffs)

    # return 100 - (sum_diff/length)*100
    return 1 - (sum_diff / length)

def get_recall(gt, mask):
    if gt.shape == mask.shape:
        assert(gt.shape == mask.shape)

        arr = mask[gt == 1]

        return np.sum(arr) / len(arr)
    else:
        return 0

def get_avg_acc(md1, md2):
    assert(md1.keys() == md2.keys())
    sum_acc = 0
    num_params = 0
    for key in md1.keys():
        mask1 = md1[key]
        mask2 = md2[key]
        num_params += len(mask1)
        sum_acc += get_sad_acc(mask1, mask2, key) * len(mask1)

    return (sum_acc / num_params) * 100

def get_avg_recall(gt_md, md):
    assert(gt_md.keys() == md.keys())
    sum_rec = 0
    num_params = 0
    for key in gt_md.keys():
        gt = gt_md[key]
        mask = md[key]
        num_params += len(mask)
        sum_rec += get_recall(gt, mask) * len(mask)

    return (sum_rec / num_params) * 100

def prune_mask(i, percent, mod=None):
    mask_dicts = {}
    if mod is None:
        model = load_model(i)
    else:
        model = mod

    t_sum = 0

    for name, param in model.named_parameters():
        if "weight" in name and "fc" not in name:
            alive = param.data.cpu().numpy()
            alive = alive[np.nonzero(alive)]

            threshold = np.percentile(abs(alive), percent, interpolation="midpoint")
            mask = np.where(abs(alive) < threshold, 0, 1)
            mask_dicts[name] = mask

    return mask_dicts

def store_prune_mask_model(model, percent, save_all_dir):
    mask_dicts = prune_mask(0, percent, model)
    np.savez("%s/model_iter_%d.npz" % (save_all_dir, 0), **mask_dicts)

def store_prune_mask(i, percent, save_all_dir):
    mask_dicts = prune_mask(i, percent)
    np.savez("%s/model_iter_%d.npz" % (save_all_dir, i), **mask_dicts)
    # with open("%s/model_iter_%d.pkl" % (save_all_dir, i), "wb") as f:
        # pickle.dump(mask_dicts, f, pickle.HIGHEST_PROTOCOL)

def total_data(percent, total=5000, freq=10):
    final_md = prune_mask(total-freq, percent)
    accs = []
    recs = []

    for i in range(0, total, freq):
        md = prune_mask(i, percent)
        acc = get_avg_acc(final_md, md)
        rec = get_avg_recall(final_md, md)
        accs.append(acc)
        recs.append(rec)

    return accs, recs

def get_metric(prev, curr, final):
    m1 = 0
    m2 = 0
    total_params = 0

    for key in prev.keys():
        prev_mask = prev[key]
        curr_mask = curr[key]
        final_mask = final[key]

        diffs = prev_mask != curr_mask
        m1 += np.sum(diffs)
        m2 += np.sum(curr_mask[diffs] == final_mask[diffs])
        total_params += len(prev_mask)

    return (m1 / total_params) * 100, (m2 / m1) * 100


def get_metrics_total(percent, total=5000, freq=10):
    last = total - freq
    final_md = prune_mask(last, percent)

    m1s = []
    m2s = []

    prev = prune_mask(0, percent)
    for i in range(freq, total, freq):
        curr = prune_mask(i, percent)
        m1, m2 = get_metric(prev, curr, final_md)
        m1s.append(m1)
        m2s.append(m2)

    return m1s, m2s

def main():
    percent = 80
    total_iter = 5000
    test_freq = 10
    # accs, recs = total_data(percent)
    m1s, m2s = get_metrics_total(percent)
    xs = [i for i in range(0, total_iter, test_freq)]
    xticks = [i for i in range(0, total_iter, 500)]

    # sns.set()
    # sns.set_palette("deep")
    # plt.plot(xs, accs, label="accuracy")
    # plt.plot(xs, recs, label="recall")
    # plt.xlabel("Iterations")
    # plt.ylabel("Percentage")
    # plt.xticks(xticks)
    # plt.legend()
    # plt.savefig("prune.png")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Prune Accuracy and Recall vs Iterations")
    ax1.plot(xs, m1s, label="Metric 1")
    ax2.plot(xs, m2s, label="Metric 2")
    ax2.set_xlabel("Iterations")

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, rotation=50, ha="right")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, rotation=50, ha="right")

    ax1.set_ylabel("Metric 1")
    ax2.set_ylabel("Metric 2")
    # ax1.legend()
    # ax2.legend()
    # fig.show()
    fig.savefig("prune.png")

if __name__ == "__main__":
    main()
