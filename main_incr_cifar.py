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

# ensure samples is a multiple of num_classes
args.lexp_len = (args.lexp_len // args.num_classes) * args.num_classes

torch.backends.cudnn.benchmark = True
mp.set_sharing_strategy("file_system")
expt_githash = subprocess.check_output(["git", "describe", "--always"])

# Defines transform
transform = transforms.ColorJitter(hue=args.h_ch, saturation=args.s_ch, brightness=args.l_ch)

# multiprocessing on a single GPU
if args.one_gpu:
    mp.get_context("spawn")

# GPU indices
train_device = 0
test_device = 1
if args.one_gpu:
    test_device = 0

if not os.path.exists(os.path.dirname(args.outfile)):
    if len(os.path.dirname(args.outfile)) != 0:
        os.makedirs(os.path.dirname(args.outfile))

batch_pt = False
if args.resume and 'batch' in os.path.splitext(args.outfile)[0]:
    batch_pt = True

num_classes = args.num_classes
test_freq = 1
total_classes = args.total_classes
num_iters = args.num_iters

# Conditional variable, shared vars for synchronization
"""
cond_var = mp.Condition()
train_counter = mp.Value("i", 0)
test_counter = mp.Value("i", 0)
dataQueue = mp.Queue()
all_done = mp.Event()
data_mgr = mp.Manager()
"""

if not os.path.exists("data_generator/cifar_mean_image.npy"):
    mean_image = None
else:
    mean_image = np.load("data_generator/cifar_mean_image.npy")

K = args.num_exemplars  # total number of exemplars

model = IncrNet(args, device=train_device, cifar=True)
if args.resume_outfile:
    model.from_resnet(args.resume_outfile)

corr_model = None
if args.feat_corr or args.track_vc:
    if args.corr_model_incr:
        copy_args = deepcopy(args)
        copy_args.pretrained = True
        copy_args.file_path = args.corr_model_incr.split('-model.')[0]
        corr_model = IncrNet(copy_args, device=test_device, cifar=True)
        corr_model = corr_model.model
    elif args.corr_model_batch:
        corr_model = IncrNet(args, device=test_device, cifar=True)
        corr_model.from_resnet(args.corr_model_batch)
        corr_model = corr_model.model
    else:
        corr_model = models.resnet34(pretrained=True)
    corr_model.eval()

if not args.mix_class:
    assert total_classes % num_classes == 0

def group_classes(classes):
    if num_classes > 1:
        return [classes[i:i+num_classes] for i in range(0, len(classes), num_classes)]
    return classes

# Randomly choose a subset of classes
if batch_pt:
    perm_id = np.array(list(set(np.load(args.batch_coverage)['classes_seen'])))
elif args.subset and args.d_order:
    perm_id = np.random.choice(100, total_classes, replace=False)
    perm_id = np.random.permutation(perm_id)
elif args.d_order:
    perm_id = np.random.permutation(total_classes)
else:
    perm_id = np.arange(total_classes)

all_classes = list(perm_id)
if not args.mix_class:
    perm_id = group_classes(list(perm_id))

print("perm_id:", perm_id)

if args.num_iters > len(perm_id):
    if args.num_iters % len(perm_id) != 0:
        raise Exception('Must have num_iters % total_classes / num_classes = 0')

    # Multiply num_repetitions by number of classes per epoch so number of total learning exposures remains same
    num_repetitions = args.num_iters // total_classes * num_classes
    perm_file = "permutation_files/permutation_%d_%d.npy" \
                % (total_classes // num_classes, num_repetitions)

    if not os.path.exists(perm_file) or args.diff_perm:
        os.makedirs("permutation_files", exist_ok=True)
        # Create random permutation file and save
        perm_arr = np.array(num_repetitions
                            * list(np.arange(len(perm_id))))
        np.random.shuffle(perm_arr)
        np.save(perm_file, perm_arr)
    else:
        print("Loading permutation file: %s" % perm_file)

    perm_id_all = list(np.load(perm_file))
    print("PERM ID ALL: ===>", perm_id_all)
    for i in range(len(perm_id_all)):
        perm_id_all[i] = perm_id[perm_id_all[i]]
    perm_id = perm_id_all

if args.mix_class:
    perm_id = group_classes(list(perm_id))

assert len(perm_id) == args.num_iters
np.random.seed(args.seed)

train_set = iCIFAR100(args, root="./data",
                             train=True,
                             n_classes=num_classes,
                             download=True,
                             transform=transform,
                             mean_image=mean_image)
test_set = iCIFAR100(args, root="./data",
                             train=False,
                             n_classes=num_classes,
                             download=True,
                             transform=None,
                             mean_image=mean_image)
if args.ft_fc:
    train_fc_set = CIFAR20(all_classes,
                           root='./data',
                           train=True,
                           download=True,
                           transform=transform,
                           mean_image=mean_image)
test_all_set = CIFAR20(all_classes,
                      root='./data',
                      train=False,
                      download=True,
                      transform=None,
                      mean_image=mean_image)

# set up running feature tracking
if args.probe:
    probe = PatchTracker(args.feat_vis_layer_name, test_all_set, track_grad=args.track_grad, device=test_device,
                         patch_file='cifar%d' % total_classes, save_file=args.outfile,
                         num_workers=0 if args.debug else args.num_workers)

if args.class_anova:
    anova = ANOVATracker(args.feat_vis_layer_name, args.outfile.split('.')[0] + '-F_stats.npz', device=test_device,
                         n_classes=total_classes)

vc_module_name = args.feat_vis_layer_name[-1]

print(len(train_set))
print(num_classes)
acc_matr = np.zeros((total_classes, num_iters))
if args.ft_fc:
    acc_matr_fc = np.zeros_like(acc_matr)
coverage = np.zeros((total_classes, num_iters))
n_known = np.zeros(num_iters, dtype=np.int32)

classes_seen = []
model_classes_seen = []  # Class index numbers stored by model
exemplar_data = []  # Exemplar set information stored by th
# acc_matr row index represents class number and column index represents
# learning exposure.
acc_matr = np.zeros((args.total_classes, args.num_iters))

# Conditional variable, shared memory for synchronization
cond_var = mp.Condition()
train_counter = mp.Value("i", 0)
test_counter = mp.Value("i", 0)
train_fc_counter = mp.Value("i", 0)
dataQueue = mp.Queue()
all_done = mp.Event()
data_mgr = mp.Manager()


expanded_classes = data_mgr.list([None for i in range(args.test_freq)])

if args.resume or args.explr_start:
    if not batch_pt and not args.explr_start:
        print("resuming model from %s-model.pth.tar" %
              os.path.splitext(args.outfile)[0])

        model = torch.load("%s-model.pth.tar" % os.path.splitext(args.outfile)[0],
                           map_location=lambda storage, loc: storage)
        model.device = train_device

        model.exemplar_means = []
        model.compute_means = True

        info_coverage = np.load("%s-coverage.npz" \
            % os.path.splitext(args.outfile)[0])
        info_matr = np.load("%s-matr.npz" % os.path.splitext(args.outfile)[0])
        if expt_githash != info_coverage["expt_githash"]:
            print("Warning : Code was changed since the last time model was saved")
            print("Last commit hash : ", info_coverage["expt_githash"])
            print("Current commit hash : ", expt_githash)

        args_resume_outfile = args.resume_outfile
        perm_id = info_coverage["perm_id"]
        num_iters_done = model.num_iters_done
        acc_matr = info_matr['acc_matr']
        args = info_matr["args"].item()

        if args_resume_outfile is not None:
            args.outfile = args.resume_outfile = args_resume_outfile
        else:
            print("Overwriting old files")

        model_classes_seen = list(
            info_coverage["model_classes_seen"][:num_iters_done])
        classes_seen = list(info_coverage["classes_seen"][:num_iters_done])
        coverage = info_coverage["coverage"]
        train_set.all_train_coverage = info_coverage["train_coverage"]

        train_counter = mp.Value("i", num_iters_done)
        test_counter = mp.Value("i", num_iters_done)

        # expanding test set to everything seen earlier
        for mdl_cl, gt_cl in zip(model_classes_seen, classes_seen):
            """
            # expanding test set to everything seen earlier
            for i, (mdl_cl, gt_cl) in enumerate(zip(model_classes_seen, classes_seen)):
                if mdl_cl not in model_classes_seen[:i]:
            """
            print("Expanding class for resuming : %d, %d" %(mdl_cl, gt_cl))
            test_set.expand([mdl_cl], [gt_cl])

        # Ensuring requires_grad = True after model reload
        for p in model.parameters():
            p.requires_grad = True
    else:
        if args.num_exemplars > 0:
            print("resuming model from %s-model.pth.tar" %
                  os.path.splitext(args.outfile)[0])

            if args.resume:
                model.from_resnet("%s-model.pth.tar" % os.path.splitext(args.outfile)[0])

                # Ensuring requires_grad = True after model reload
                for p in model.parameters():
                    p.requires_grad = True

            if args.explr_model is None:
                coverage_model = '%s-model.pth.tar' % (args.batch_coverage.split('-coverage')[0])
                assert os.path.exists(coverage_model), "Could not find model corresponding to specified coverage file"
                args.explr_model = coverage_model
            model_with_explrs = torch.load(args.explr_model, map_location=lambda storage, loc: storage)
            assert set(model_with_explrs.classes) == set(all_classes)
            all_classes = model_with_explrs.classes
            # Transfer exemplars to the newly created model
            model.exemplar_sets = []
            model.eset_le_maps = []
            for explr, le_maps in zip(model_with_explrs.exemplar_sets, model_with_explrs.eset_le_maps):
                num_sample = args.num_exemplars if args.fix_explr else args.num_exemplars // total_classes
                assert explr.shape[0] >= num_sample
                sample = np.random.choice(np.arange(explr.shape[0]), num_sample)
                model.exemplar_sets += [explr[sample]]
                model.eset_le_maps += [le_maps[sample]]
            num_explr = args.num_exemplars if not args.fix_explr else args.num_exemplars * total_classes
            assert len(model.exemplar_sets) * model.exemplar_sets[0].shape[0] == args.num_exemplars, \
                "Specified exemplar set does not match exemplar size provided"
            model.compute_means = True
            model.exemplar_means = model_with_explrs.exemplar_means
            model.n_occurrences = [0] * total_classes

            model.num_iters_done = 0
            if args.resume:
                info_matr = np.load("%s-matr.npz" % os.path.splitext(args.outfile)[0])
                args_resume_outfile = args.resume_outfile
                model.num_iters_done = 0
                num_iters_done = model.num_iters_done
                new_args = info_matr["args"].item()
                for k, v in new_args.__dict__.items():
                    args.__dict__[k] = v

                if args_resume_outfile is not None:
                    args.outfile = args.resume_outfile = args_resume_outfile
                else:
                    print("Overwriting old files")
            num_iters_done = model.num_iters_done

            train_counter = mp.Value("i", num_iters_done)
            test_counter = mp.Value("i", num_iters_done)

        # initialize model with output channels for all classes
        model.increment_classes(all_classes)
        all_model_classes = [model.classes_map[c] for c in all_classes]

        # expanding test set to all classes
        for mdl_cl, gt_cl in zip(all_model_classes, all_classes):
            print("Expanding class for resuming : %d, %d" %(mdl_cl, gt_cl))
            test_set.expand([mdl_cl], [gt_cl])

save_data = ['Iteration', 'Model_classes', 'Test_accuracy', 'Train_loss', 'Exposure_time', 'Test_time']
if args.ft_fc:
    save_data += ['FTFC_accuracy', 'FTFC_time']
if args.feat_corr:
    save_data += ['Feat_match_correlation', 'Correlation_time']
writer = CSVWriter(args.outfile, *save_data)

def train_run(device):
    global train_set
    model.cuda(device=device)

    train_wait_time = 0
    s = len(classes_seen)
    print("####### Train Process Running ########")
    print("Args: ", args)
    train_wait_time = 0

    running_activations = {}

    while s < args.num_iters:
        if s == 0:
            cond_var.acquire()
            train_counter.value += 1
            expanded_classes[s % args.test_freq] = None

            if train_counter.value == test_counter.value + args.test_freq:
                temp_model = copy.deepcopy(model)
                temp_model.cpu()
                write_data = {'Train_loss': np.nan, 'Exposure_time': np.nan}
                dataQueue.put((temp_model, write_data))
            cond_var.notify_all()
            cond_var.release()

            s += 1
            continue

        time_ptr = time.time()
        # Do not start training till test process catches up
        cond_var.acquire()
        # while loop to avoid spurious wakeups
        while test_counter.value + args.test_freq <= train_counter.value:
            print("[Train Process] Waiting on test process")
            print("[Train Process] train_counter : ", train_counter.value)
            print("[Train Process] test_counter : ", test_counter.value)
            cond_var.wait()
        cond_var.release()
        train_wait_time += time.time() - time_ptr

        start_time = time.time()

        # Keep a copy of previous model for distillation
        prev_model = copy.deepcopy(model)
        prev_model.cuda(device=device)
        for p in prev_model.parameters():
            p.requires_grad = False

        curr_class = perm_id[s]
        if not hasattr(curr_class, '__iter__'):
            curr_class = [curr_class]

        classes_seen.append(curr_class)
        curr_expanded = []
        for c in curr_class:

            if c not in model.classes_map and c not in curr_expanded:
                curr_expanded += [c]
        model.increment_classes(curr_expanded)
        model.cuda(device=device)

        model_curr_class_idx = [model.classes_map[c] for c in curr_class]
        model_classes_seen.append(model_curr_class_idx)

        # Load Datasets
        print("Loading training examples for"\
              " class indexes (%s), (%s), at iteration %d" %
              (', '.join(map(lambda x: str(x), model_curr_class_idx)), ', '.join(map(lambda x: str(x), curr_class)), s))
        train_set.load_data_class(curr_class, model_curr_class_idx, s)

        model.train()

        mean_loss = model.update_representation_icarl(train_set,
                                                      prev_model,
                                                      model_curr_class_idx,
                                                      args.num_workers)

        model.eval()
        del prev_model
        m = args.num_exemplars if args.fix_explr else int(K / model.n_classes)

        if not args.fix_explr:
            model.reduce_exemplar_sets(m)
        # Construct exemplar sets for current class
        print("Constructing exemplar set for class index (%s) , (%s) ..." %
              (', '.join(map(lambda x: str(x), model_curr_class_idx)), ', '.join(map(lambda x: str(x), curr_class))),
              end="")
        images, le_maps = train_set.get_image_classes(
                model_curr_class_idx)
        mean_images = np.array([mean_image]*len(images[0]))
        model.construct_exemplar_sets(images,
                                      mean_images,
                                      le_maps, None, m,
                                      model_curr_class_idx, s)
        model.n_known = model.n_classes

        n_known[s] = model.n_known
        print("Model num classes : %d, " % model.n_known)

        if s > 0:
            coverage[:,s] = coverage[:,s-1]
        coverage[model_curr_class_idx,s] = \
                            train_set.get_train_coverages(perm_id[s])
        print("Coverage of current classes now: %s" %
                    ', '.join(map(lambda x: str(x), coverage[model_curr_class_idx,s])))

        for y, P_y in enumerate(model.exemplar_sets):
            print("Exemplar set for class-%d:" % (y), P_y.shape)

        exemplar_data.append(list(model.eset_le_maps))

        # Time learning exposures
        lexp_time = time.time() - start_time

        cond_var.acquire()
        train_counter.value += 1
        if len(curr_expanded) > 0:
            expanded_classes[s % args.test_freq] = model_curr_class_idx, curr_class
        else:
            expanded_classes[s % args.test_freq] = None

        if train_counter.value == test_counter.value + args.test_freq:
            temp_model = copy.deepcopy(model)
            temp_model.cpu()
            write_data = {'Train_loss': mean_loss, 'Exposure_time': lexp_time}
            dataQueue.put((temp_model, write_data))
        cond_var.notify_all()
        cond_var.release()

            
            
        np.savez(args.outfile[:-4] + "-coverage.npz", \
                 coverage=coverage, \
                 perm_id=perm_id, n_known=n_known, \
                 train_coverage=train_set.all_train_coverage, \
                 model_classes_seen=model_classes_seen, \
                 classes_seen=classes_seen, \
                 expt_githash=expt_githash, \
                 exemplar_data=np.array(exemplar_data))
        # loop var increment
        s += 1

    time_ptr = time.time()
    all_done.wait()
    train_wait_time += time.time() - time_ptr
    print("[Train Process] Done, total time spent waiting : ", train_wait_time)


def test_run(device):
    global test_set
    corr_model_ = corr_model
    print("####### Test Process Running ########")
    test_model = None
    s = args.test_freq * (len(classes_seen)//args.test_freq)

    # Initialize VC Dataset
    # TODO should we use transform?
    vc_save_file = args.outfile.split('.')[0] + '-vc_acc'
    writers = [writer]
    if args.track_vc:
        assert corr_model is not None, 'No corr_model specified for Visual Concept identification'
        vc_dataset = get_vc_dataset(args, corr_model, args.feat_vis_layer_name[-1], all_classes,
                                    root='./data', train=True, transform=transform, mean_image=mean_image, download=False)
        vc_writer = CSVWriter(vc_save_file + '.csv', 'Iteration', *(str(idx) for idx in vc_dataset.kept_idxs))
        writers += [vc_writer]

    vc_weights = []

    test_wait_time = 0
    with ContextMerger(*writers):
        while s < args.num_iters:

            # Wait till training is done
            time_ptr = time.time()
            cond_var.acquire()
            while train_counter.value < test_counter.value + args.test_freq:
                print("[Test Process] Waiting on train process")
                print("[Test Process] train_counter : ", train_counter.value)
                print("[Test Process] test_counter : ", test_counter.value)
                cond_var.wait()
            cond_var.release()
            test_wait_time += time.time() - time_ptr

            cond_var.acquire()
            prev_model = None
            if s > 0:
                prev_model = test_model
            test_model, write_data = dataQueue.get()
            expanded_classes_copy = copy.deepcopy(expanded_classes)
            test_counter.value += args.test_freq
            cond_var.notify_all()
            cond_var.release()

            # write data from train process
            writer.write(**write_data)

            start_time = time.time()

            # test set only needs to be expanded
            # when a new exposure is seen
            for expanded_class in expanded_classes_copy:
                if expanded_class is not None:
                    model_cl, gt_cl = expanded_class
                    print("[Test Process] Loading test data")
                    test_set.expand(model_cl, gt_cl)

            print("[Test Process] Test Set Length:", len(test_set))


            test_model.device = device
            test_model.cuda(device=device)
            test_model.eval()
            if args.ft_fc:
                train_loader = torch.utils.data.DataLoader(train_fc_set,
                                                           batch_size=args.batch_size_ft_fc, shuffle=True,
                                                           num_workers=0 if args.debug else args.num_workers,
                                                           pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set,
                                                      batch_size=args.batch_size_test, shuffle=False,
                                                      num_workers=0 if args.debug else args.num_workers,
                                                      pin_memory=True)
            test_all_loader = torch.utils.data.DataLoader(test_all_set,
                                                          batch_size=args.batch_size_test, shuffle=False,
                                                          num_workers=0 if args.debug else args.num_workers,
                                                          pin_memory=True)
            if args.feat_corr:
                correlation_loader = torch.utils.data.DataLoader(test_all_set,
                                                                 batch_size=args.batch_size_corr, shuffle=False,
                                                                 num_workers=0 if args.debug else args.num_workers,
                                                                 pin_memory=True)

            writer.write(Model_classes=test_model.n_known, Iteration=s)

            ############################# Test Accuracy (Seen Classes) ######################################

            # make sure test_set has been populated
            if len(test_set) > 0:
                print("[Test Process] Computing Accuracy matrix...")
                # TODO make sure computing and storing accuracies currectly during multi-class per exposure
                all_labels = []
                all_preds = []
                with torch.no_grad():
                    for indices, images, labels in test_loader:
                        images = Variable(images).cuda(device=device)

                        preds = test_model.classify(images)
                        all_preds.append(preds.data.cpu().numpy())
                        all_labels.append(labels.numpy())

                    all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                for i in range(test_model.n_known):
                    class_preds = all_preds[all_labels == i]
                    correct = np.sum(class_preds == i)
                    total = len(class_preds)

                    acc_matr[i, s] = (100.0 * correct/total)

                test_acc = np.mean(acc_matr[:test_model.n_known, s])

                # Track test loop time before ft-fc
                test_time = time.time() - start_time
                writer.write(Test_accuracy=test_acc, Test_time=test_time)

                print('[Test Process] =======> Test Accuracy after %d'
                  ' learning exposures : ' %
                  (s + args.test_freq), test_acc)


                print("[Test Process] Saving model and other data")
                test_model.cpu()
                test_model.num_iters_done = s + args.test_freq
                if not args.save_all:
                    torch.save(test_model, "%s-model.pth.tar" %
                               os.path.splitext(args.outfile)[0])
                else:
                    torch.save(test_model, "%s-saved_models/model_iter_%d.pth.tar"\
                                            %(os.path.join(args.save_all_dir, \
                                            os.path.splitext(args.outfile)[0]), s))

                # add nodes for unseen classes to output layer
                test_model.increment_classes([c for c in all_classes if c not in test_model.classes_map])
                test_model.cuda(device=device)

            else:
                writer.write(Test_accuracy=np.nan, Test_time=np.nan)

            ######################### VC Accuracy #######################################################

            if args.track_vc:
                print('[Test Process] Testing accuracy over visual concepts...')
                vc_acc_, vc_weight = test_vc_accuracy(args, test_model.model, vc_module_name, vc_dataset, device=device)
                vc_acc = {str(idx): acc for idx, acc in zip(vc_dataset.kept_idxs, vc_acc_)}
                vc_writer.write(Iteration=s, **vc_acc)
                vc_weights += [vc_weight]
                if args.save_vc_weights:
                    np.save(vc_save_file + '-weights.npy', torch.stack(vc_weights, dim=0).numpy())

                print('[Test Process] Average accuracy over visual concepts after %s iterations: %.2f' %
                      (s, vc_acc_.mean()))

            ########################## ANOVA Class-activation Test ######################################

            if args.class_anova:
                anova.gather(test_model.model, loader=test_all_loader)

            ########################## Activation and Gradient Probing ##################################

            if args.probe:
                probe.probe(test_model.model)

            ########################## Correlation Analysis #############################################

            if args.feat_corr and (prev_model is not None or not args.sequential_corr):
                print('[Test Process] Computing Matching Feature Correlations....', end='')
                start_time = time.time()

                if args.sequential_corr:
                    corr_model_ = prev_model.model

                assert corr_model_, 'Correlation Model was never loaded'

                corr_model_.cuda(device=device)
                assert next(corr_model_.parameters()).is_cuda
                for feat_name in args.feat_vis_layer_name:
                    matches, corr, matrix = match(device, correlation_loader, test_model.model, corr_model_, feat_name,
                                                  replace=False, dump_cache=True)
                    if args.save_matr:
                        matrix_dir = args.outfile.split('.')[0] + '-corr_matrix/'
                        try:
                            os.mkdir(matrix_dir)
                        except FileExistsError:
                            pass
                        np.save(matrix_dir + '%s-matr-%d.npy' % (args.outfile.split('/')[-1].split('.')[0], s), matrix)
                # put save gpu space by putting model back when we're done
                corr_model_.cpu()

                corr_time = time.time() - start_time

                writer.write(Feat_match_correlation=corr.mean(), Correlation_time=corr_time)

                print('done')

            ######################### FT-FC ##########################################################

            if args.ft_fc:
                print('[Test Process] Finetuning FC layer')
                start_time = time.time()
                test_model.train()
                test_model.train_fc(args, train_loader)
                test_model.eval()

                print('[Test Process] Testing FT-FC')
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for indices, images, labels in test_all_loader:
                        images = Variable(images).cuda(device=device)
                        preds = test_model.classify(images)
                        # take only the network predictions
                        if type(preds) is tuple:
                            _, preds = preds
                        all_preds += [preds.data.cpu().numpy()]
                        all_labels += [labels.numpy()]
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                for i in range(total_classes):
                    preds = all_preds[all_labels == i]
                    accuracy = 100. * np.sum(preds == i) / np.sum(all_labels == i)
                    acc_matr_fc[i, s] = accuracy
                test_acc = np.mean(acc_matr_fc[:, s])
                print(test_acc)

                print('[Test Process] Saving FT-FC Accuracy')

                np.save('%s-fc-matr.npz' % os.path.splitext(args.outfile)[0],
                        acc_matr_fc)

                np.savez('%s-matr.npz' % os.path.splitext(args.outfile)[0],
                         acc_matr=acc_matr,
                         model_hyper_params=model.fetch_hyper_params(),
                         args=args, num_iters_done=s)

                # Track ft-fc runtime (in eons lmao)
                ftfc_time = time.time() - start_time
                writer.write(FTFC_accuracy=test_acc, FTFC_time=ftfc_time)

            # loop var increment
            s += args.test_freq

        print("[Test Process] Done, total time spent waiting : ",
              test_wait_time)
        all_done.set()


def cleanup(train_process, test_process):
    train_process.terminate()
    test_process.terminate()


def main():
    train_process = mp.Process(target=train_run, args=(train_device,))
    test_process = mp.Process(target=test_run, args=(test_device,))
    atexit.register(cleanup, train_process, test_process)
    train_process.start()
    test_process.start()

    train_process.join()
    print("Train Process Completeid")
    test_process.join()
    print("Test Process Completed")


if __name__ == "__main__":
    main()