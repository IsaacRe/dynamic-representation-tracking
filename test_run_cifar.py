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
from vc_utils.vc_dataset import set_base_dataset, get_vc_dataset, test_vc_accuracy_v1, test_vc_accuracy_v2,\
    test_threshold_acc
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
                    help='Actually performs pruning (as opposed to just computing the masks)')
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
parser.add_argument('--validate_multiple_thresholds', action='store_true', help='Validate the thresholds passed below')
parser.add_argument('--thresholds_to_validate', type=float, nargs='*', default=[],
                    help='Threshold values for binary features on which to validate accuracy of final model')
parser.add_argument('--eval_threshold_acc', action='store_true', help='Evaluate accuracy of the thresholded model '
                                                                      'during each exposure')
parser.add_argument('--validate_final_only', action='store_true', help='Exit after final model validation has been '
                                                                       'performed')
parser.add_argument('--vc_epochs', type=int, default=5, help='For thresholded model accuracy validation - '
                                                             'number of epochs to finetune on binary features')
parser.add_argument('--lr_vc', type=float, default=0.1, help='Learning rate for VC classification layer')
parser.add_argument('--batch_size_train_vc', type=int, default=100, help='Batch size for training VC classification'
                                                                         ' layer')
parser.add_argument('--batch_size_test_vc', type=int, default=100, help='Batch size for testing VC classification')
parser.add_argument('--vc_data_balance', action='store_true', help='Balance +/- samples for VC dataset')
parser.add_argument('--vc_recall', action='store_true', help='Compute recall for VC classification performance')
parser.add_argument('--save_activations', action='store_true', help='Store activations of the final model at the'
                                                                    ' feat_vis layer')
parser.add_argument('--lr_threshold', type=float, default=0.04, help='Learning rate for threshold learning')


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

test_model = IncrNet(args, device=0, cifar=True)

def load_model(i, device=0):
    model = IncrNet(args, device=device, cifar=True)
    model.from_resnet(args.save_all_dir + '/model_iter_%d.pth.tar' % i, load_fc=True)
    return model

def reorder_fc(model):
    w = model.fc.weight.data
    w_copy = w.clone()
    classes_seen = np.load(args.save_all_dir + '-coverage.npz')['classes_seen'].flatten()
    classes = []
    for c in classes_seen:
        if c in classes:
            continue
        classes += [c]
        if len(classes) >= 20:
            break
    for i, c in enumerate(classes):
        w[c] = w_copy[i]

corr_model = load_model(load_iters[-1])
corr_model = corr_model.model
reorder_fc(corr_model)
corr_model.eval()

# ensure samples is a multiple of num_classes
args.lexp_len = (args.lexp_len // args.num_classes) * args.num_classes

torch.backends.cudnn.benchmark = True
mp.set_sharing_strategy("file_system")
expt_githash = subprocess.check_output(["git", "describe", "--always"])

if not os.path.exists(os.path.dirname(args.outfile)):
    if len(os.path.dirname(args.outfile)) != 0:
        os.makedirs(os.path.dirname(args.outfile))

all_classes = list(range(args.total_classes))
total_classes = len(all_classes)

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


np.random.seed(args.seed)

# Defines transform
transform = transforms.ColorJitter(hue=args.h_ch, saturation=args.s_ch, brightness=args.l_ch)

if args.test_saved:
    test_set = iCIFAR100(args, root="./data",
                                 train=False,
                                 n_classes=args.num_classes,
                                 download=True,
                                 transform=None,
                                 mean_image=mean_image)
if args.ft_fc or args.eval_threshold_acc or args.validate_multiple_thresholds:
    train_fc_set = CIFAR20(all_classes,
                           root='./data',
                           train=True,
                           download=True,
                           transform=transform,
                           mean_image=mean_image)
if args.ft_fc or args.eval_threshold_acc or args.validate_multiple_thresholds:
    test_all_set = CIFAR20(all_classes,
                          root='./data',
                          train=False,
                          download=True,
                          transform=None,
                          mean_image=mean_image)

# set up running feature tracking
if args.probe:
    probe = PatchTracker(args.feat_vis_layer_name, test_all_set, track_grad=args.track_grad, device=0,
                         patch_file='cifar%d' % total_classes, save_file=args.outfile,
                         num_workers=0 if args.debug else args.num_workers)

if args.class_anova:
    anova = ANOVATracker(args.feat_vis_layer_name, args.outfile.split('.')[0] + '-F_stats.npz', device=0,
                         n_classes=total_classes)

vc_module_name = args.feat_vis_layer_name[-1]

acc_matr = np.zeros((total_classes, num_iters))
if args.ft_fc:
    acc_matr_fc = np.zeros_like(acc_matr)
n_known = np.zeros(num_iters, dtype=np.int32)

classes_seen = []
model_classes_seen = []  # Class index numbers stored by model
exemplar_data = []  # Exemplar set information stored by th
# acc_matr row index represents class number and column index represents
# learning exposure.
acc_matr = np.zeros((args.total_classes, args.num_iters))

expanded_classes = [None for i in range(args.test_freq)]

save_data = ['Iteration', 'Model_classes', 'Test_accuracy', 'Test_time']
if args.ft_fc:
    save_data += ['FTFC_accuracy', 'FTFC_time']
if args.feat_corr:
    save_data += ['Feat_match_correlation', 'Correlation_time']
writer = CSVWriter(args.outfile, *save_data)


def test_run(device):
    global test_set
    corr_model_ = corr_model
    print("####### Test Process Running ########")
    test_model = None

    # Data loader initialization
    if args.ft_fc or args.eval_threshold_acc or args.validate_multiple_thresholds:
        test_all_loader = torch.utils.data.DataLoader(test_all_set,
                                                      batch_size=args.batch_size_test, shuffle=False,
                                                      num_workers=0 if args.debug else args.num_workers,
                                                      pin_memory=True)
    if args.feat_corr:
        correlation_loader = torch.utils.data.DataLoader(test_all_set,
                                                         batch_size=args.batch_size_corr, shuffle=False,
                                                         num_workers=0 if args.debug else args.num_workers,
                                                         pin_memory=True)
    if args.ft_fc or args.eval_threshold_acc or args.validate_multiple_thresholds:
        train_loader = torch.utils.data.DataLoader(train_fc_set,
                                                   batch_size=args.batch_size_ft_fc, shuffle=True,
                                                   num_workers=0 if args.debug else args.num_workers,
                                                   pin_memory=True)

    # VC threshold validation
    if args.validate_multiple_thresholds:
        model_ = load_model(load_iters[-1]).model
        model_.fc = torch.nn.Linear(model_.fc.in_features, total_classes)
        model_.cuda(0)
        t_accs, ft_acc = test_threshold_acc(args, test_all_loader, model_, args.feat_vis_layer_name[-1],
                                            train_loader=train_loader, ts=args.thresholds_to_validate)
        save = {str(t): acc for t, acc in zip(args.thresholds_to_validate, t_accs)}
        save['baseline'] = ft_acc
        np.savez('%s-bin-acc.npz' % args.feat_vis_layer_name[-1],
                 **save)

        if args.validate_final_only:
            sys.exit(0)

    # Initialize VC Dataset
    # TODO should we use transform?
    vc_save_file = args.outfile.split('.')[0] + '-vc_acc'
    writers = [writer]
    if args.track_vc:
        assert corr_model is not None, 'No corr_model specified for Visual Concept identification'
        vc_dataset_train = get_vc_dataset(args, corr_model, args.feat_vis_layer_name[-1], all_classes,
                                          balance=args.vc_data_balance,
                                          root='./data', train=True, transform=None, mean_image=mean_image)
        vc_dataset_test = get_vc_dataset(args, corr_model, args.feat_vis_layer_name[-1], all_classes,
                                         balance=args.vc_data_balance,
                                         root='./data', train=False, transform=None, mean_image=mean_image)
        #acc, w = test_vc_accuracy(args, corr_model, args.feat_vis_layer_name[-1], vc_dataset, vc_dataset_test, device=device,
        #                          uniform_init=True, epochs=10)

        vc_writer = CSVWriter(vc_save_file + '.csv', 'Iteration', *(str(idx) for idx in vc_dataset_test.kept_idxs))
        writers += [vc_writer]
        np.savez('%s-coverage.npz' % vc_save_file, kept_idxs=vc_dataset_test.kept_idxs,
                 sorted_filters=vc_dataset_test.sorted_filters, num_positive=vc_dataset_test.num_p)

    vc_weights = []
    ft_accs = []
    bin_ft_accs = []

    test_wait_time = 0
    with ContextMerger(*writers):
        for s in load_iters:

            prev_model = None
            if s > 0:
                prev_model = test_model

            start_time = time.time()

            # test set only needs to be expanded
            # when a new exposure is seen
            if args.test_saved:
                for expanded_class in expanded_classes:
                    if expanded_class is not None:
                        model_cl, gt_cl = expanded_class
                        print("[Test Process] Loading test data")
                        test_set.expand(model_cl, gt_cl)

                print("[Test Process] Test Set Length:", len(test_set))


            test_model = load_model(s)
            test_model.device = device
            test_model.cuda(device=device)
            test_model.eval()

            if args.test_saved:
                test_loader = torch.utils.data.DataLoader(test_set,
                                                          batch_size=args.batch_size_test, shuffle=False,
                                                          num_workers=0 if args.debug else args.num_workers,
                                                          pin_memory=True)

            writer.write(Model_classes=test_model.n_known, Iteration=s)

            ############################# Thresholded Model Accuracy ########################################

            if args.eval_threshold_acc:
                fc = test_model.fc
                model_ = test_model.model
                model_.fc = torch.nn.Linear(fc.in_features, total_classes)
                model_.fc.cuda(device)
                [t_acc], ft_acc = test_threshold_acc(args, test_all_loader, model_, args.feat_vis_layer_name[-1],
                                                    train_loader=train_loader, ts=[args.present_vc_threshold])
                ft_accs += [ft_acc]
                bin_ft_accs += [t_acc]
                np.savez('%s-bin-acc.npz' % args.outfile.split('.')[0], iteration=s, baseline=np.stack(ft_accs),
                         **{str(args.present_vc_threshold): np.stack(bin_ft_accs)})

            ############################# Test Accuracy (Seen Classes) ######################################

            # make sure test_set has been populated
            if args.test_saved and len(test_set) > 0:
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

                # add nodes for unseen classes to output layer
                test_model.increment_classes([c for c in all_classes if c not in test_model.classes_map])
                test_model.cuda(device=device)

            else:
                writer.write(Test_accuracy=np.nan, Test_time=np.nan)

            ######################### VC Accuracy #######################################################

            if args.track_vc:
                print('[Test Process] Testing accuracy over visual concepts...')
                vc_acc_, vc_weight = test_vc_accuracy_v1(args, test_model.model, vc_module_name, vc_dataset_train,
                                                          vc_dataset_test, device=device, recall=args.vc_recall,
                                                         train=True)
                vc_acc = {str(idx): acc for idx, acc in zip(vc_dataset_test.kept_idxs, vc_acc_)}
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
                         model_hyper_params=test_model.fetch_hyper_params(),
                         args=args, num_iters_done=s)

                # Track ft-fc runtime (in eons lmao)
                ftfc_time = time.time() - start_time
                writer.write(FTFC_accuracy=test_acc, FTFC_time=ftfc_time)

        print("[Test Process] Done, total time spent waiting : ",
              test_wait_time)


def main():
    test_run(0)

    print("Test Process Completed")


if __name__ == "__main__":
    main()
