from model import IncrNet
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
import time
import numpy as np
import cv2
import copy
import subprocess
import os
import torch.multiprocessing as mp
import atexit
import sys
import pdb

from dataset_incr_cifar import iCIFAR10, iCIFAR100

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
parser.add_argument("--init_lr_ft", default=0.001, type=float,
                    help="Init learning rate for balanced finetuning (for E2E)")
parser.add_argument("--num_epoch", default=5, type=int,
                    help="Number of epochs")
parser.add_argument("--num_epoch_ft", default=10, type=int,
                    help="Number of epochs for balanced finetuning (for E2E)")
parser.add_argument("--lrd", default=5, type=float,
                    help="Learning rate decrease factor")
parser.add_argument("--wd", default=0.00001, type=float,
                    help="Weight decay for SGD")
parser.add_argument("--batch_size", default=200, type=int,
                    help="Mini batch size for training")
parser.add_argument("--llr_freq", default=10, type=int,
                    help="Learning rate lowering frequency for SGD (for E2E)")
parser.add_argument("--batch_size_test", default=200, type=int,
                    help="Mini batch size for testing")

# CRIB options
parser.add_argument("--lexp_len", default=100, type=int,
                    help="Number of frames in Learning Exposure")
parser.add_argument("--size_test", default=100, type=int,
                    help="Number of test images per object")
parser.add_argument("--num_exemplars", default=400, type=int,
                    help="number of exemplars")
parser.add_argument("--img_size", default=224, type=int,
                    help="Size of images input to the network")
parser.add_argument("--rendered_img_size", default=300, type=int,
                    help="Size of rendered images")
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

# Model options
parser.add_argument("--algo", default="icarl", type=str,
                    help="Algorithm to run. Options : icarl, e2e, lwf")
parser.add_argument("--no_dist", dest="dist", action="store_false",
                    help="Option to switch off distillation loss")
parser.add_argument("--pt", dest="pretrained", action="store_true",
                    help="Option to start from an ImageNet pretrained model")
parser.add_argument('--ncm', dest='ncm', action='store_true',
                    help='Use nearest class mean classification (for E2E)')
parser.add_argument('--network', dest='network', action='store_true',
                    help='Use network output to classify (for iCaRL)')
parser.add_argument('--sample', default='none', type=str,
                    help='Sampling mechanism to be performed')
parser.add_argument('--explr_neg_sig', dest='explr_neg_sig', action='store_true', help='Option to use exemplars as negative signals (for iCaRL)')
parser.add_argument('--random_explr', dest='random_explr', action='store_true',
                    help='Option for random exemplar set')
parser.add_argument('--loss', default='BCE', type=str,
                    help='Loss to be used in classification')
parser.add_argument('--record_iters', type=int, default=20,
                    help='Number of iterations per epoch to record update statistics')
parser.add_argument('--file_path', default='', type=str,
                    help='Path to csv file of pretrained model')
parser.add_argument('--fixed_ex', dest='fixed_ex', action='store_true',
                    help='Option to use a fixed set of exemplars')
parser.add_argument('--ptr_model', dest='ptr_model', action='store_true',
                    help='Option to use a pretrained a model')

# Training options
parser.add_argument("--fix_exposure", action='store_true', help="Fix order of class exposures")
parser.add_argument("--diff_order", dest="d_order", action="store_true",
                    help="Use a random order of classes introduced")
parser.add_argument("--subset", dest="subset", action="store_true",
                    help="Use a random subset of classes")
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


parser.set_defaults(pre_augment=False)
parser.set_defaults(ncm=False)
parser.set_defaults(dist=False)
parser.set_defaults(pretrained=True)
parser.set_defaults(d_order=False)
parser.set_defaults(subset=False)
parser.set_defaults(jitter=True)
parser.set_defaults(save_all=False)
parser.set_defaults(resume=False)
parser.set_defaults(one_gpu=False)
parser.set_defaults(sample_w_replacement=True)




# Print help if no arguments passed
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
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
cond_var = mp.Condition()
train_counter = mp.Value("i", 0)
test_counter = mp.Value("i", 0)
dataQueue = mp.Queue()
all_done = mp.Event()
data_mgr = mp.Manager()

if not os.path.exists("data_generator/cifar_mean_image.npy"):
    mean_image = None
else:
    mean_image = np.load("data_generator/cifar_mean_image.npy")

K = args.num_exemplars  # total number of exemplars

model = IncrNet(args, device=train_device, cifar=True)

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

print ("perm_id:", perm_id)

if args.num_iters > args.total_classes:
    if not args.num_iters % args.total_classes == 0:
        raise Exception("Currently no support for num_iters%total_classes != 0")

    # Multiply num_repetitions by number of classes per epoch to number of total learning exposures remains same
    num_repetitions = args.num_iters // total_classes * num_classes
    perm_file = "permutation_files/permutation_%d_%d.npy" \
                                    % (total_classes, num_repetitions)

    if not os.path.exists(perm_file) and not args.fix_exposure:
        os.makedirs("permutation_files", exist_ok=True)
        # Create random permutation file and save
        perm_arr = np.array(num_repetitions 
                            * list(np.arange(total_classes)))
        np.random.shuffle(perm_arr)
        np.save(perm_file, perm_arr)
    else:
        print("Loading permutation file: %s" % perm_file)

    if not args.fix_exposure:
        perm_id_all = np.load(perm_file)
    else:
        perm_id_all = np.array(num_repetitions * list(np.arange(total_classes)))
    print("PERM ID ALL: ===>", perm_id_all)
    for i in range(len(perm_id_all)):
        perm_id_all[i] = perm_id[perm_id_all[i]]
    perm_id = perm_id_all

# Group classes together, returns list of length len(perm_id) / num_classes
all_classes = list(set(perm_id))
perm_id = group_classes(perm_id)

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

acc_matr = np.zeros((total_classes, num_iters))
coverage = np.zeros((total_classes, num_iters))
n_known = np.zeros(num_iters, dtype=np.int32)

classes_seen = []
model_classes_seen = []  # Class index numbers stored by model
exemplar_data = []  # Exemplar set information stored by the model
# acc_matr row index represents class number and column index represents
# learning exposure.
if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
    acc_matr = np.zeros((args.total_classes, args.num_iters))
elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
    acc_matr_ncm = np.zeros((args.total_classes, args.num_iters))
    acc_matr_network = np.zeros((args.total_classes, args.num_iters))

# Conditional variable, shared memory for synchronization
"""
cond_var = mp.Condition()
train_counter = mp.Value("i", 0)
test_counter = mp.Value("i", 0)
dataQueue = mp.Queue()
all_done = mp.Event()
data_mgr = mp.Manager()
"""
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
        if (args.algo == "icarl" and not args.network) \
                or (args.algo == "e2e" and not args.ncm):
            acc_matr = info_matr['acc_matr']
        elif (args.algo == "icarl" and args.network) \
                or (args.algo == "e2e" and args.ncm):
            acc_matr_ncm = info_matr['acc_matr_ncm']
            acc_matr_network = info_matr['acc_matr_network']
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
            print("Expanding class for resuming : %d, %d" %(mdl_cl, gt_cl))
            test_set.expand([mdl_cl], [gt_cl])

        # Ensuring requires_grad = True after model reload
        for p in model.parameters():
            p.requires_grad = True
    else:
        print("resuming model from %s-model.pth.tar" %
              os.path.splitext(args.outfile)[0])

        if args.resume:
            model.from_resnet("%s-model.pth.tar" % os.path.splitext(args.outfile)[0])

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

        # Ensuring requires_grad = True after model reload
        for p in model.parameters():
            p.requires_grad = True


def train_run(device):
    global train_set
    model.cuda(device=device)

    train_wait_time = 0
    s = len(classes_seen)
    print("####### Train Process Running ########")
    print("Args: ", args)
    train_wait_time = 0

    while s < args.num_iters:
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

            if c not in model.classes_map:
                model.increment_classes([c])
                model.cuda(device=device)
                curr_expanded += [c]

        model_curr_class_idx = [model.classes_map[c] for c in curr_class]
        model_classes_seen.append(model_curr_class_idx)

        # Load Datasets
        print("Loading training examples for"\
              " class indexes (%s), (%s), at iteration %d" %
              (', '.join(map(lambda x: str(x), model_curr_class_idx)), ', '.join(map(lambda x: str(x), curr_class)), s))
        train_set.load_data_class(curr_class, model_curr_class_idx, s)

        model.train()

        model.update_representation_icarl(train_set, \
                                          prev_model, \
                                          model_curr_class_idx, \
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


        cond_var.acquire()
        train_counter.value += 1
        if len(curr_expanded) > 0:
            expanded_classes[s % args.test_freq] = model_curr_class_idx, curr_class
        else:
            expanded_classes[s % args.test_freq] = None

        if train_counter.value == test_counter.value + args.test_freq:
            temp_model = copy.deepcopy(model)
            temp_model.cpu()
            dataQueue.put(temp_model)
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
    print("####### Test Process Running ########")
    test_model = None
    s = args.test_freq * (len(classes_seen)//args.test_freq)

    test_wait_time = 0
    with open(args.outfile, "w") as file:
        if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
            print("Model classes, Test Accuracy", file=file)
        elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
            print("Model classes, Test Accuracy NCM, Test Accuracy Network", file=file)
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
            test_model = dataQueue.get()
            expanded_classes_copy = copy.deepcopy(expanded_classes)
            test_counter.value += args.test_freq
            cond_var.notify_all()
            cond_var.release()


    
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
            test_loader = torch.utils.data.DataLoader(test_set, 
                batch_size=args.batch_size_test, shuffle=False, \
                num_workers=0 if args.debug else args.num_workers, pin_memory=True)

            print("%d, " % test_model.n_known, end="", file=file)

            print("[Test Process] Computing Accuracy matrix...")
            # TODO make sure computing and storing accuracies currectly during multi-class per exposure
            all_labels = []
            if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                
                all_preds = []
            elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                all_preds_ncm = []
                all_preds_network = []
            with torch.no_grad():
                for indices, images, labels in test_loader:
                    images = Variable(images).cuda(device=device)

                    if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                        preds = test_model.classify(images, 
                                                    mean_image, 
                                                    args.img_size)
                        all_preds.append(preds.data.cpu().numpy())
                    elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                        preds_ncm, preds_network = \
                                test_model.classify(images, 
                                                mean_image, 
                                                args.img_size)
                        all_preds_ncm.append(preds_ncm.data.cpu().numpy())
                        all_preds_network.append(preds_network.data.cpu().numpy())
                    all_labels.append(labels.numpy())
            if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                    
                all_preds = np.concatenate(all_preds, axis=0)
            elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                all_preds_ncm = np.concatenate(all_preds_ncm, axis=0)
                all_preds_network = np.concatenate(all_preds_network, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            for i in range(test_model.n_known):
                if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                    class_preds = all_preds[all_labels == i]
                    correct = np.sum(class_preds == i)
                    total = len(class_preds)
                elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                    class_preds_ncm = all_preds_ncm[all_labels == i]
                    class_preds_network = \
                        all_preds_network[all_labels == i]
                    correct_ncm = np.sum(class_preds_ncm == i)
                    correct_network = np.sum(class_preds_network == i)
                    total = len(class_preds_ncm)
                    




                if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                    acc_matr[i, s] = (100.0 * correct/total)
                elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                    acc_matr_ncm[i, s] = (100.0 * correct_ncm/total)
                    acc_matr_network[i, s] = (100.0 * correct_network/total)

            if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                test_acc = np.mean(acc_matr[:test_model.n_known, s])
                print('%.2f ,' % test_acc, file=file)
                print('[Test Process] =======> Test Accuracy after %d'
                  ' learning exposures : ' %
                  (s + args.test_freq), test_acc)
            elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                test_acc_ncm = np.mean(acc_matr_ncm[:test_model.n_known, s])
                test_acc_network = np.mean(acc_matr_network[:test_model.n_known, s])

                print('%.2f ,' % test_acc_ncm, file=file)
                print('%.2f ,' % test_acc_network, file=file)

                print('[Test Process] =======> Test Accuracy for NCM after %d'
                      ' learning exposures : ' %
                      (s + args.test_freq), test_acc_ncm)
                print('[Test Process] =======> Test Accuracy for network output after %d'
                      ' learning exposures : ' %
                      (s + args.test_freq), test_acc_network)

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

            # loop var increment
            s += args.test_freq

            if (args.algo == "icarl" and not args.network) \
                        or (args.algo == "e2e" and not args.ncm):
                np.savez('%s-matr.npz' % os.path.splitext(args.outfile)[0], 
                     acc_matr=acc_matr, 
                     model_hyper_params=model.fetch_hyper_params(), 
                     args=args, num_iters_done=s)
            elif (args.algo == "icarl" and args.network) \
                        or (args.algo == "e2e" and args.ncm):
                np.savez('%s-matr.npz' % os.path.splitext(args.outfile)[0], 
                     acc_matr_ncm=acc_matr_ncm,
                     acc_matr_network=acc_matr_network, 
                     model_hyper_params=model.fetch_hyper_params(), 
                     args=args, num_iters_done=s)

            file.flush()
            
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
    print("Train Process Completed")
    test_process.join()
    print("Test Process Completed")


if __name__ == "__main__":
    main()