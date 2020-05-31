import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR100
from torch.utils.data.dataset import Subset
import cv2
from utils.model_utils import kaiming_normal_init
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
from tqdm import tqdm
from feature_matching import match
from dataset_batch_cifar import CIFAR20
from model import IncrNet
import pdb

from dataset_incr_cifar import iCIFAR10, iCIFAR100

parser = argparse.ArgumentParser(description="Incremental learning")

# Test options
parser.add_argument("--confusion_matrix", action='store_true', help="Compute confusion matrix for loaded model")

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

# Hyperparameters
parser.add_argument("--init_lr", default=0.002, type=float,
                    help="initial learning rate")
parser.add_argument("--num_epoch", default=5, type=int,
                    help="Number of epochs")
parser.add_argument("--lrd", default=5, type=float,
                    help="Learning rate decrease factor")
parser.add_argument("--llr_freq", default=10, type=int,
                    help="Learning rate lowering frequency for SGD (for E2E)")
parser.add_argument("--wd", default=0.00001, type=float,
                    help="Weight decay for SGD")
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--batch_size", default=200, type=int,
                    help="Mini batch size for training")
parser.add_argument("--batch_size_test", default=200, type=int,
                    help="Mini batch size for testing")

# Model options
parser.add_argument("--no_pt", dest="pretrained", action="store_false",
                    help="Option to start from an ImageNet pretrained model")
parser.add_argument("--ncm", dest="ncm", action="store_true",
                    help="Use nearest class mean classification (for E2E)")
parser.add_argument("--img_size", default=224, type=int,
                    help="Size of images input to the network")
parser.add_argument("--rendered_img_size", default=300, type=int,
                    help="Size of rendered images")
parser.add_argument("--total_classes", default=20, type=int,
                    help="Total number of classes")

# Training Options
parser.add_argument("--classes_from_coverage", action='store_true',
                    help="Get cifar classes subset from 'classes_seen' in the provided coverage file")
parser.add_argument("--batch_exemplar", action='store_true',
                    help="Train on a set of exemplars from a previous incremental training session")
parser.add_argument("--coverage_file", type=str,
                    help="Location of the npz containing 'exemplar_data' file - for batch exemplar training")
parser.add_argument("--no_jitter", dest="jitter", action="store_false",
                    help="Option for no color jittering (for iCaRL)")
parser.add_argument("--h_ch", default=0.02, type=float,
                    help="Color jittering : max hue change")
parser.add_argument("--s_ch", default=0.05, type=float,
                    help="Color jittering : max saturation change")
parser.add_argument("--l_ch", default=0.1, type=float,
                    help="Color jittering : max lightness change")

# Correlation Analysis/Feature Tracking Options
parser.add_argument('--feat_vis_layer_name', nargs='+', type=str, default=['layer2.0.conv1'],
                    help='Specify layer for feature visualziation/tracking')
parser.add_argument('--feat_corr', action='store_true', help='Conduct correlation analysis')
parser.add_argument('--save_matr', action='store_true', help='Save correlation matrix during each iteration')
parser.add_argument('--batch_size_corr', type=int, default=30, help='Batch size used for correlation computation')
parser.add_argument('--corr_model_batch', type=str, default='batch_model/batch_model-cifar20-model.pth.tar',
                    help='Specify batch trained model for correlation')

# System options
parser.add_argument("--test_freq", default=1, type=int,
                    help="Number of iterations of training after"
                         " which a test is done/model saved")
parser.add_argument("--num_workers", default=8, type=int,
                    help="Maximum number of threads spawned at any" 
                         "stage of execution")
parser.add_argument("--one_gpu", dest="one_gpu", action="store_true",
                    help="Option to run multiprocessing on 1 GPU")
parser.add_argument("--debug", action='store_true')
parser.add_argument('--seed', type=int, default=1, help='Set torch and numpy seed')

parser.set_defaults(ncm=False)
parser.set_defaults(dist=False)
parser.set_defaults(pretrained=True)
parser.set_defaults(jitter=True)
parser.set_defaults(save_all=False)
parser.set_defaults(resume=False)
parser.set_defaults(one_gpu=False)


# Print help if no arguments passed
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
mp.set_sharing_strategy("file_system")
expt_githash = subprocess.check_output(["git", "describe", "--always"])

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

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

test_freq = 1
total_classes = args.total_classes

if not os.path.exists("data_generator/cifar_mean_image.npy"):
    mean_image = None
else:
    mean_image = np.load("data_generator/cifar_mean_image.npy")


def build_model():
    model = models.resnet34(pretrained=args.pretrained)
    if not args.pretrained:
        model.apply(kaiming_normal_init)
    feat_size = model.fc.in_features
    model.fc = torch.nn.Linear(feat_size, args.total_classes, bias=False)
    return model


model = build_model()
if args.feat_corr:
    import pickle
    with open('args.pkl', 'rb') as f:
        load_args = pickle.load(f)
    corr_model = IncrNet(load_args, device=test_device, cifar=True)
    corr_model.from_resnet(args.corr_model_batch)
    corr_model = corr_model.model

assert mean_image is not None
transform_jitter = transforms.ColorJitter(hue=args.h_ch, saturation=args.s_ch, brightness=args.l_ch)
transform_resize = transforms.Resize(args.img_size)
# Use same settings as in icarl augmentation implemenation in iCifar10
transform_crop = transforms.RandomCrop(args.img_size, padding=4)


def norm(img):
    # Resize and transpose to CxWxH all train images
    #resized_train_images = np.zeros((len(train_data),
    #                                 args.img_size, args.img_size, 3), dtype=np.uint8)
    # normalize by mean
    return img - torch.Tensor(mean_image / 255.)


train_transform = transforms.Compose([transform_jitter, transform_resize, transform_crop, transforms.ToTensor(), norm])
test_transform = transforms.Compose([transform_resize, transforms.ToTensor(), norm])

if args.classes_from_coverage:
    classes = np.load(args.coverage_file)['classes_seen']
    classes_ = []
    for i in range(len(classes)):
        if classes[i] not in classes_:
            classes_ += [classes[i]]
    classes = classes_
else:
    classes = list(np.random.choice(100, total_classes, replace=False))
classes_map = {c: i for c, i in zip(classes, range(total_classes))}

train_set = CIFAR100(root="./data",
                     train=True,
                     download=True,
                     transform=train_transform)

test_set = CIFAR100(root="./data",
                     train=False,
                     download=True,
                     transform=test_transform)

corr_set = CIFAR20(classes,
                   root='./data',
                   train=False,
                   download=True,
                   transform=None,
                   mean_image=mean_image)

num_epoch = args.num_epoch

# Load exemplars to use as training data
if args.batch_exemplar:
    classes_seen = np.load(args.coverage_file)['classes_seen']
    classes = []
    for c in classes_seen:
        if c in classes:
            continue
        classes += [c]
    classes_map = {c: i for c, i in zip(classes, range(total_classes))}
    explr_data = np.load(args.coverage_file)['exemplar_data']
    final_explr = explr_data[-1]
    # get tensor of exposure number, sample index for every sample across exemplar sets
    train_data = np.concatenate(final_explr, axis=0)
    # pass indexes to sample from Cifar-100
    train_set = Subset(train_set, train_data[:, 1])
    train_labels = train_set.dataset.train_labels
    # iterate through each sample across exemplar sets
    for exposure, idx in train_data:
        # get cifar-100 class id for each of the sampled classes
        class_id = classes_seen[exposure]
        # set current label to corresponding index in Cifar-20 sampled classes
        train_labels[idx] = classes_map[class_id]
    test_set = Subset(test_set, np.where(np.isin(test_set.test_labels, classes))[0])
    test_labels = test_set.dataset.test_labels
    for i in range(len(test_labels)):
        if test_labels[i] not in classes_map:
            continue
        test_labels[i] = classes_map[test_labels[i]]
else:
    classes_map = {c: i for c, i in zip(classes, range(total_classes))}
    train_set = Subset(train_set, np.where(np.isin(train_set.train_labels, classes))[0])
    train_labels = train_set.dataset.train_labels
    # iterate through each sample across exemplar sets
    for i in range(len(train_labels)):
        if train_labels[i] not in classes_map:
            continue
        train_labels[i] = classes_map[train_labels[i]]
    test_set = Subset(test_set, np.where(np.isin(test_set.test_labels, classes))[0])
    test_labels = test_set.dataset.test_labels
    for i in range(len(test_labels)):
        if test_labels[i] not in classes_map:
            continue
        test_labels[i] = classes_map[test_labels[i]]

train_dl = torch.utils.data.DataLoader(train_set,
                                       num_workers=0 if args.debug else args.num_workers,
                                       batch_size=args.batch_size)
test_dl = torch.utils.data.DataLoader(test_set,
                                      num_workers=0 if args.debug else args.num_workers,
                                      batch_size=args.batch_size_test)
if args.feat_corr:
    correlation_loader = torch.utils.data.DataLoader(corr_set, shuffle=False,
                                                     num_workers=0 if args.debug else args.num_workers,
                                                     batch_size=args.batch_size_corr)

acc_matr = np.zeros((int(total_classes), num_epoch))
coverage = np.zeros((int(total_classes), num_epoch))

# Conditional variable, shared memory for synchronization
cond_var = mp.Condition()
train_counter = mp.Value("i", 0)
test_counter = mp.Value("i", 0)
dataQueue = mp.Queue()
all_done = mp.Event()
data_mgr = mp.Manager()
expanded_classes = data_mgr.list([None for i in range(args.test_freq)])

start_epoch = 0
num_epoch = args.num_epoch
old_outfile = args.outfile

if args.resume:
    print("resuming model from %s-model.pth.tar" %
          os.path.splitext(args.outfile)[0])

    model = torch.load("%s-model.pth.tar" % os.path.splitext(args.outfile)[0],
                       map_location=lambda storage, loc: storage)
    model.device = train_device

    if not args.confusion_matrix:
        info_coverage = np.load("%s-coverage.npz" \
                                % os.path.splitext(args.outfile)[0])
    info_matr = np.load("%s-matr.npz" % os.path.splitext(args.outfile)[0])
    if not args.confusion_matrix and "expt_githash" in info_coverage.files and expt_githash != info_coverage["expt_githash"]:
        print("Warning : Code was changed since the last time model was saved")
        print("Last commit hash : ", info_coverage["expt_githash"])
        print("Current commit hash : ", expt_githash)

    args_resume_outfile = args.resume_outfile
    num_iters_done = acc_matr.shape[1]
    acc_matr = info_matr["acc_matr"]
    for k, v in info_matr['args'].item().__dict__.items():
        args.__dict__[k] = v
    args.num_epoch = num_epoch
    start_epoch = num_iters_done

    if args_resume_outfile is not None:
        args.outfile = args.resume_outfile = args_resume_outfile
    else:
        print("Overwriting old files")

    if not args.confusion_matrix:
        coverage = info_coverage["coverage"]
        train_set.all_train_coverage = info_coverage["train_coverage"]

    # Ensuring requires_grad = True after model reload
    for p in model.parameters():
        p.requires_grad = True


def train_run(device):
    #global train_dl
    model.cuda(device=device)
    print("####### Train Process Running ########")
    print("Args: ", args)
    train_wait_time = 0
    lr = args.init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)

    criterion = torch.nn.CrossEntropyLoss()
    num_batches_per_epoch = len(train_dl)

    model.train()

    with tqdm(total=num_batches_per_epoch*num_epoch) as pbar:
        for epoch in range(start_epoch, num_epoch):

            if (epoch+1) % args.llr_freq == 0:
                tqdm.write('Lowering Learning rate at epoch %d' %
                           (epoch+1))
                lr = lr * 1.0/args.lrd
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            for i, (images, labels) in enumerate(train_dl):


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


                images = Variable(images).cuda(device=device)
                labels = Variable(labels).cuda(device=device)

                optimizer.zero_grad()

                logits = model.forward(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                tqdm.write('Epoch [%d/%d], Minibatch [%d/%d] Loss: %.4f'
                           % ((epoch+1), num_epoch, i % num_batches_per_epoch+1,
                              num_batches_per_epoch, loss.data))

                pbar.update(1)

                cond_var.acquire()
                train_counter.value += 1

                if train_counter.value == test_counter.value + args.test_freq:
                    temp_model = copy.deepcopy(model)
                    temp_model.cpu()
                    dataQueue.put(temp_model)
                cond_var.notify_all()
                cond_var.release()

    time_ptr = time.time()
    all_done.wait()
    train_wait_time += time.time() - time_ptr
    print("[Train Process] Done, total time spent waiting : ", train_wait_time)


def test_run(device):
    #global test_dl
    print("####### Test Process Running ########")
    test_model = None

    test_wait_time = 0
    with open(args.outfile, "w") as file:
        print("Model classes, Test Accuracy", file=file)
        for epoch in range(start_epoch, num_epoch):

            for itr in range(0, len(train_dl), args.test_freq):

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

                print("[Test Process] Test Set Length:", len(test_dl.dataset.indices))

                test_model.cuda(device=device)
                test_model.eval()

                print("[Test Process] Computing Accuracy matrix...")
                all_labels = []
                all_preds = []
                with torch.no_grad():
                    for images, labels in test_dl:
                        images = Variable(images).cuda(device=device)
                        logits = test_model.forward(images)
                        _, preds = logits.max(dim=1)
                        all_preds.append(preds.data.cpu().numpy())
                        all_labels.append(labels.numpy())
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                for i in range(total_classes):
                    class_preds = all_preds[all_labels == i]
                    correct = np.sum(class_preds == i)
                    total = len(class_preds)
                    acc_matr[i, epoch] = (100.0 * correct/total)

                test_acc = np.mean(acc_matr[:, epoch])
                print("%.2f ," % test_acc, file=file)
                print("[Test Process] =======> Test Accuracy after %d"
                      " learning exposures : " %
                      (epoch + args.test_freq), test_acc)

                if args.feat_corr:
                    print("[Test Process] Computing Correlation Matrix...")
                    corr_model.cuda(device)
                    matches, corr, matrix = match(device, correlation_loader, test_model, corr_model,
                                                  args.feat_vis_layer_name[0], replace=False)

                    if args.save_matr:
                        matrix_dir = args.outfile.split('.')[0] + '-corr_matrix/'
                        try:
                            os.mkdir(matrix_dir)
                        except FileExistsError:
                            pass
                        np.save(matrix_dir + '%s-matr-%d.npy' % (args.outfile.split('/')[-1].split('.')[0], epoch), matrix)

                print("[Test Process] Saving model and other data")
                test_model.cpu()
                test_model.num_iters_done = epoch
                if not args.save_all:
                    torch.save(test_model, "%s-model.pth.tar" %
                               os.path.splitext(args.outfile)[0])
                else:
                    torch.save(test_model, "%s/model_epoch_%d_iter_%d.pth.tar"
                                            %(args.save_all_dir, epoch, itr))

                np.savez(args.outfile[:-4] + "-matr.npz", acc_matr=acc_matr,
                         githash=expt_githash, args=args, num_iter_done=epoch)

        print("[Test Process] Done, total time spent waiting : ",
              test_wait_time)
        all_done.set()


def confusion_matrix(device):
    # Wait till training is done
    """
    time_ptr = time.time()
    test_wait_time = 0
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
    test_counter.value += args.test_freq
    cond_var.notify_all()
    cond_var.release()
    """
    test_model = model

    print("[Test Process] Test Set Length:", len(test_dl.dataset.indices))

    test_model.cuda(device=device)
    test_model.eval()

    print("[Test Process] Computing Accuracy matrix...")
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_dl:
            images = Variable(images).cuda(device=device)
            logits = test_model.forward(images)
            _, preds = logits.max(dim=1)
            all_preds.append(preds.data.cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    confusion_matrix = np.ndarray((len(classes), len(classes)))
    for c0 in classes:
        for c1 in classes:
            curr_indices = np.where(all_labels == c0)
            confusion_matrix[c0, c1] = np.where(all_preds[curr_indices] == c1)[0].shape[0] / curr_indices[0].shape[0]
    np.save(os.path.splitext(old_outfile)[0] + '-conf_matr.npy', confusion_matrix)


def cleanup(train_process, test_process):
    train_process.terminate()
    test_process.terminate()


def main():
    if args.confusion_matrix:
        confusion_matrix(train_device)
        return
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
