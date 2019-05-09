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
parser.add_argument("--pt", dest="pretrained", action="store_true",
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

assert mean_image is not None
transform_jitter = transforms.ColorJitter(hue=args.h_ch, saturation=args.s_ch, brightness=args.l_ch)
transform_resize = transforms.Resize(args.img_size)

def resize_transpose_and_norm(train_data):
    # Resize and transpose to CxWxH all train images
    resized_train_images = np.zeros((len(train_data),
                                     args.img_size, args.img_size, 3), dtype=np.uint8)
    for i, train_image in enumerate(train_data):
        resized_train_images[i] = cv2.resize(train_image,
                                             (args.img_size, args.img_size))
    # normalize by mean
    #resized_train_images = (resized_train_images - mean_image.transpose(2,1,0)) / 255.
    return resized_train_images

classes = list(np.random.choice(100, total_classes, replace=False))
classes_map = {c: i for c, i in zip(classes, range(total_classes))}

train_set = CIFAR100(root="./data",
                     train=True,
                     download=True,
                     transform=transforms.Compose([transform_jitter, transform_resize, transforms.ToTensor()]))

test_set = CIFAR100(root="./data",
                     train=False,
                     download=True,
                     transform=transforms.Compose([transform_resize, transforms.ToTensor()]))

num_epoch = args.num_epoch

# Load exemplars to use as training data
if args.batch_exemplar:
    classes_seen = np.load(args.coverage_file)['classes_seen']
    classes = list(set(classes_seen))
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
    pass  # fix train set to total_classes

train_dl = torch.utils.data.DataLoader(train_set,
                                       num_workers=0 if args.debug else args.num_workers,
                                       batch_size=args.batch_size)
test_dl = torch.utils.data.DataLoader(test_set,
                                      num_workers=0 if args.debug else args.num_workers,
                                      batch_size=args.batch_size_test)

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

if args.resume:
    print("resuming model from %s-model.pth.tar" %
          os.path.splitext(args.outfile)[0])

    model = torch.load("%s-model.pth.tar" % os.path.splitext(args.outfile)[0],
                       map_location=lambda storage, loc: storage)
    model.device = train_device

    info_coverage = np.load("%s-coverage.npz" \
                            % os.path.splitext(args.outfile)[0])
    info_matr = np.load("%s-matr.npz" % os.path.splitext(args.outfile)[0])
    if expt_githash != info_coverage["expt_githash"]:
        print("Warning : Code was changed since the last time model was saved")
        print("Last commit hash : ", info_coverage["expt_githash"])
        print("Current commit hash : ", expt_githash)

    args_resume_outfile = args.resume_outfile
    num_iters_done = acc_matr.shape[1]
    acc_matr = info_matr["acc_matr"]
    args = info_matr["args"].item()
    args.num_epoch = num_epoch
    start_epoch = num_iters_done

    if args_resume_outfile is not None:
        args.outfile = args.resume_outfile = args_resume_outfile
    else:
        print("Overwriting old files")

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
            if (epoch+1) % args.llr_freq == 0:
                tqdm.write('Lowering Learning rate at epoch %d' %
                           (epoch+1))
                lr = lr * 1.0/args.lrd
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            for i, (images, labels) in enumerate(train_dl):

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

            print("[Test Process] Saving model and other data")
            test_model.cpu()
            test_model.num_iters_done = epoch
            if not args.save_all:
                torch.save(test_model, "%s-model.pth.tar" %
                           os.path.splitext(args.outfile)[0])
            else:
                torch.save(test_model, "%s-saved_models/model_iter_%d.pth.tar"
                                        %(os.path.join(args.save_all_dir,
                                        os.path.splitext(args.outfile)[0]), epoch))

            np.savez(args.outfile[:-4] + "-matr.npz", acc_matr=acc_matr,
                     githash=expt_githash, args=args, num_iter_done=epoch)

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
