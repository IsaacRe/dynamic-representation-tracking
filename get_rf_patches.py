from torchvision.models import resnet34
import torchvision.transforms as transforms
from dataset_batch_cifar import CIFAR20
from model import IncrNet
import torch
import argparse
import pickle
import numpy as np
from feature_vis import PatchAggregator


parser = argparse.ArgumentParser()
parser.add_argument('--num_patch', type=int, default=400, help='Number of total patches to collect')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size to use for rf evaluation')
parser.add_argument('--layer_name', type=str, default='layer1.2.conv2', help='Layer for rf evaluation')
parser.add_argument('--total_classes', type=int, default=20, help='Total number of classes to use')
parser.add_argument('--pt', type=str, default='', help='Specify file to load model from, default: use pytorch pt model')
parser.add_argument('--debug', action='store_true', help='Restrict parallel workers for data loading during debugging')
args = parser.parse_args()


model = resnet34(pretrained=True)
if args.pt:
    with open('args.pkl', 'rb') as f:
        dummy_args = pickle.load(f)
    model = IncrNet(dummy_args, 0)
    model.from_resnet(args.pt)
    model = model.model
else:
    feat_size = model.fc.in_features
    model.fc = torch.nn.Linear(feat_size, args.total_classes, bias=False)

mean_image = np.load("data_generator/cifar_mean_image.npy")

test_set = CIFAR20(range(args.total_classes),
                   root="./data",
                   train=False,
                   download=True,
                   transform=None,
                   mean_image=mean_image)

loader = torch.utils.data.DataLoader(test_set,
                                     num_workers=0 if args.debug else args.num_workers,
                                     batch_size=args.batch_size)

model.cuda()
rf_agg = PatchAggregator(model, args.layer_name, loader, args.num_patch)

rf_agg.aggregate_patches()