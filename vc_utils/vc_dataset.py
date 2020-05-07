from os.path import exists
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm.auto import tqdm
from dataset_batch_cifar import CIFAR20
from dataset_batch_imagenet import ImageNet
from vc_utils.activation_threshold import ConvActivationThreshold
from vc_utils.activation_tracker import ActivationTracker
from vc_utils.prune import Pruner
from vc_utils.hook_utils import CustomContext, HookManager

DATASETS = {'ImageNet': ImageNet, 'CIFAR': CIFAR20}


def set_base_dataset(dataset_name):
    global VisualConceptDataset
    assert dataset_name in DATASETS, 'Dataset %s is not a valid base class for VisualConceptDataset. You must choose ' \
                                     'one of %s .' % (dataset_name, ', '.join(DATASETS))
    dataset = DATASETS[dataset_name]
    VisualConceptDataset = _define_VisualConceptDataset(dataset)


def get_vc_dataset(args, model, layer_name, *dset_args, device=0, **dset_kwargs):
    vc_dset = VisualConceptDataset(args, model, layer_name, *dset_args, device=device, **dset_kwargs)

    return vc_dset


def train_classification_layer(args, network, module_name, vc_dset, device=0, act_tracker=None):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    in_dim = act_tracker.modules[module_name].out_channels
    classification_layer = VCLogitLayer(in_dim, vc_dset.kept_idxs).to(device)
    bce = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(classification_layer.parameters(), lr=args.lr_vc)
    for i, img, valid, vc_lbl in tqdm(vc_dset.get_loader(args.batch_size_train_vc)):
        img, vc_lbl = img.to(device), vc_lbl.to(device)
        with act_tracker.track_all_context():
            network(img)
            activations = act_tracker.get_module_activations(module_name, cpu=False)
        out = classification_layer(activations)
        loss = bce(out[valid], vc_lbl[valid])  # only use valid localities in framing loss
        loss.backward()
        optim.step()
        optim.zero_grad()

    return classification_layer


def test_vc_accuracy(args, network, module_name, vc_dset, device=0, classification_layer=None, act_tracker=None):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    if classification_layer is None:
        classification_layer = train_classification_layer(args, network, module_name, vc_dset,
                                                          device=device, act_tracker=act_tracker)

    total = np.zeros((vc_dset.total_classes,))
    correct = np.zeros((vc_dset.total_classes,))
    for i, img, valid, vc_lbl in tqdm(vc_dset.get_loader(args.batch_size_test_vc)):
        img, vc_lbl = img.to(device), vc_lbl.to(device)
        with act_tracker.track_all_context():
            network(img)
            activations = act_tracker.get_module_activations(module_name, cpu=False)
        out = classification_layer(activations)
        pred = torch.sigmoid(out).round()
        pred[~valid] = -1.  # only consider correct predictions among valid localities
        correct_mask = (pred == vc_lbl).transpose(1, 0).flatten(start_dim=1, end_dim=3)
        correct += correct_mask.sum(dim=1).cpu().numpy()

        # Get total number of valid data points for each vc class
        total += valid.transpose(1, 0).flatten(start_dim=1, end_dim=3).sum(dim=1).cpu().numpy()

    return correct / total * 100., classification_layer.conv1x1.weight.data.cpu()[:, :, 0, 0]


def _define_VisualConceptDataset(base_dataset):
    class VisualConceptDataset(base_dataset):

        def __init__(self, args, network, module_name, *dset_args, batch_size=100, device=0, store_on_gpu=False,
                     **dset_kwargs):
            super(VisualConceptDataset, self).__init__(*dset_args, **dset_kwargs)
            self.labeler = VCLabeler(args, network, module_name, device=device, store_on_gpu=store_on_gpu)
            self.valid_localities = None
            if hasattr(self, 'train_labels'):
                self.class_train_labels = self.train_labels
                self._prune_mask, self.train_labels, self.valid_localities = \
                    self.labeler.label_data(self.get_loader(batch_size=batch_size))
                self.total_classes = self.train_labels.shape[1]
            if hasattr(self, 'test_labels'):
                self.class_test_labels = self.test_labels
                self._prune_mask, self.test_labels, self.valid_localities = \
                    self.labeler.label_data(self.get_loader(batch_size=batch_size))
                self.total_classes = self.test_labels.shape[1]

        def __getitem__(self, item):
            i, img, lbl = super(VisualConceptDataset, self).__getitem__(item)
            if self.valid_localities is not None:
                return i, img, self.valid_localities[item], lbl
            return i, img, lbl

        def get_loader(self, batch_size=100):
            return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False)

        @property
        def kept_idxs(self):
            return np.where(~self._prune_mask)[0]

        @property
        def pruned_idxs(self):
            return np.where(self._prune_mask)[0]

    return VisualConceptDataset


class NullVisualConceptDataset:

    def __init__(self, *args, **kwargs):
        raise AssertionError('Base dataset for VisualConceptDataset has not been set. Call set_base_dataset before'
                             ' initializing objects of class VisualConceptDataset')


class VCLogitLayer(torch.nn.Module):

    def __init__(self, in_dim, vc_idxs, threshold_estimate=1.0):
        super(VCLogitLayer, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=len(vc_idxs), kernel_size=1)

        # initialize layer weights with best guess under assumption that activations never change
        self.conv1x1.weight.data.zero_()
        for i, idx in enumerate(vc_idxs):
            self.conv1x1.weight.data[i, idx, :, :] = 1.0
        self.conv1x1.bias.data[:] = -threshold_estimate

    def forward(self, x):
        return self.conv1x1(x)


class VCLabeler:

    def __init__(self, args, network, module_name, store_on_gpu=False, device=0):
        self.layer_name = module_name
        self.min_data_points = args.vc_dataset_size
        self.absent_threshold = args.absent_vc_threshold
        self.present_threshold = args.present_vc_threshold
        self.save_prune_mask = args.save_prune_mask
        self.load_prune_mask = args.load_prune_mask
        self.device = device
        self.hook_manager = HookManager()
        self.network = network
        self.load_pruned_path = args.load_pruned_path if exists(args.load_pruned_path) and args.load_prune_mask \
            else None
        self.thresholder = ConvActivationThreshold(network, module_name,
                                                   store_on_gpu=store_on_gpu,
                                                   hook_manager=self.hook_manager)
        self.pruner = Pruner(network, module_name,
                             prune_ratio=args.prune_ratio,
                             load_pruned_path=self.load_pruned_path,
                             store_on_gpu=store_on_gpu,
                             save_pruned_path=args.save_pruned_path,
                             hook_manager=self.hook_manager)
        self.act_tracker = ActivationTracker(module_names=[module_name], network=network,
                                             store_on_gpu=store_on_gpu,
                                             hook_manager=self.hook_manager)

    @staticmethod
    def get_balanced_data_mask(present, absent, min_data_points):
        num_filters = present.shape[1]
        min_per_bin_class = min_data_points // 2

        p_samples = np.stack(np.where(present), axis=1)
        p_filter_samples = [p_samples[p_samples[:, 1] == i] for i in range(num_filters)]

        n_samples = np.stack(np.where(absent), axis=1)
        n_filter_samples = [n_samples[n_samples[:, 1] == i] for i in range(num_filters)]

        discard_filters = {i for i in range(num_filters) if
                           n_filter_samples[i].shape[0] < min_per_bin_class or
                           p_filter_samples[i].shape[0] < min_per_bin_class}

        # Randomly sample min_per_bin_class samples for each binary class of each visual concept and arrange by vc
        kept_filter_samples = [
            (
                i,
                np.concatenate([p_s[np.random.choice(p_s.shape[0], min_per_bin_class, replace=False)],
                                n_s[np.random.choice(n_s.shape[0], min_per_bin_class, replace=False)]], axis=0)
            )
            for i, (p_s, n_s) in enumerate(zip(p_filter_samples, n_filter_samples))
            if i not in discard_filters
        ]

        assert len(kept_filter_samples) > 0, 'All filters were removed. Change threhsolds or et min_data_points lower'

        kept_filter_idxs, kept_sample_idxs = zip(*kept_filter_samples)
        kept_sample_idxs = tuple(np.concatenate(kept_sample_idxs, axis=0).transpose(1, 0))

        select_samples = torch.zeros_like(present).type(torch.bool)
        select_samples[kept_sample_idxs] = True

        return present[:, kept_filter_idxs], select_samples[:, kept_filter_idxs], discard_filters

    def label_data(self, loader, seed=0):
        rand_state = np.random.get_state()
        np.random.seed(seed)

        self.network.to(self.device)

        activations = self.act_tracker.compute_activations_from_data(loader, device=self.device)
        if not self.load_prune_mask or not self.load_pruned_path:
            self.pruner.compute_prune_mask_from_activations(activations, save=self.save_prune_mask)

        pruned = self.pruner.prune(self.layer_name, activations[self.layer_name], keep_shape=False)

        absent = torch.zeros_like(pruned)
        absent[pruned < self.absent_threshold] = 1.0

        present = torch.zeros_like(pruned)
        present[pruned >= self.present_threshold] = 1.0

        labels, select_data, discard_filters = self.get_balanced_data_mask(present, absent, self.min_data_points)
        print('VCLabeler: Discarding samples of visual concepts corresponding to %d filters due to unbalanced '
              'concept presence/absence: %s\nTo avoid this, try different thresholds'
              % (len(discard_filters), ', '.join(str(f) for f in discard_filters)))

        np.random.set_state(rand_state)

        # Prune filters whose visual concepts were excluded from dataset due to imbalance
        if len(discard_filters) > 0:
            (kept_filters,) = np.where(~self.pruner.prune_mask[self.layer_name])
            self.pruner.prune_mask[self.layer_name][kept_filters[np.array(list(discard_filters))]] = True

        return self.pruner.prune_mask[self.layer_name], labels, select_data


VisualConceptDataset = NullVisualConceptDataset
