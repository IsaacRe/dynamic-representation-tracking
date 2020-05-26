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
from vc_utils.dynamic_threshold import ThresholdLearner

DATASETS = {'ImageNet': ImageNet, 'CIFAR': CIFAR20}


def set_base_dataset(dataset_name):
    global VisualConceptDataset
    assert dataset_name in DATASETS, 'Dataset %s is not a valid base class for VisualConceptDataset. You must choose ' \
                                     'one of %s .' % (dataset_name, ', '.join(DATASETS))
    dataset = DATASETS[dataset_name]
    VisualConceptDataset = _define_VisualConceptDataset(dataset)


def get_vc_dataset(args, model, layer_name, *dset_args, device=0, balance=False, **dset_kwargs):
    vc_dset = VisualConceptDataset(args, model, layer_name, *dset_args, device=device, balance=balance, **dset_kwargs)

    return vc_dset


def train_classification_layer_v1(args, network, module_name, vc_dset, device=0, act_tracker=None):
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


def test_vc_accuracy_v1(args, network, module_name, vc_dset, device=0, classification_layer=None, act_tracker=None,
                        recall=False):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    if classification_layer is None:
        #classification_layer = train_classification_layer(args, network, module_name, vc_dset,
        #                                                  device=device, act_tracker=act_tracker)
        in_dim = act_tracker.modules[module_name].out_channels
        classification_layer = VCLogitLayer(in_dim, vc_dset.kept_idxs, args.present_vc_threshold).to(device)

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
        correct_mask = (pred == vc_lbl)

        if recall:
            # Only care about correctly classified instances where the label is 1
            correct += (correct_mask.type(torch.float) * vc_lbl).sum(dim=0).sum(dim=1).sum(dim=1).type(torch.int).cpu().numpy()

            # Get total number of positive, valid data points
            total_ = vc_lbl * valid.type(torch.float).to(device)
            total_ = total_.sum(dim=0).sum(dim=1).sum(dim=1)
            total += total_.type(torch.int).cpu().numpy()
        else:
            correct += correct_mask.sum(dim=0).sum(dim=1).sum(dim=1).cpu().numpy()

            # Get total number of valid data points for each vc class
            total += valid.transpose(1, 0).flatten(start_dim=1, end_dim=3).sum(dim=1).cpu().numpy()

    return correct / total * 100., classification_layer.conv1x1.weight.data.cpu()[:, :, 0, 0]


def train_classification_layer(args, network, module_name, vc_dset, device=0, act_tracker=None,
                               uniform_init=False, epochs=1):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    in_dim = act_tracker.modules[module_name].out_channels
    if not uniform_init:
        classification_layer = VCLogitLayer(in_dim, vc_dset.kept_idxs).to(device)
    else:
        classification_layer = VCLogitLayer(in_dim, vc_dset.kept_idxs, uniform_init=1 / 512).to(device)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    filter_weights = None
    weights = None
    optim = torch.optim.SGD(classification_layer.parameters(), lr=args.lr_vc)
    test_loader = vc_dset.get_loader(args.batch_size_train_vc)
    pbar = tqdm(total=epochs * len(test_loader))
    for e in range(epochs):
        for i, img, vc_lbl in test_loader:
            if filter_weights is None:
                filter_weights = vc_dset.filter_weights[None, :, None, None].repeat(vc_lbl.shape[0], 1,
                                                                                    *vc_lbl.shape[2:]).to(device)
                weights = torch.ones_like(filter_weights).to(device)
            img, vc_lbl = img.to(device), vc_lbl.to(device)
            with act_tracker.track_all_context():
                network(img)
                activations = act_tracker.get_module_activations(module_name, cpu=False)
            out = classification_layer(activations)
            loss = bce(out, vc_lbl)  # only use valid localities in framing loss
            weights[:] = 1.0
            weights[vc_lbl == 1.0] = filter_weights[vc_lbl == 1.0]
            loss = (loss * weights).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.update(1)
    pbar.close()

    return classification_layer


def test_vc_accuracy(args, network, module_name, vc_dset_train, vc_dset, device=0, classification_layer=None,
                     act_tracker=None, uniform_init=False, epochs=1, train=True, recall=False):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    if classification_layer is None:
        if train:
            classification_layer = train_classification_layer(args, network, module_name, vc_dset_train,
                                                          device=device, act_tracker=act_tracker,
                                                          uniform_init=uniform_init, epochs=epochs)
        else:
            classification_layer = VCLogitLayer(512, range(512)).cuda(device)

    total = np.zeros((vc_dset.total_classes,))
    correct = np.zeros((vc_dset.total_classes,))
    for i, img, vc_lbl in tqdm(vc_dset.get_loader(args.batch_size_test_vc)):
        img, vc_lbl = img.to(device), vc_lbl.to(device)
        with act_tracker.track_all_context():
            network(img)
            activations = act_tracker.get_module_activations(module_name, cpu=False)
        out = classification_layer(activations)
        pred = torch.sigmoid(out).round()
        correct_mask = (pred == vc_lbl)
        if recall:
            # Only care about correctly classified instances where the label is 1
            correct += (correct_mask.type(torch.float) * vc_lbl).sum(dim=0).sum(dim=1).sum(dim=1).cpu().numpy()

            # Get total number of positive data points for each vc class
            total += vc_lbl.sum(dim=0).sum(dim=1).sum(dim=1).cpu().numpy()
        else:
            correct += correct_mask.sum(dim=0).sum(dim=1).sum(dim=1).cpu().numpy()

            # Get total number of data points for each vc class
            total += np.array([vc_lbl.transpose(1, 0).flatten(start_dim=1, end_dim=3).shape[1]] * vc_dset.total_classes)

    return correct / total * 100., classification_layer.conv1x1.weight.data.cpu()[:, :, 0, 0]


def _define_VisualConceptDataset(base_dataset):
    class VisualConceptDataset(base_dataset):

        def __init__(self, args, network, module_name, *dset_args, batch_size=100, device=0, store_on_gpu=False,
                     version='2.0', balance=False, **dset_kwargs):
            super(VisualConceptDataset, self).__init__(*dset_args, **dset_kwargs, crop=False)
            self.labeler = VCLabeler(args, network, module_name, device=device, store_on_gpu=store_on_gpu)
            self.valid_localities = None
            self.filter_weights = None
            self._prune_mask = None
            self.sorted_filters = None

            if version == '2.0':
                if hasattr(self, 'train_labels'):
                    labels_attr = 'train_labels'
                else:
                    assert hasattr(self, 'test_labels')
                    labels_attr = 'test_labels'
                setattr(self, 'class_' + labels_attr, getattr(self, labels_attr))
                self._prune_mask, labels, self.valid_localities, self.sorted_filters = \
                    self.labeler.label_data(self.get_loader(batch_size=batch_size, shuffle=True),
                                            self.get_loader(batch_size=batch_size),
                                            order_by_importance=True,
                                            balance=balance,
                                            save_acts=args.save_activations)
                self.total_classes = labels.shape[1]
                setattr(self, labels_attr, labels)

                # get weights for positive samples for data balancing
                ratio_positive = labels.mean(dim=0).mean(dim=1).mean(dim=1)
                self.filter_weights = (1 - ratio_positive) / ratio_positive

            else:
                # old code version 1.0
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

            print('\n\n A total of %d/%d filters discarded\n\n' % (len(self.pruned_idxs),
                                                                   len(self.pruned_idxs) + len(self.kept_idxs)))

        def __getitem__(self, item):
            i, img, lbl = super(VisualConceptDataset, self).__getitem__(item)
            if self.valid_localities is not None:
                return i, img, self.valid_localities[item], lbl
            # if vc data has not been initialized behave as inherited dataset
            return i, img, lbl

        def get_loader(self, batch_size=100, shuffle=False):
            return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)

        @property
        def kept_idxs(self):
            if self._prune_mask is None:
                if self.train:
                    return np.arange(self.train_labels.shape[1])
                return np.arange(self.test_labels.shape[1])
            return np.where(~self._prune_mask)[0]

        @property
        def pruned_idxs(self):
            if self._prune_mask is None:
                return np.array([])
            return np.where(self._prune_mask)[0]

    return VisualConceptDataset


class NullVisualConceptDataset:

    def __init__(self, *args, **kwargs):
        raise AssertionError('Base dataset for VisualConceptDataset has not been set. Call set_base_dataset before'
                             ' initializing objects of class VisualConceptDataset')


class VCLogitLayer(torch.nn.Module):

    def __init__(self, in_dim, vc_idxs, threshold_estimate=0.5, uniform_init=None):
        super(VCLogitLayer, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=len(vc_idxs), kernel_size=1)

        if uniform_init is None:
            # initialize layer weights with best guess under assumption that activations never change
            self.conv1x1.weight.data.zero_()
            for i, idx in enumerate(vc_idxs):
                self.conv1x1.weight.data[i, idx, :, :] = 1.0
        else:
            self.conv1x1.weight.data.fill_(uniform_init)

        self.conv1x1.bias.data[:] = -threshold_estimate

    def forward(self, x):
        return self.conv1x1(x)


class VCLabeler:

    def __init__(self, args, network, module_name, store_on_gpu=False, device=0,
                 init_t=0.5, temperature=0.2, version='2.0'):
        init_t = args.present_vc_threshold
        self.layer_name = module_name
        self.init_t = init_t
        self.min_data_points = args.vc_dataset_size
        self.absent_threshold = args.absent_vc_threshold
        self.present_threshold = args.present_vc_threshold
        self.save_prune_mask = args.save_prune_mask
        self.load_prune_mask = args.load_prune_mask
        self.device = device
        self.hook_manager = HookManager()
        self.network = network
        self.version = version
        self.load_pruned_path = args.load_pruned_path if exists(args.load_pruned_path) and args.load_prune_mask \
            else None

        # dynamic pruning
        if version == '2.0':
            self.dynamic_thresholder = ThresholdLearner(network, module_name, self.hook_manager, device, init_t,
                                                        temperature)
            self.pruner = Pruner(network, module_name,
                                 prune_ratio=args.prune_ratio,
                                 load_pruned_path=self.load_pruned_path,
                                 store_on_gpu=store_on_gpu,
                                 save_pruned_path=args.save_pruned_path,
                                 hook_manager=self.hook_manager)

        # threshold pruning
        else:
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
    def get_balanced_data_mask(present, absent, min_data_points, balance=True):
        num_filters = present.shape[1]
        min_per_bin_class = min_data_points // 2

        num_p = present.sum(dim=0).sum(dim=1).sum(dim=1)
        num_a = absent.sum(dim=0).sum(dim=1).sum(dim=1)

        discard_filters = set(np.where(np.bitwise_or(num_a < min_per_bin_class, num_p < min_per_bin_class))[0])
        kept_filters = np.where(np.bitwise_and(num_a >= min_per_bin_class, num_p >= min_per_bin_class))[0]

        if not balance:
            return present[:, kept_filters], None, discard_filters

        select = torch.zeros_like(present).type(torch.bool)
        for i in kept_filters:
            p_select = select[:, i][present[:, i] == 1]
            p_select[np.random.choice(int(num_p[i]), min_per_bin_class, replace=False)] = True
            select[:, i][present[:, i] == 1] = p_select

            a_select = select[:, i][absent[:, i] == 1]
            a_select[np.random.choice(int(num_a[i]), min_per_bin_class, replace=False)] = True
            select[:, i][absent[:, i] == 1] = a_select

        return present[:, kept_filters], select[:, kept_filters], discard_filters

    def label_data(self, *loaders, seed=0, order_by_importance=False, balance=False, save_acts=False):
        rand_state = np.random.get_state()
        np.random.seed(seed)

        self.network.to(self.device)

        if self.version == '2.0':
            train_loader, loader = loaders
            np.random.set_state(rand_state)

            #TODO experiments
            """
            if test_threshold_acc is not None:
                accs = []
                torch.save(self.network.state_dict(), 'temp.pth')
                def reload_params():
                    self.network.load_state_dict(torch.load('temp.pth'))

            # 0 No train test on t=1.0
            reload_params()
            self.dynamic_thresholder.fill_thresholds(1.0)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 1 No train test on t=1.5
            reload_params()
            self.dynamic_thresholder.fill_thresholds(1.5)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 2: Control, 5 epochs training - no thresholding
            reload_params()
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, threshold=False)
            accs += [self.dynamic_thresholder.test(loader, threshold=False)]

            # 3: Train on hard threshold (fixed, threshold=0.5)
            reload_params()
            self.dynamic_thresholder.fill_thresholds(0.5)
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, lr_thresholds=0, hard=True)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 4: Train on soft threshold (fixed, threshold=0.5)
            reload_params()
            self.dynamic_thresholder.fill_thresholds(0.5)
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, lr_thresholds=0, hard=False)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 5: Train on hard threshold (fixed, threshold=1.0)
            reload_params()
            self.dynamic_thresholder.fill_thresholds(1.0)
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, lr_thresholds=0, hard=True)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 6: Train on soft threshold (fixed, threshold=1.0)
            reload_params()
            self.dynamic_thresholder.fill_thresholds(1.0)
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, lr_thresholds=0, hard=False)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 7: Train on hard threshold (fixed, threshold=1.5)
            reload_params()
            self.dynamic_thresholder.fill_thresholds(1.5)
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, lr_thresholds=0, hard=True)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]

            # 8: Train on soft threshold (fixed, threshold=1.5)
            reload_params()
            self.dynamic_thresholder.fill_thresholds(1.5)
            self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, lr_thresholds=0, hard=False)
            accs += [self.dynamic_thresholder.test(loader, hard=True)]
            np.save('experiments-temp.npy', np.array(accs))
            """

            # train thresholds
            #self.dynamic_thresholder.train(train_loader, epochs=3, lr_network=0.01, fit_threshold=False,
            #                               hard_threshold=True)
            
            #self.dynamic_thresholder.fill_thresholds(0.5)

            # test performance
            #self.dynamic_thresholder.test(loader, threshold=False)

            # predict feature presence
            raw_activations, binary_activations = self.dynamic_thresholder.predict(loader,
                                                                                   output_raw=order_by_importance)

            if save_acts:
                np.save('%s-activations.npy' % self.layer_name, raw_activations.flatten().numpy())

            labels = binary_activations

            discard_filter_mask = None
            if balance:
                labels, select, discard_filters = self.get_balanced_data_mask(labels, -(labels - 1), 1000)
            else:
                labels, _, discard_filters = self.get_balanced_data_mask(labels, -(labels - 1), 1000)
                select = torch.ones_like(labels).type(torch.bool)
            if len(discard_filters) > 0:
                discard_filter_mask = np.zeros(binary_activations.shape[1]).astype(np.bool_)
                discard_filter_mask[(np.array(list(discard_filters)),)] = True

            # arrange VCs by filter importance
            sorted_filters = None
            if order_by_importance:
                importances, sorted_importances = self.pruner.sort_filter_activations_by_metric(raw_activations)
                sorted_filters = np.array([filter_idx for importance, filter_idx in sorted_importances])

            return discard_filter_mask, labels, select, sorted_filters

        else:
            loader, = loaders
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
