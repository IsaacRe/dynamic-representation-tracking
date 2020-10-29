from os.path import exists
import os
from time import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from dataset_batch_cifar import CIFAR20
from dataset_batch_imagenet import ImageNet
from vc_utils.activation_threshold import ConvActivationThreshold
from vc_utils.activation_tracker import ActivationTracker
from vc_utils.prune import Pruner
from vc_utils.hook_utils import CustomContext, HookManager
from vc_utils.dynamic_threshold import ThresholdLearner, sigmoid_with_temperature

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


def test_features(test_loader, model, layer_name, device=0):
    pruner = Pruner(model, layer_name)
    results = pruner.test_features(test_loader, layer_name, device=device)
    return results


def test_threshold_acc(args, test_loader, model, layer_name, train_loader=None, ts=None, device=0):
    if ts is None:
        ts = [args.present_vc_threshold]

    labeler = VCLabeler(args, model, layer_name, device=device)
    return labeler.test_thresholds(test_loader, *ts, train_loader=train_loader, epochs=args.vc_epochs,
                                   epochs_ft=args.ft_fc_epochs,
                                   lr=args.lr_threshold, lr_ft=args.ft_fc_lr)


def compute_corr(loader, model, act_tracker, module_name, device):
    print('Computing mean')
    mean = np.zeros(512)
    total = 0
    for i, img, valid, lbl in tqdm(loader):
        with act_tracker.track_all_context():
            model(img.to(device))
            activations = act_tracker.get_module_activations(module_name, cpu=False)
            mean = mean + activations[:,:,0,0].sum(dim=0).cpu().numpy()
            total += img.shape[0]
    mean = mean / total
    pass

    print('Computing correlation matrix')
    var = np.zeros(512)
    corr = np.zeros((512, 512))
    total = 0
    for i, img, valid, lbl in tqdm(loader):
        with act_tracker.track_all_context():
            model(img.to(device))
            activations = act_tracker.get_module_activations(module_name, cpu=False)

            # compute var
            deviation = activations[:,:,0,0].cpu().numpy() - mean[None]
            var = var + (deviation ** 2).sum(axis=0)
            total += img.shape[0]

            # compute correlation
            corr = corr + deviation.T.dot(deviation)

    # compute var
    var = var / total
    stds = np.sqrt(var)
    stds = stds[:,None].dot(stds[None])

    # compute corr
    corr = corr / total
    corr = corr / stds

    pass


def train_classification_layer_v1(args, network, module_name, vc_dset, device=0, act_tracker=None,
                                  classification_layer=None, uniform_init=False):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    in_dim = 512
    normalize = False
    if classification_layer is None:
        uniform_init = 1/512 if uniform_init else None
        classification_layer = VCLogitLayer(in_dim, vc_dset.kept_idxs, uniform_init=uniform_init,
                                            threshold_estimate=args.present_vc_threshold).to(device)

    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    filter_weights = None
    weights = None
    optim = torch.optim.Adam(classification_layer.parameters(), lr=0.03)
    train_loader = vc_dset.get_loader(args.batch_size_train_vc, shuffle=True)

    """
    # DEBUG
    k = vc_dset.kept_idxs

    def set_diag(x):
        w = classification_layer.conv1x1.weight
        w_ = torch.zeros_like(w)
        w_[:, k, 0, 0] = torch.diag(torch.ones(w.shape[0]) * x).to(device)
        w_ += (1 - x) / 512
        w.data = w_.data

    diag_vs = [0.85, 0.9, 0.95, 0.97, 0.99, 1.0, 1.1]
    loss_initial = []
    loss_after_epoch = []
    new_mean = []
    args.vc_epochs = len(diag_vs)
    # END DEBUG"""
    #compute_corr(train_loader, network, act_tracker, module_name, device)

    num_epochs = 5
    pbar = tqdm(total=len(train_loader) * num_epochs)
    losses = []
    for e in range(num_epochs):

        if e == 4:
            optim = torch.optim.Adam(classification_layer.parameters(), lr=0.005)

        """
        # DEBUG
        set_diag(diag_vs[e])
        # END DEBUG
        """
        vc_pos_count = np.zeros(512)
        vc_total = 0

        for i, img, valid, vc_lbl in train_loader:

            vc_pos_count = vc_pos_count + vc_lbl[:,:,0,0].sum(dim=0).cpu().numpy()
            vc_total += img.shape[0]

            if filter_weights is None:
                filter_weights = vc_dset.filter_weights[None, :, None, None].repeat(vc_lbl.shape[0], 1,
                                                                                    *vc_lbl.shape[2:]).to(device)
                weights = torch.ones_like(filter_weights).to(device)
            img, vc_lbl = img.to(device), vc_lbl.to(device)
            with act_tracker.track_all_context():
                network(img)
                activations = act_tracker.get_module_activations(module_name, cpu=False)
            out = classification_layer(activations)
            loss = bce(out, vc_lbl)
            weights[:] = 1.0
            weights[vc_lbl == 1.0] = filter_weights[vc_lbl == 1.0]
            loss = (loss * weights)[valid].mean()  # only use valid localities in framing loss
            losses += [loss.item()]
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.update(1)

        """
            # DEBUG
            loss_initial += [loss.item()]

        loss_after_epoch += [loss.item()]
        new_mean += [classification_layer.conv1x1.weight[:, k, 0, 0].diagonal().mean().item()]
        # END DEBUG
        """
    np.save('vc_count.npy', vc_pos_count)
    pbar.close()

    return classification_layer


def test_vc_accuracy_v1(args, network, module_name, vc_dset_train, vc_dset_test, device=0, classification_layer=None,
                        act_tracker=None, recall=False, train=False, uniform_init=False):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    if classification_layer is None:
        if train:
            classification_layer = train_classification_layer_v1(args, network, module_name, vc_dset_train,
                                                                 device=device, act_tracker=act_tracker,
                                                                 uniform_init=uniform_init)
            classification_layer.set_vc_idxs(vc_dset_test.kept_idxs)
        else:
            in_dim = 512
            classification_layer = VCLogitLayer(in_dim, vc_dset_test.kept_idxs, args.present_vc_threshold).to(device)

    total = np.zeros((vc_dset_test.total_classes,))
    correct = np.zeros((vc_dset_test.total_classes,))
    for i, img, valid, vc_lbl in tqdm(vc_dset_test.get_loader(args.batch_size_test_vc)):
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

    return correct / total * 100., classification_layer.weight.data.cpu()[:, :, 0, 0]


def train_classification_layer_v2(args, network, module_name, vc_dset, device=0, act_tracker=None,
                               uniform_init=False):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    in_dim = act_tracker.modules[module_name].out_channels
    if not uniform_init:
        classification_layer = VCLogitLayerV2(in_dim, vc_dset.kept_idxs).to(device)
    else:
        classification_layer = VCLogitLayerV2(in_dim, vc_dset.kept_idxs, uniform_init=1 / 512).to(device)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    filter_weights = None
    weights = None
    optim = torch.optim.SGD(classification_layer.parameters(), lr=args.lr_vc)
    train_loader = vc_dset.get_loader(args.batch_size_train_vc)
    pbar = tqdm(total=args.vc_epochs * len(train_loader))
    for e in range(args.vc_epochs):
        for i, img, valid, vc_lbl in train_loader:
            if filter_weights is None:
                filter_weights = vc_dset.filter_weights[None, :, None, None].repeat(vc_lbl.shape[0], 1,
                                                                                    *vc_lbl.shape[2:]).to(device)
                weights = torch.ones_like(filter_weights).to(device)
            img, vc_lbl = img.to(device), vc_lbl.to(device)
            with act_tracker.track_all_context():
                network(img)
                activations = act_tracker.get_module_activations(module_name, cpu=False)
            out = classification_layer(activations)
            loss = bce(out, vc_lbl)
            weights[:] = 1.0
            weights[vc_lbl == 1.0] = filter_weights[vc_lbl == 1.0]
            loss = (loss * weights)[valid].mean()  # only use valid localities in framing loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            pbar.update(1)
    pbar.close()

    return classification_layer


def test_vc_accuracy_v2(args, network, module_name, vc_dset_train, vc_dset, device=0, classification_layer=None,
                        act_tracker=None, uniform_init=False, epochs=1, train=True, recall=False):
    if act_tracker is None:
        act_tracker = ActivationTracker(module_names=[module_name], network=network, store_on_gpu=True)
    if classification_layer is None:
        if train:
            classification_layer = train_classification_layer_v2(args, network, module_name, vc_dset_train,
                                                                 device=device, act_tracker=act_tracker,
                                                                 uniform_init=uniform_init)
            classification_layer.set_vc_idxs(vc_dset.kept_idxs)
        else:
            classification_layer = VCLogitLayerV2(None, vc_dset.kept_idxs)

    total = np.zeros((vc_dset.total_classes,))
    correct = np.zeros((vc_dset.total_classes,))
    classification_layer.eval()
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
            correct += (correct_mask.type(torch.float) * vc_lbl).sum(dim=0).sum(dim=1).sum(dim=1).type(
                torch.int).cpu().numpy()

            # Get total number of positive, valid data points
            total_ = vc_lbl * valid.type(torch.float).to(device)
            total_ = total_.sum(dim=0).sum(dim=1).sum(dim=1)
            total += total_.type(torch.int).cpu().numpy()
        else:
            correct += correct_mask.sum(dim=0).sum(dim=1).sum(dim=1).cpu().numpy()

            # Get total number of valid data points for each vc class
            total += valid.transpose(1, 0).flatten(start_dim=1, end_dim=3).sum(dim=1).cpu().numpy()

    return correct / total * 100., classification_layer.threshold.data.cpu()


def _define_VisualConceptDataset(base_dataset):
    class VisualConceptDataset(base_dataset):

        def __init__(self, args, network, module_name, *dset_args, train=True, batch_size=100, device=0,
                     store_on_gpu=False, version='1', balance=False, cache=False, **dset_kwargs):
            print('Getting VC %s dataset...' % ('train' if train else 'test'))
            super(VisualConceptDataset, self).__init__(*dset_args, train=train, **dset_kwargs, crop=False)
            self.is_train = train
            self.labeler = VCLabeler(args, network, module_name, device=device, store_on_gpu=store_on_gpu)
            self.valid_localities = None
            self.filter_weights = None
            self._prune_mask = None
            self.sorted_filters = None
            self.save_file = 'cache/%s-vc_%s-%s_data-t_%s.npz' % (args.save_all_dir.split('/')[-1],
                                                                  module_name,
                                                                  'train' if train else 'test',
                                                                  str(args.present_vc_threshold))

            if cache and exists(self.save_file):
                self.load_cache()
            else:
                if train:
                    labels_attr = 'train_labels'
                else:
                    assert hasattr(self, 'test_labels')
                    labels_attr = 'test_labels'
                setattr(self, 'class_' + labels_attr, getattr(self, labels_attr))
                self._prune_mask, labels, self.valid_localities, self.sorted_filters, self.num_p = \
                    self.labeler.label_data(self.get_loader(batch_size=batch_size, shuffle=True),
                                            self.get_loader(batch_size=batch_size),
                                            order_by_importance=True,
                                            balance=balance,
                                            save_acts=args.save_activations)
                setattr(self, labels_attr, labels)

                # get weights for positive samples for data balancing
                ratio_positive = labels.mean(dim=0).mean(dim=1).mean(dim=1)
                self.filter_weights = (1 - ratio_positive) / ratio_positive

                if cache:
                    self.save_cache()

            print('\n\n A total of %d/%d filters discarded\n\n' % (len(self.pruned_idxs),
                                                                   len(self.pruned_idxs) + len(self.kept_idxs)))

            if train:
                self.total_classes = self.train_labels.shape[1]
            else:
                self.total_classes = self.test_labels.shape[1]

        def save_cache(self):
            np.savez(self.save_file,
                     labels=getattr(self, 'train_labels' if self.is_train else 'test_labels').numpy(),
                     valid_localities=self.valid_localities.numpy(),
                     filter_weights=self.filter_weights.numpy(),
                     prune_mask=self._prune_mask,
                     sorted_filters=self.sorted_filters,
                     num_positive=self.num_p)

        def load_cache(self):
            file = np.load(self.save_file)
            setattr(self, 'train_labels' if self.is_train else 'test_labels', file['labels'])
            self.valid_localities = torch.Tensor(file['valid_localities']).type(torch.bool)
            self.filter_weights = torch.Tensor(file['filter_weights'])
            self._prune_mask = file['prune_mask']
            if self._prune_mask.size == 1 and self._prune_mask.item() is None:
                self._prune_mask = None
            self.sorted_filters = file['sorted_filters']
            self.num_p = file['num_positive']

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

    def __init__(self, in_dim, vc_idxs, threshold_estimate=0.5, uniform_init=None, normalize=False):
        super(VCLogitLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(in_dim, in_dim, 1, 1))
        self.threshold = threshold_estimate
        self.vc_idxs = vc_idxs
        self.normalize = normalize

        if uniform_init is None:
            # initialize layer weights with best guess under assumption that activations never change
            for idx in vc_idxs:
                self.weight.data[idx, idx, :, :] = 1.0
        else:
            self.weight.data.fill_(uniform_init)

        #self.bias.data[:] = -threshold_estimate

    def forward(self, x):
        # normalize
        if self.normalize:
            x = x / x.abs().max(dim=0)[0]
        #w = F.softmax(self.weight, dim=1)
        return F.conv2d(x, self.weight) - self.threshold
        #return self.conv1x1(x)[:, self.vc_idxs]

    def set_vc_idxs(self, vc_idxs):
        self.vc_idxs = vc_idxs

    @property
    def vc_identiy(self):
        return self.weight[np.arange(512), self.vc_idxs, 0, 0]

    @property
    def vc_weight(self):
        return self.weight[:, :, 0, 0]

    @property
    def vc_threshold(self):
        return self.threshold


class VCLogitLayer_(torch.nn.Module):

    def __init__(self, in_dim, vc_idxs, threshold_estimate=0.5, uniform_init=None):
        super(VCLogitLayer, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.vc_idxs = vc_idxs

        if uniform_init is None:
            # initialize layer weights with best guess under assumption that activations never change
            self.conv1x1.weight.data.zero_()
            for idx in vc_idxs:
                self.conv1x1.weight.data[idx, idx, :, :] = 1.0
        else:
            self.conv1x1.weight.data.fill_(uniform_init)

        self.conv1x1.bias.data[:] = -threshold_estimate

    def forward(self, x):
        return self.conv1x1(x)[:, self.vc_idxs]

    def set_vc_idxs(self, vc_idxs):
        self.vc_idxs = vc_idxs

    @property
    def vc_identiy(self):
        return self.conv1x1.weight[np.arange(512), self.vc_idxs, 0, 0]

    @property
    def vc_weight(self):
        return self.conv1x1.weight[:, :, 0, 0]

    @property
    def vc_threshold(self):
        return self.conv1x1.bias

class VCLogitLayerV2(torch.nn.Module):

    def __init__(self, in_dim, vc_idxs, threshold_estimate=0.5, uniform_init=None, temperature=1.0):
        super(VCLogitLayerV2, self).__init__()
        self.threshold = torch.nn.Parameter(torch.zeros(1).fill_(threshold_estimate))
        self.vc_idxs = vc_idxs

    def set_vc_idxs(self, vc_idxs):
        self.vc_idxs = vc_idxs

    def forward(self, x):
        return x[:, self.vc_idxs] - self.threshold

    def parameters(self):
        return [self.threshold]


class VCLabeler:

    def __init__(self, args, network, module_name, store_on_gpu=False, device=0,
                 init_t=0.5, temperature=0.2):
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
        self.load_pruned_path = args.load_pruned_path if exists(args.load_pruned_path) and args.load_prune_mask \
            else None

        # dynamic pruning
        self.dynamic_thresholder = ThresholdLearner(network, module_name, hook_manager=self.hook_manager,
                                                    device=device, init_t=init_t,
                                                    temperature=temperature, lr=args.lr_threshold)
        self.pruner = Pruner(network, module_name,
                             prune_ratio=args.prune_ratio,
                             load_pruned_path=self.load_pruned_path,
                             store_on_gpu=store_on_gpu,
                             save_pruned_path=args.save_pruned_path,
                             hook_manager=self.hook_manager)

    def test_thresholds(self, test_loader, *ts, train_loader=None, epochs=5, epochs_ft=5, lr=0.01, lr_ft=0.02,
                        test_full=True):
        accs = []
        self.network.cpu()

        def make_tmp_file():
            return 'tmp/%s' % str(np.random.rand()).split('.')[-1]

        tmp_file = make_tmp_file()
        while exists(tmp_file):
            tmp_file = make_tmp_file()
        torch.save(self.network.state_dict(), tmp_file)

        def reload_params():
            self.network.cpu()
            self.network.load_state_dict(torch.load(tmp_file))
            self.network.cuda(self.device)

        for t in ts:
            reload_params()
            self.dynamic_thresholder.set_threshold(t)
            if train_loader is not None:
                self.network.train()
                self.dynamic_thresholder.train(train_loader, epochs=epochs, lr_network=lr, lr_thresholds=0, hard=True)
            self.network.eval()
            accs += [self.dynamic_thresholder.test(test_loader)]

        ft_acc = None
        if test_full:
            reload_params()
            if train_loader is not None:
                self.network.train()
                self.dynamic_thresholder.train(train_loader, epochs=epochs, lr_network=lr_ft, threshold=False)
            self.network.eval()
            ft_acc = self.dynamic_thresholder.test(test_loader, threshold=False)

        os.remove(tmp_file)
        return accs, ft_acc

    @staticmethod
    def get_balanced_data_mask(present, absent, min_data_points, balance=True):
        num_filters = present.shape[1]
        min_per_bin_class = min_data_points // 2

        num_p = present.sum(dim=0).sum(dim=1).sum(dim=1)
        num_a = absent.sum(dim=0).sum(dim=1).sum(dim=1)

        discard_filters = set(np.where(np.bitwise_or(num_a < min_per_bin_class, num_p < min_per_bin_class))[0])
        kept_filters = np.where(np.bitwise_and(num_a >= min_per_bin_class, num_p >= min_per_bin_class))[0]

        if not balance:
            return present[:, kept_filters], None, discard_filters, num_p

        select = torch.zeros_like(present).type(torch.bool)
        for i in kept_filters:
            p_select = select[:, i][present[:, i] == 1]
            p_select[np.random.choice(int(num_p[i]), min_per_bin_class, replace=False)] = True
            select[:, i][present[:, i] == 1] = p_select

            a_select = select[:, i][absent[:, i] == 1]
            a_select[np.random.choice(int(num_a[i]), min_per_bin_class, replace=False)] = True
            select[:, i][absent[:, i] == 1] = a_select

        return present[:, kept_filters], select[:, kept_filters], discard_filters, num_p

    def label_data(self, *loaders, seed=0, order_by_importance=False, balance=False, save_acts=False,
                   train_threshold=False):
        rand_state = np.random.get_state()
        np.random.seed(seed)

        self.network.to(self.device)

        train_loader, loader = loaders
        np.random.set_state(rand_state)

        if train_threshold:
            print('VCLabeler: learning threshold for feature binarization...')
            self.dynamic_thresholder.train(train_loader)
        else:
            print('VCLabeler: using fixed threshold of %.2f' % self.init_t)

        # predict feature presence
        print('VCLabeler: binarizing featuremap activations...')
        raw_activations, binary_activations = self.dynamic_thresholder.predict(loader,
                                                                               output_raw=order_by_importance)
        if save_acts:
            np.save('%s-activations.npy' % self.layer_name, raw_activations.flatten().numpy())

        print('VCLabeler: Discarding filters with insufficient positive localities')

        labels = binary_activations

        discard_filter_mask = None
        if balance:
            labels, select, discard_filters, num_p = self.get_balanced_data_mask(labels, -(labels - 1),
                                                                                 self.min_data_points)
        else:
            labels, _, discard_filters, num_p = self.get_balanced_data_mask(labels, -(labels - 1),
                                                                            self.min_data_points)
            select = torch.ones_like(labels).type(torch.bool)
        if len(discard_filters) > 0:
            discard_filter_mask = np.zeros(binary_activations.shape[1]).astype(np.bool_)
            discard_filter_mask[(np.array(list(discard_filters)),)] = True

        print('VCLabeler: Sorting remaining filters')

        # arrange VCs by filter importance
        sorted_filters = None
        if order_by_importance:
            importances, sorted_importances = self.pruner.sort_filter_activations_by_metric(raw_activations)
            sorted_filters = np.array([filter_idx for importance, filter_idx in sorted_importances])

        print('VC labeling finished.')

        return discard_filter_mask, labels, select, sorted_filters, num_p

        """ loader, = loaders
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
        """


VisualConceptDataset = NullVisualConceptDataset
