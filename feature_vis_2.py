import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.optim import SGD
import matplotlib.pyplot as plt
from pathlib import Path
import os.path as path
import copy
from tqdm.auto import tqdm


class PatchAggregator:
    """
    Class to handle visualization of maximally activating receptive field inputs for a particular conv filter
    """

    def __init__(self, model, layer_names, loader, num_patch=400, dataset_name='cifar20', activation_threshold=2.0):
        self.loader = loader
        self.num_patch = num_patch
        self.model = model
        self.feature_m_names = []
        self.feature_modules = []
        self.m_name_lookup = {}
        self.layers = layer_names
        self.activation_threshold = activation_threshold
        # TODO try without batchnorm
        """
        self.rf_model = copy.deepcopy(model)
        self.rf_model.apply(self.disable_batchnorm)
        self.feature_module_rf = None
        self.f_hook_rf = None
        """
        self.max_rf_input = []
        self.max_full_img = []
        self.save_file = 'feat-tracking-data-%s.npz' % dataset_name
        self.max_activations = {n: [] for n in self.layers}
        # TODO incorporate gradient into patch selection
        self.grads = {}
        self.f_hook_var = None
        self.b_hook_var = None
        self.batch_idxs = None

    def set_batch_idxs(self, idxs):
        self.batch_idxs = idxs

    def aggregate_patches(self):
        """
        Save patches obtained from receptive fields of maximally activated features
        :param loader:
        :return:
        """
        self.model.eval()
        self.setup_hooks()

        for indices, images, labels in tqdm(self.loader):
            self.batch_idxs = indices

            input = images.cuda()
            labels = labels.cuda()

            # save self.max_activations
            with torch.no_grad():
                out = self.model(input)

            """
            # save self.grads
            loss = self.model.ce_loss(out, labels)
            loss.backward()
            self.model.zero_grad()
            """

        self.save()
        self.remove_hooks()

    def remove_hooks(self):
        self.f_hook_var.remove()
        self.f_hook_var = None
        self.b_hook_var.remove()
        self.b_hook_var = None

    def save(self):
        """
        Save index of found maximal activations and corresponding sample index
        :return: None
        """
        # sort by activation value
        for l, acts in self.max_activations.items():
            self.max_activations[l] = sorted(acts, reverse=True)[:self.num_patch]

        acts = {n: np.array([idx for act, idx in l]) for n, l in self.max_activations.items()}

        np.savez(self.save_file, **acts)
        self.max_activations = {layer: [] for layer in self.layers}

    def disable_batchnorm(self, module):
        """
        If the passed module is a batchnorm instance, disable it.
        Used only to assist calculation of layers' receptive fields
        :param module: module
        :return:
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            module.forward_ = module.forward
            module.forward = lambda x: x

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, (name, m) in enumerate(self.model.named_modules()):
            # setup tracking for initial module
            if name in self.layers:
                self.f_hook_var = m.register_forward_hook(self.f_hook)
                self.feature_m_names += [name]
                self.feature_modules += [m]
                self.m_name_lookup[m] = name
                self.b_hook_var = m.register_backward_hook(self.b_hook)
        assert self.f_hook_var is not None

    def f_hook(self, module, input, output):
        """
        Forward hook to compute maximal activations
        :param module: the conv2d module
        :param input: conv2d input
        :param output: conv2d output
        :return: None
        """
        assert self.batch_idxs is not None,\
            'Batch indices were not saved before forward pass!'
        # determine maximal activations in each sample
        output = output.cpu()

        median = output.median()
        for i, out in enumerate(output):
            # select a random filter to find max activations for to capture more diversity in patches
            options = np.where(out.max(dim=1)[0].max(dim=1)[0] > median)[0]
            if len(options) == 0:
                continue
            filter_idx = np.random.choice(options)

            max_acts_ = np.where(out == out[filter_idx].max())  # get spatial coords of maximum activation
            max_acts_ = np.array([self.batch_idxs[i]] * len(max_acts_[0])), *max_acts_  # get full batch coords
            max_acts_ = list(zip(*max_acts_))  # reformat as separate indices
            # combine with activation values
            self.max_activations[self.m_name_lookup[module]] += [tuple([out.max().item()] + [max_acts_])]

    def b_hook(self, grad_in, grad_out):
        # TODO
        pass


class PatchTracker:

    def __init__(self, layer_names, dataset, save_file, patch_file='cifar20',
                 img_size=224, batch_size=100, device=0, num_workers=4, track_grad=False):
        self.track_grad = track_grad
        self.model = None
        self.layers = layer_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_file = 'feat-tracking-data-%s.npz' % patch_file
        self.acts_save = save_file.split('.')[0] + '-activations.npz'
        self.grads_save = save_file.split('.')[0] + '-gradients.npz'
        self.iters = 0

        # define lookups and idx tables
        self.act_idxs = np.load(self.patch_file)
        self.batch2dset_idxs = None

        self.num_patch = self.act_idxs[layer_names[0]].shape[0]
        self.activations_temp = {l: None for l in self.layers}
        self.grads_temp = {l: None for l in self.layers}
        self.activations = {l: [] for l in self.layers}
        self.grads = {l: [] for l in self.layers}
        self.img_size = img_size
        self.device = device
        self.f_hook_var = None
        self.b_hook_var = None
        self.m_names = []
        self.modules = []
        self.m_name_lookup = {}

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.loader = self.get_loader(dataset)

    def get_loader(self, dataset):
        idxs_used = []
        for l in self.layers:
            idxs_used += list(self.act_idxs[l][:, 0, 0])
        idxs_used = np.array(idxs_used)

        sampler = SubsetRandomSampler(idxs_used)
        loader = DataLoader(dataset, self.batch_size, shuffle=False, sampler=sampler, num_workers=self.num_workers)

        return loader

    def invert_lookup(self, idxs):
        """
        Lookup inversion
        :return: inverted lookup
        """
        idxs_ = idxs[:, None].repeat(len(idxs), 1)
        counts = np.arange(len(idxs))[None].repeat(len(idxs), 0)
        return np.where(idxs_ == counts)[1][idxs]

    def init_stats(self, layer, feature_length):
        self.activations_temp[layer] = np.zeros((self.num_patch, feature_length))
        if self.track_grad:
            self.grads_temp[layer] = np.zeros((self.num_patch, feature_length))

    def reset_stats(self):
        self.activations_temp = {l: None for l in self.layers}
        self.grads_temp = {l: None for l in self.layers}

    def save(self):
        acts = {}
        grads = {}
        for l in self.layers:
            # make sure all patch activations have been accumulated for this layer
            assert np.all(self.activations_temp[l] != 0)
            self.activations[l] += [self.activations_temp[l]]
            acts[l] = np.stack(self.activations[l], axis=0)
            if self.track_grad:
                # make sure all patch grads have been accumulated for this layer
                #assert np.all(self.grads_temp[l] != 0)
                self.grads[l] += [self.grads_temp[l]]
                grads[l] = np.stack(self.grads[l], axis=0)

        np.savez(self.acts_save, **acts)
        if self.track_grad:
            np.savez(self.grads_save, **grads)

        self.reset_stats()

    def probe(self, model):
        """
        Probe the network layer specified at initilization to find activations across patches
        :return: np.array of max layer activations for each patch
        """
        self.model = model
        self.setup_hooks()

        for indices, images, labels in tqdm(self.loader):
            images, labels = images.to(self.device), labels.to(self.device)

            context = NullContext()
            if not self.track_grad:
                context = torch.no_grad()

            # set self.batch2dset_idxs
            self.batch2dset_idxs = indices

            with context:
                out = self.model(images)

                if self.track_grad:
                    loss = self.ce_loss(out, labels)
                    loss.backward(retain_graph=False)
                    self.model.zero_grad()

        self.save()
        self.iters += 1
        self.remove_hook()
        self.model = None

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, (name, m) in enumerate(self.model.named_modules()):
            # setup tracking for initial module
            if name in self.layers:
                self.f_hook_var = m.register_forward_hook(self.f_hook)
                if self.track_grad:
                    self.b_hook_var = m.register_backward_hook(self.b_hook)
                self.m_names += [name]
                self.modules += [m]
                self.m_name_lookup[m] = name
        assert self.f_hook_var is not None

    def f_hook(self, module, input, output):
        assert self.batch2dset_idxs is not None, 'Batch idxs were not saved before forward pass!'
        layer = self.m_name_lookup[module]

        # initialize storage if we havent already
        if self.activations_temp[layer] is None:
            self.init_stats(layer, output.shape[1])

        # save activation value
        for batch_idx, dset_idx in enumerate(self.batch2dset_idxs):
            dset_idx = int(dset_idx)
            patch_idx = np.where(self.act_idxs[layer][:, 0, 0] == dset_idx)[0]
            if len(patch_idx) == 0:
                continue
            assert len(patch_idx) == 1, 'Currently only support for single patch per sample in dataset'
            patch_idx = int(patch_idx[0])
            self.activations_temp[layer][patch_idx] = \
                output[(batch_idx, slice(None), *self.act_idxs[layer][patch_idx, 0, 2:])].detach().cpu().numpy()

    def b_hook(self, module, grad_in, grad_out):
        (grad_out,) = grad_out

        assert self.batch2dset_idxs is not None, 'Batch idxs were not saved before forward pass!'
        assert self.grads_temp is not None, "grads_temp was not initialized"
        layer = self.m_name_lookup[module]

        # save gradient information
        for batch_idx, dset_idx in enumerate(self.batch2dset_idxs):
            dset_idx = int(dset_idx)
            patch_idx = np.where(self.act_idxs[layer][:, 0, 0] == dset_idx)[0]
            if len(patch_idx) == 0:
                continue
            assert len(patch_idx) == 1, 'Currently only support for single patch per sample in dataset'
            patch_idx = int(patch_idx[0])
            self.grads_temp[layer][patch_idx] = grad_out[(batch_idx, slice(None),
                                                          *self.act_idxs[layer][patch_idx, 0, 2:])].cpu().numpy()

    def remove_hook(self):
        self.f_hook_var.remove()
        self.f_hook_var = None
        if self.b_hook_var:
            self.b_hook_var.remove()
            self.b_hook_var = None


class GradTracker:

    def __init__(self, loader, model, layer_name, img_size=224, batch_size=100, device=0):
        self.loader = loader
        self.model = model
        self.layer = layer_name
        self.grads = []
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.hook = None
        self.m_name = None
        self.module = None

    def accumulate(self):
        """
        Get gradients from all samples in the passed dataloader
        :return: np.array of gradient backpropped to each feature for each sample in the dataloader
        """
        self.model.eval()
        self.setup_hooks()

        for indices, images, labels in tqdm(self.loader):
            images = images.cuda(device=self.device)
            labels = labels.cuda(device=self.device)

            out = self.model(images)
            loss = self.model.ce_loss(out, labels)

            loss.backward()
            self.model.zero_grad()

        ret = np.concatenate(self.grads, axis=0)
        self.grads = []
        self.remove_hook()
        return ret

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, (name, m) in enumerate(self.model.named_modules()):
            # setup tracking for initial module
            if name == self.layer:
                self.hook = m.register_backward_hook(self.b_hook)
                self.m_name = name
                self.module = m

    def remove_hook(self):
        self.hook.remove()
        self.hook = None

    def b_hook(self, grad_in, grad_out):
        """
        Backward hoot to save gradient of loss wrt activation
        :param grad_in: gradient of module input
        :param grad_out: gradient of module output
        :return: None
        """
        self.grads += grad_out.cpu()  # [batch X features]


class NullContext:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass