import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
import matplotlib.pyplot as plt
from pathlib import Path
import os.path as path
import copy
from tqdm.auto import tqdm


class EarlyStopping(Exception):
    pass


class PatchAggregator:
    """
    Class to handle visualization of maximally activating receptive field inputs for a particular conv filter
    """

    def __init__(self, model, layer_name, loader, num_patch=400):
        self.loader = loader
        self.num_patch = num_patch
        self.model = model
        self.model.eval()
        self.rf_model = copy.deepcopy(model)
        self.rf_model.apply(self.disable_batchnorm)
        self.feature_m_name = None
        self.feature_module = None
        self.feature_module_rf = None
        self.layer = layer_name
        self.max_rf_input = []
        # TODO vvv
        self.max_full_img = []
        self.save_dir = 'rf-imgs/%s' % layer_name
        Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        self.epoch = 0
        self.max_activations = [[] for i in range(loader.batch_size)]
        self.f_hook = None
        self.f_hook_rf = None
        self.current_input = None
        self.img_index = 0
        self.saved_inputs = None
        self.setup_hooks()

    def aggregate_patches(self):
        """
        Save patches obtained from receptive fields of maximally activated features
        :param loader:
        :return:
        """
        for indices, images, labels in tqdm(self.loader):
            input = images.cuda()
            # save self.max_activations
            try:
                with torch.no_grad():
                    self.model(input)
            except EarlyStopping:
                pass
            # save self.max_rf_input and self.max_rf_full_img
            input.requires_grad = True
            self.current_input = input
            try:
                self.rf_model(input)
            except EarlyStopping:
                pass

            self.save()

    def save(self):
        """
        Concatenate and save the max rf inputs accumulated to a npy file
        :return: None
        """
        def shape2slice(shape):
            return slice(0, shape[0]), slice(0, shape[1]), slice(None)

        if self.saved_inputs is not None:
            max_shape = max([inp.shape[0] for inp in self.max_rf_input] + [self.saved_inputs.shape[1]]), \
                        max([inp.shape[1] for inp in self.max_rf_input] + [self.saved_inputs.shape[2]]), 3
            saved_inputs = np.zeros((len(self.max_rf_input) + self.saved_inputs.shape[0], *max_shape))
            saved_inputs[(slice(self.saved_inputs.shape[0]), *shape2slice(self.saved_inputs.shape[1:]))] = self.saved_inputs
        else:
            max_shape = max([inp.shape[0] for inp in self.max_rf_input]), \
                        max([inp.shape[1] for inp in self.max_rf_input]), 3
            saved_inputs = np.zeros((len(self.max_rf_input), *max_shape))
        start_idx = 0 if self.saved_inputs is None else self.saved_inputs.shape[0]
        for i, inp in enumerate(self.max_rf_input, start_idx):
            saved_inputs[(i, *shape2slice(inp.shape))] = inp
        self.saved_inputs = saved_inputs
        np.save('%s.npy' % self.save_dir, self.saved_inputs)
        self.img_index += len(self.max_rf_input)
        self.max_rf_input = []
        self.max_full_img = []

    def get_img(self, vect):
        """
        Normalizes and transposes the image array to be a valid plottable image
        :param vect: [F x H x W] image array
        :return: transformed image
        """
        img_min = vect.min().item()
        img_max = vect.max().item()
        img = vect.data.cpu().transpose(0, 2).numpy()
        img -= img_min
        img /= (img_max - img_min)

        return img

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

    def compute_rf(self, activation, rf_input, samples):
        """
        Get receptive field of the feature of the specified sample at the specified spatial index.
        :param activation: [>= batch]-sized tensor containing max activations for each sample in the batch
        :param rf_input: the input to the network with require_grad=True
        :return: The section of the original input responsible for computing the passed activation
        """
        activation.sum().backward(retain_graph=True)
        rf_masks = [np.where(rf_input.grad[s].cpu() != 0) for s in samples]
        rf_input.grad.zero_()
        self.rf_model.zero_grad()

        return rf_masks

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, ((name, m), (name_, m_)) in enumerate(zip(self.model.named_modules(), self.rf_model.named_modules())):
            assert name == name_
            # setup tracking for initial module
            if name == self.layer:
                self.f_hook = m.register_forward_hook(self.f_hook_feature)
                self.feature_m_name = name
                self.feature_module = m
                self.f_hook_rf = m_.register_forward_hook(self.f_hook_findrf)
                self.feature_module_rf = m_

    def f_hook_findrf(self, module, input, output):
        """
        Forward hook to save rf data corresponding to previously computed maximal activations
        :param module: the conv2d module
        :param input: the conv2d input
        :param output: the conv2d output
        :return:
        """
        # iteratively find batches of receptive fields of maximal activations across batch dim
        while any([len(idxs) > 0 for idxs in self.max_activations]):
            # max_activations should be a list(list(idx)) for activation[idx] = a particular max activation

            # get list of samples for which max activations were found
            #samples = [s for idxs, s in zip(self.max_activations, range(self.loader.batch_size)) if len(idxs) > 0]
            samples = [x[0][0] for x in self.max_activations if len(x) > 0]
            next_batch = zip(*[idxs.pop(0) for idxs in self.max_activations if len(idxs) > 0])
            next_batch = tuple([np.array(idxs) for idxs in next_batch])
            next_batch = output[next_batch]

            # compute sample-wise rf for corresponding max activations
            rf_masks = self.compute_rf(next_batch, self.current_input, samples)

            # get masks of individual samples
            for rf_mask, s in zip(rf_masks, samples):
                h_slice = slice(rf_mask[1].min(), rf_mask[1].max() + 1)
                w_slice = slice(rf_mask[2].min(), rf_mask[2].max() + 1)
                img_arr = self.current_input[s, :, h_slice, w_slice]

                self.max_rf_input += [self.get_img(img_arr)]

            # TODO max_rf_full_img
        self.max_activations = [[] for _ in self.max_activations]

    def f_hook_feature(self, module, input, output):
        """
        Forward hook to compute maximal activations
        :param module: the conv2d module
        :param input: conv2d input
        :param output: conv2d output
        :return:
        """
        # determine maximal activations in each sample
        # TODO try simply gathering highest activations for each sample
        output = output.cpu()
        max_acts = []
        for i, out in enumerate(output):
            max_acts_ = np.where(out == out.max())
            max_acts_ = np.array([i] * len(max_acts_[0])), *max_acts_
            max_acts_ = list(zip(*max_acts_))
            max_acts += [tuple([out.max().item()] + [max_acts_])]
        max_acts = sorted(max_acts, reverse=True)[:self.num_patch*self.loader.batch_size//len(self.loader.dataset)]
        for val, act in max_acts:
            self.max_activations += [act]


class PatchTracker:

    def __init__(self, model, layer_name, img_size=224, batch_size=100, device=0):
        self.model = model
        self.layer = layer_name
        self.patch_file = 'rf-imgs/%s.npy' % layer_name
        assert path.exists(self.patch_file)
        self.patches = np.load(self.patch_file).transpose(0, 3, 1, 2)
        self.num_patch = self.patches.shape[0]
        self.activations = []
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.hook = None
        self.m_name = None
        self.module = None

    def probe(self):
        """
        Probe the network layer specified at initilization to find activations across patches
        :return: np.array of max layer activations for each patch
        """
        self.setup_hooks()

        # gather patches
        def shape2slice(shape):
            return tuple([slice(0, s) for s in shape])

        for i in tqdm(range(0, self.patches.shape[0], self.batch_size)):
            batch = torch.zeros(self.batch_size, 3, self.img_size, self.img_size).cuda(self.device)
            data = torch.cuda.FloatTensor(self.patches[i: i+self.batch_size])
            batch[shape2slice(data.shape)] = data

            with torch.no_grad():
                self.model(batch)

        ret = np.concatenate(self.activations, axis=0)
        self.activations = []
        self.remove_hook()
        return ret

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, (name, m) in enumerate(self.model.named_modules()):
            # setup tracking for initial module
            if name == self.layer:
                self.hook = m.register_forward_hook(self.f_hook)
                self.m_name = name
                self.module = m

    def f_hook(self, module, input, output):
        # get maximum activation for each patch in the batch
        self.activations += [output.max(dim=3)[0].max(dim=2)[0].cpu().numpy()]

    def remove_hook(self):
        self.hook.remove()
        self.hook = None


class FeatureVis:
    """
    Class to handle visualization of maximally activating receptive field inputs for a particular conv filter during
    continual class learning
    """

    def __init__(self, model, layer_name, idx, save_file):
        self.model = model
        #self.rf_model = copy.deepcopy(model)
        #self.rf_model.apply(self.disable_batchnorm)
        self.f_hook_first_ = None
        self.f_hook_feature_ = None
        self.feature_m_name = None
        self.feature_module = None
        self.layer = layer_name
        self.idx = idx
        self.max_activation = 0
        self.max_rf_input = None
        self.max_full_img = None
        self.current_input = None
        self.save_dir = save_file.split('.')[0] + '-rf-imgs'
        Path(self.save_dir).mkdir(exist_ok=True)
        self.epoch = 0
        self.max_activations = []
        self.setup_hooks()

    def advance_epoch(self):
        self.max_activations += [self.max_activation]
        self.max_activation = 0
        np.save(self.save_dir + '/epoch-%d-rf.npy' % self.epoch, self.max_rf_input)
        np.save(self.save_dir + '/epoch-%d-img.npy' % self.epoch, self.max_full_img)
        self.max_rf_input = self.max_full_img = None
        self.epoch += 1

    def get_img(self, vect):
        img_min = vect.min().item()
        img_max = vect.max().item()
        img = vect.data.cpu().transpose(0, 2).numpy()
        img -= img_min
        img /= (img_max - img_min)

        return img

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

    def compute_rf(self, idx):
        """
        Get receptive field of the feature of the specified sample at the specified spatial index.
        :param batch_idx: index of the sample in batch whose feature activation is maximal
        :param x_idx: spatial index of the feature whose receptive field to compute
        :param y_idx: spatial index of the feature whose receptive field to compute
        :return: The section of the original input responsible for computing the passed activation
        """
        x_idx, y_idx = idx

        # calculated for layer1.2.conv2 ONLY
        def get_range(activation_idx):
            assert self.layer == 'layer1.2.conv2',\
                'Receptive field calculations have not been made for layer %s' % self.layer
            return max(activation_idx * 4 - 29, 0), min(activation_idx * 4 + 30, 224)

        rfield_idx = slice(*get_range(x_idx)), slice(*get_range(y_idx))

        return self.current_input[rfield_idx], rfield_idx

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, (name, m) in enumerate(self.model.named_modules()):
            # setup tracking for initial module
            if i == 1:  # (first module in named_modules is the entire network, itself
                self.f_hook_feature_ = m.register_forward_pre_hook(self.f_pre_hook)
            if name == self.layer:
                self.f_hook_feature_ = m.register_forward_hook(self.f_hook_feature)
                self.feature_m_name = name
                self.feature_module = m

    def f_hook_feature(self, module, input, output):
        """
        Defines the forward hook for the conv2d module whose features we care about
        :param module: the conv2d module
        :param input: conv2d input
        :param output: conv2d output
        :return:
        """

        """
        # Used to assist initial receptive field calculations for a particular layer
        output[0, self.idx, 0, 0].backward(retain_graph=True)
        grad = self.current_input.grad[batch_idx, self.idx]
        print(np.where(grad.cpu() != 0))
        """

        """
        # Used to check which filter has the max activation for the current batch features
        # Effectively useless since high activation early on does not correlate with useful feature encodings
        out, idx = output.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0].max(dim=0)
        """

        out = output[:, self.idx]
        out1, idx1 = out.max(dim=0)
        out2, idx2 = out1.max(dim=0)
        out3, idx3 = out2.max(dim=0)

        max_activation = out3
        batch_idx, x_idx, y_idx = idx1[(idx2[idx3], idx3)].item(), idx2[idx3].item(), idx3.item()
        activation_idx = x_idx, y_idx
        self.current_input = self.get_img(self.current_input[batch_idx])

        if max_activation <= self.max_activation:
            self.current_input = None
            return None

        # update current maximally activating feature data
        self.max_activation = max_activation.item()
        self.max_rf_input, r_field_idx = self.compute_rf(activation_idx)

        plt.imshow(self.max_rf_input)
        plt.show()

        # outline receptive field on original image
        x_slice, y_slice = r_field_idx
        self.current_input[x_slice, np.array([y_slice.start, y_slice.stop - 1])] = np.array([1.0, 0.0, 0.0])
        self.current_input[np.array([x_slice.start, x_slice.stop - 1]), y_slice] = np.array([1.0, 0.0, 0.0])

        plt.imshow(self.current_input)
        plt.show()

        self.max_full_img = self.current_input
        self.current_input = None

    def f_pre_hook(self, module, input):
        """
        Defines forward hook for the first module of the network, whose job is saving the input image
        :param module: the first network module
        :param input: the image to save
        :param output: the module output
        :return:
        """
        input = input[0]
        self.current_input = input
        input.requires_grad = True

        return (input,)