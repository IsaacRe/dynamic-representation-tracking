import numpy as np
import torch
from torch.optim import SGD
import matplotlib.pyplot as plt


class FeatureVis:
    """
    Class to handle visualization of maximally activating receptive field inputs for a particular conv filter
    """

    def __init__(self, model, layer_name, idx):
        self.model = model
        #self.model.apply(self.disable_batchnorm)
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
        self.setup_hooks()

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

    def enable_batchnorm(self, module):
        """
        Enable passed batchnorm layer. Used only to assist calculation of layers' receptive fields
        :param module: module
        :return:
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()

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
        assert hasattr(self.model, 'named_modules'), "Model does not have modules() method"
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