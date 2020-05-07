import torch
import numpy as np
from vc_utils.activation_tracker import ActivationTracker
from vc_utils.hook_utils import find_network_modules_by_name


class ConvActivationThreshold(ActivationTracker):

    def __init__(self, network, *module_names, load_thresholds_path=None, store_on_gpu=False,
                 save_thresholds_path=None, hook_manager=None):
        super(ConvActivationThreshold, self).__init__(module_names=module_names, network=network,
                                                      store_on_gpu=store_on_gpu, hook_manager=hook_manager)
        self.thresholds = {}
        self.save_thresholds_path = save_thresholds_path
        self.register_modules_for_thresholding_by_name(*module_names)
        if load_thresholds_path:
            self.load_thresholds(load_thresholds_path)
            for m in module_names:
                assert m in self.thresholds, 'Prune mask was not found for module %s in file loaded from %s' \
                                             % (m, load_thresholds_path)
            for m in self.thresholds:
                assert m in module_names, 'Prune mask for module %s was found in file loaded from %s,' \
                                          'but no module with name %s was passed to constructor' % \
                                          (m, load_thresholds_path, m)

    def threshold(self, module_name, out, thresholds=None):
        if thresholds is None:
            thresholds = self.thresholds[module_name]

        out = out.clone()
        if thresholds is not None:
            absent_idxs = np.where(out.data.cpu().numpy() < thresholds[None, :, None, None])
            present_idxs = np.where(out.data.cpu().numpy() >= thresholds[None, :, None, None])
            out[absent_idxs] = 0.
            out[present_idxs] = 1.
        return out

    ##########################  Hooks  #######################################################

    def threshold_hook(self, module, inp, out):
        return self.threshold(module.name, out)

    ##########################  Thresholding  ################################################

    @staticmethod
    def get_quartile_means(activations, q):
        means = -np.ones((activations.shape[0],))
        for i, act in enumerate(activations):
            mean = act.mean()
            for j in range(q):
                act = act[act > mean]
                mean = act.mean()
            means[i] = mean
        assert not np.any(means == -1)
        return means

    def save_thresholds(self, path):
        np.savez(path, **self.thresholds)

    def load_thresholds(self, path):
        self.thresholds = {m: file for m, file in np.load(path).files()}

    def compute_thresholds_from_activations(self, activations, q=1, save=False):
        for module_name in self.modules:
            activations = self.flatten_activations(activations[module_name])
            self.thresholds[module_name] = self.get_quartile_means(activations, q)

        if self.save_thresholds_path and save:
            self.save_thresholds(self.save_thresholds_path)

    def compute_thresholds_from_data(self, loader, device=0, q=1, save=False):
        activations = self.compute_activations_from_data(loader, device=device)
        self.compute_thresholds_from_activations(activations, q=q, save=save)

    ##########################  Hook Registration  ##########################################

    def register_modules_for_thresholding(self, *modules, **named_modules):
        self.hook_manager.register_forward_hook(self.threshold_hook, *modules, hook_fn_name='ConvActivationThreshold.threshold_hook',
                                   activate=False, **named_modules)
        for module in list(modules) + list(named_modules.values()):
            self.thresholds[module.name] = None

    def register_modules_for_thresholding_by_name(self, *module_names, network=None):
        if network is None:
            network = self.network
        assert network is not None, 'To register a module by name, network object must be provided at initialization' \
                                    'or at function call'
        modules = find_network_modules_by_name(network, module_names)
        self.register_modules_for_thresholding(**{name: module for name, module in zip(module_names, modules)})

    ##########################  Context Management  #########################################

    def threshold_all_context(self):
        return self.hook_manager.hook_all_context(hook_types=[self.threshold_hook])
