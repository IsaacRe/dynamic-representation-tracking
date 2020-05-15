import numpy as np
import torch
from vc_utils.activation_tracker import ActivationTracker
from vc_utils.hook_utils import find_network_modules_by_name


class Pruner(ActivationTracker):

    def __init__(self, network, *module_names, prune_ratio=0.8, load_pruned_path=None, store_on_gpu=False,
                 save_pruned_path=None, hook_manager=None):
        super(Pruner, self).__init__(module_names=module_names, network=network, store_on_gpu=store_on_gpu,
                                     hook_manager=hook_manager)
        self.prune_ratio = prune_ratio
        self.prune_mask = {}
        self.save_pruned_path = save_pruned_path
        if module_names:
            self.register_modules_for_pruning_by_name(*module_names)
        if load_pruned_path:
            self.load_prune_mask(load_pruned_path)
            for m in module_names:
                assert m in self.prune_mask, 'Module %s was not found in prune_mask loaded from %s' % (m,
                                                                                                       load_pruned_path)
            for m in self.prune_mask:
                assert m in module_names, 'Module %s was found in prune_mask loaded from %s, but was not passed' \
                                          ' to constructor' % (m, load_pruned_path)

    def prune(self, module_name, out, prune_mask=None, keep_shape=True):
        if prune_mask is None:
            prune_mask = self.prune_mask[module_name]

        out = out.clone()
        if prune_mask is not None:
            if keep_shape:
                out[:, prune_mask[module_name]] = 0.
            else:
                out = out[:, np.bitwise_not(prune_mask)]

        return out

    #########################  Hooks  ########################################################

    def prune_hook(self, module, inp, out):
        return self.prune(module.name, out)

    #########################  Pruning  #######################################################

    @staticmethod
    def compute_variance(node_activations):
        return ((node_activations - node_activations.mean(dim=1)[:, None]) ** 2).mean(dim=1)

    def save_prune_mask(self, path):
        np.savez(path, **self.prune_mask)

    def load_prune_mask(self, path):
        self.prune_mask = {m: file for m, file in np.load(path).items()}

    def sort_filter_activations_by_metric(self, activations):
        metric = self.compute_variance(self.flatten_activations(activations))
        return metric, sorted([(v, i) for i, v in enumerate(metric)], reverse=True)

    def compute_prune_mask_from_activations(self, activations, prune_ratio=None, save=False):
        if prune_ratio is None:
            prune_ratio = self.prune_ratio

        for module_name in self.modules:
            metric, metric_sort = self.sort_filter_activations_by_metric(activations[module_name])

            # Sort by variance and prune
            cutoff = int(len(metric) * (1 - prune_ratio))
            cutoff_val = metric_sort[cutoff][0]
            self.prune_mask[module_name] = (metric < cutoff_val).numpy()

        if self.save_pruned_path and save:
            self.save_prune_mask(self.save_pruned_path)

    def compute_prune_mask_from_data(self, loader, device=0, prune_ratio=0.8, compute_metric=compute_variance,
                                     save=False):
        activations = self.compute_activations_from_data(loader, device=device)
        self.compute_prune_mask_from_activations(activations, prune_ratio=prune_ratio, compute_metric=compute_metric,
                                                 save=save)

    #########################  Hook Registration  #############################################

    def register_modules_for_pruning(self, *modules, **named_modules):
        self.hook_manager.register_forward_hook(self.prune_hook, *modules, hook_fn_name='Pruner.prune_hook',
                                                activate=False, **named_modules)
        for module in list(modules) + list(named_modules.values()):
            self.prune_mask[module.name] = None

    def register_modules_for_pruning_by_name(self, *module_names, network=None):
        if network is None:
            network = self.network
        assert network is not None, 'To register a module by name, network object must be provided at initialization' \
                                    'or at function call'
        modules = find_network_modules_by_name(network, module_names)
        self.register_modules_for_pruning(**{name: module for name, module in zip(module_names, modules)})

    #########################  Context Management  ############################################

    def prune_all_context(self):
        return self.hook_manager.hook_all_context(hook_types=[self.prune_hook])
