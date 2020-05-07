import torch
import numpy as np
from time import time
from vc_utils.hook_utils import HookManager, find_network_modules_by_name, data_pass


class ActivationTracker:

    def __init__(self, modules=None, module_names=None, network=None, store_on_gpu=False, save_file=None,
                 hook_manager=None):
        if not save_file:
            self.save_file = 'activations_' + str(time()) + '.npz'
        self.network = network
        self.hook_manager = hook_manager if hook_manager else HookManager()
        self.activations = {}
        self.store_on_gpu = store_on_gpu
        if modules and len(modules) > 0:
            if module_names:
                assert len(modules) == len(module_names), 'Both modules and module_names were provided,' \
                                                          ' but differ in length'
                named_modules = {name: module for name, module in zip(module_names, modules)}
                self.register_modules_for_tracking(**named_modules)
            else:
                self.register_modules_for_tracking(*modules)
        elif module_names and len(module_names) > 0:
            self.register_modules_for_tracking_by_name(*module_names)

    @property
    def modules(self):
        return self.hook_manager.name_to_module

    @staticmethod
    def flatten_activations(activations):
        # if outputs are featuremaps aggregate over spatial dims and sample dim
        if len(activations.shape) == 4:
            return activations.transpose(1, 0).flatten(start_dim=1, end_dim=3)
        # if outputs are vectors only need to aggregate over sample dim
        elif len(activations) == 2:
            return activations.transpose(1, 0)
        else:
            raise TypeError('Output type unknown for Tensor: \n%s' % repr(activations))

    def get_module_activations(self, module_name, cpu=True):
        acts = torch.cat(self.activations[module_name], dim=0)
        if cpu:
            acts = acts.cpu()
        return acts

    def get_all_activations(self):
        return {m: self.get_module_activations(m) for m in self.activations}

    def reset_module_activations(self, module_name):
        self.activations[module_name] = []

    def reset_all_activations(self):
        for m in self.activations:
            self.activations[m] = []

    def save_activations(self):
        np.savez(self.save_file, **self.get_all_activations())

    def compute_activations_from_data(self, loader, network=None, device=0, save=False):
        if network is None:
            network = self.network
        assert network is not None, 'Network must be provided at initialization or as parameter to data_pass'

        with self.track_all_context(save=save, reset=True):
            data_pass(loader, network, device=device, gradient=False)
            return self.get_all_activations()

    ######################  Hooks  ####################################################

    def track_hook(self, module, inp, out):
        out = out.data
        if not self.store_on_gpu:
            out = out.cpu()
        self.activations[module.name] += [out]

    ######################  Hook Registration  ########################################

    def register_modules_for_tracking(self, *modules, **named_modules):
        self.hook_manager.register_forward_hook(self.track_hook, *modules, hook_fn_name='ActivationTracker.track_hook',
                                   activate=False, **named_modules)
        for module in list(modules) + list(named_modules):
            self.activations[module] = []

    def register_modules_for_tracking_by_name(self, *module_names, network=None):
        if network is None:
            network = self.network
        assert network is not None, 'To register a module by name, network object must be provided at initialization' \
                                    'or at function call'
        modules = find_network_modules_by_name(network, module_names)
        self.register_modules_for_tracking(**{name: module for name, module in zip(module_names, modules)})

    ######################  Context Management  ########################################

    def track_all_context(self, save=False, reset=True):
        enter_fns = []
        exit_fns = []
        if save:
            exit_fns += [self.save_activations]
        if reset:
            exit_fns += [self.reset_all_activations]
        return self.hook_manager.hook_all_context(hook_types=[self.track_hook],
                                                  add_enter_fns=enter_fns,
                                                  add_exit_fns=exit_fns)
