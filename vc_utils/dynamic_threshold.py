import torch
from tqdm.auto import tqdm
from vc_utils.hook_utils import find_network_modules_by_name, data_pass
from vc_utils.activation_tracker import ActivationTracker
from gumbel_softmax import gumbel_softmax_binary


def sigmoid_with_temperature(logits, t):
    return torch.sigmoid(logits / t)


class ThresholdLearner(ActivationTracker):

    def __init__(self, network, module_name, hook_manager=None, device=0, init_t=0.5,
                 temperature=1.0, lr=0.01):
        [module] = find_network_modules_by_name(network, [module_name])
        super(ThresholdLearner, self).__init__(modules=[module], module_names=[module_name], network=network,
                                               hook_manager=hook_manager)
        self.network = network
        self.module_name = module_name
        self.module = module
        self.register_detach_hook()
        self.register_soft_threshold_hook()
        self.register_hard_threshold_hook()

        # hyperparams
        self.device = device
        self.init_t = init_t
        self.temperature = temperature

        self.cache_thresholded_activations = False
        self.thresholded_activations = []
        self.threshold = torch.nn.Parameter(torch.zeros(1).fill_(init_t).to(device))
        self.lr = lr
        self.t_optim = torch.optim.SGD([self.threshold], lr=lr)

    def reset_all_activations(self):
        super(ThresholdLearner, self).reset_all_activations()
        self.thresholded_activations = []
        self.cache_thresholded_activations = False

    def set_threshold(self, t):
        self.init_t = t
        self.threshold.data.fill_(t)

    def set_lr(self, lr):
        self.lr = lr
        self.t_optim = torch.optim.SGD([self.threshold], lr=lr)

    def threshold_activations_gumbel(self, activations, hard=False):
        logits = activations - self.threshold
        if hard:
            ret = logits.clone()
            ret[logits >= 0] = 1.0
            ret[logits < 0] = 0.0
            return ret
        return gumbel_softmax_binary(logits, self.temperature, device=self.device)

    def threshold_activations(self, activations, hard=False):
        logits = activations - self.threshold
        if hard:
            ret = logits.clone()
            ret[logits >= 0] = 1.0
            ret[logits < 0] = 0.0
            return ret
        return sigmoid_with_temperature(logits, self.temperature)

    def get_thresholded_activations(self, cpu=True):
        acts = torch.cat(self.thresholded_activations, dim=0)
        if cpu:
            acts = acts.cpu()
        return acts

    ###############  Hook registration  #################################

    def register_detach_hook(self):
        named_module = {self.module_name: self.module}
        self.hook_manager.register_forward_hook(self.detach_hook,
                                                hook_fn_name='ThresholdLearner.detach_hook',
                                                activate=False,
                                                **named_module)

    def register_soft_threshold_hook(self):
        named_module = {self.module_name: self.module}
        self.hook_manager.register_forward_hook(self.soft_threshold_hook,
                                                hook_fn_name='ThresholdLearner.soft_threshold_hook',
                                                activate=False,
                                                **named_module)

    def register_hard_threshold_hook(self):
        named_module = {self.module_name: self.module}
        self.hook_manager.register_forward_hook(self.hard_threshold_hook,
                                                hook_fn_name='ThresholdLearner.hard_threshold_hook',
                                                activate=False,
                                                **named_module)

    ###############  Hooks  #############################################

    def detach_hook(self, module, inp, out):
        return out.detach()

    def base_threshold_hook(self, module, inp, out, hard=False):
        acts = self.threshold_activations(out.detach(), hard=hard)
        if self.cache_thresholded_activations:
            self.thresholded_activations += [acts.data.clone().cpu()]
        return acts

    def soft_threshold_hook(self, *args):
        return self.base_threshold_hook(*args, hard=False)

    def hard_threshold_hook(self, *args):
        return self.base_threshold_hook(*args, hard=True)

    ###############  Contexts  ##########################################

    def threshold_context(self, threshold=True, hard=False, cache_thresholded_acts=False, cache_raw_acts=False,
                          reset=True):
        hooks = []
        exit_fns = []
        if cache_raw_acts:
            hooks += [self.track_hook]
        if reset:
            exit_fns += [self.reset_all_activations]

        # resolve functionality to be performed by hook methods
        self.cache_thresholded_activations = cache_thresholded_acts
        if not threshold:
            t_hook = self.detach_hook
        elif hard:
            t_hook = self.hard_threshold_hook
        else:
            t_hook = self.soft_threshold_hook

        # ensure predict_hook is activated after track_hook so the non-threhsolded activations are recorded
        hooks += [t_hook]
        return self.hook_manager.hook_all_context(hook_types=hooks, add_exit_fns=exit_fns)

    #####################################################################

    def train(self, loader, epochs=1, lr_network=0.005, lr_thresholds=None, threshold=True, hard=False):
        """
        Fit the network's downstream layers (after thresholding is applied) to the thresholded output
        :param loader: data loader for the training data
        :param epochs: number of epochs to train for
        :param lr_network: learning rate for training of network parameters
        :param lr_thresholds: learning rate to use for fitting thresholds
        :param hard_threshold: if True, hard thresholding will be used during training of modeling paramters
        :return: None
        """
        # determine whether to train thresholds
        fit_threshold = threshold and (lr_thresholds is None or lr_thresholds > 0)
        if fit_threshold:
            # if training the thresholds, soft-thresholding must be used
            hard = False

        net_params = []
        net_names, net_modules = zip(*self.network.named_modules())
        for m in net_modules[net_names.index(self.module_name) + 1:]:
            net_params += list(m.parameters())
        net_optim = torch.optim.SGD(net_params, lr=lr_network)
        if lr_thresholds is not None and fit_threshold:
            self.set_lr(lr_thresholds)
        ce_loss = torch.nn.CrossEntropyLoss()

        pbar = tqdm(total=epochs * len(loader))
        with self.threshold_context(threshold=threshold, hard=hard):
            for e in range(epochs):
                losses = []
                for i, (idx, x, y) in enumerate(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.network(x)

                    loss = ce_loss(out, y)
                    losses += [loss.item() * len(y)]

                    loss.backward()
                    net_optim.step()
                    net_optim.zero_grad()
                    if fit_threshold:
                        self.t_optim.step()
                        self.t_optim.zero_grad()

                    if e == 0 and i == 0:
                        print('Loss before epoch 0: %.5f' % loss.item())
                    pbar.update(1)
                print('Avg loss for epoch %d: %.5f' % (e, sum(losses) / len(loader.dataset)))
        pbar.close()

    def test(self, loader, threshold=True, hard=True):
        total = correct = 0
        with torch.no_grad():
            with self.threshold_context(threshold=threshold, hard=hard):
                for i, x, y in tqdm(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.network(x)

                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)

        print('Model accuracy after binarization of features: %d/%d (%.2f)' % (correct, total, correct / total * 100.))
        return correct / total * 100.

    def predict(self, loader, output_raw=False):
        with torch.no_grad():
            with self.threshold_context(hard=True, cache_thresholded_acts=True, cache_raw_acts=output_raw):
                data_pass(loader, self.network, device=self.device, gradient=False)
                raw = None
                if output_raw:
                    raw = self.get_module_activations(self.module_name)
                return raw, self.get_thresholded_activations()
