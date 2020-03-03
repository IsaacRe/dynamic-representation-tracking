import torch
from scipy.stats import f_oneway
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ANOVATracker:
    def __init__(self, layer_names, save_file, loader=None, device=0, n_classes=20, grad=False, normalize=True,
                 mean_fstat=False):
        self.model = None
        self.mean = mean_fstat
        self.layers = layer_names
        self.save_file = save_file
        self.iters = 0
        self.n_classes = n_classes
        self.normalize = normalize
        self.grad = grad

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.f_stat_threshold = 500

        self.curr_class_acts = {l: [] for l in self.layers}
        self.max_acts_by_class = {l: [[] for _ in range(n_classes)] for l in layer_names}
        self.classes = []
        self.batch_labels = None
        self.p_values = {l: [] for l in layer_names}

        self.current_activation_mean = []
        self.current_activation_var = []

        self.device = device
        self.hook_var = {l: None for l in layer_names}
        self.m_names = []
        self.modules = []
        self.m_name_lookup = {}

        #debug
        self.raw_grad = {}

        self.loader = loader

    def reset_stats(self):
        self.max_acts_by_class = {l: [[] for _ in range(self.n_classes)] for l in self.layers}
        self.current_activation_mean = []

    def consolidate_activations(self):
        for l in self.layers:
            for c in range(self.n_classes):
                self.max_acts_by_class[l][c] = np.concatenate(self.max_acts_by_class[l][c])

    def save(self, curr_class):
        self.consolidate_activations()
        max_acts_by_class = {}
        curr_act_mean = np.stack(self.current_activation_mean).mean()
        curr_act_var = np.stack(self.current_activation_var).mean()
        for l in self.layers:
            max_acts_by_class[l] = []
            for c in range(self.n_classes):
                if self.normalize:
                    max_acts_by_class[l] += [(self.max_acts_by_class[l][c] - curr_act_mean) / curr_act_var]
                else:
                    max_acts_by_class[l] += [self.max_acts_by_class[l][c]]

            # get layer's p-value

            # single feature ANOVA
            F_stat, p_val = f_oneway(*max_acts_by_class[l])
            if self.mean:
                F_stat = F_stat.mean()
            self.p_values[l] += [F_stat]

            # mixed feature ANOVA
            """
            F_stat, p_val = f_oneway(*[m.flatten() for m in max_acts_by_class[l]])
            self.p_values[l] += [F_stat]
            """

            #threshold_mask = F_stat < self.f_stat_threshold
            #self.p_values[l] += [threshold_mask.sum()]

            #curr_class_mean = np.concatenate([max_acts_by_class[l][c] for c in curr_class]).mean()
            #other_class_mean = np.concatenate([max_acts_by_class[l][c] for c in range(self.n_classes) if c not in curr_class]).mean()


            #self.curr_class_acts[l] += [curr_class_mean - other_class_mean]

        #np.savez(self.save_file.split('.')[0] + 'curr_class_act.npz', **{l: np.stack(self.curr_class_acts[l]) for l in self.layers})
        np.savez(self.save_file, **{l: np.stack(self.p_values[l]) for l in self.layers})
        self.reset_stats()

    def setup_hooks(self):
        assert hasattr(self.model, 'named_modules'), "Model does not have named_modules() method"
        for i, (name, m) in enumerate(self.model.named_modules()):
            # setup tracking for initial module
            if name in self.layers:
                if self.grad:
                    self.hook_var[name] = m.register_backward_hook(self.b_hook)
                else:
                    self.hook_var[name] = m.register_forward_hook(self.f_hook)
                self.m_names += [name]
                self.modules += [m]
                self.m_name_lookup[m] = name
        assert self.hook_var is not None

    def f_hook(self, module, input, output):
        assert self.batch_labels is not None, 'self.batch_labels was never assigned'
        layer = self.m_name_lookup[module]
        mean = output.mean()

        self.current_activation_mean += [mean.cpu().detach().numpy()]
        # estimate variance
        n_samples = output.flatten().shape[0]
        self.current_activation_var += [((output - mean)**2).mean().cpu().detach().numpy() * n_samples / (n_samples - 1)]

        # save max activation value
        if len(output.shape) > 2:
            max_acts = output.max(dim=2)[0].max(dim=2)[0].cpu().detach().numpy()  # [batch X features]
        else:
            max_acts = output.cpu().detach().numpy()

        # group activations by class
        for c in range(self.n_classes):
            class_idxs = np.where(self.batch_labels.cpu().detach().numpy() == c)
            self.max_acts_by_class[layer][c] += [max_acts[class_idxs]]

    def b_hook(self, module, grad_in, grad_out):
        grad_out, = grad_out
        assert self.batch_labels is not None, 'self.batch_labels was never assigned'
        layer = self.m_name_lookup[module]
        mean = grad_out.mean()

        # TODO
        self.raw_grad[layer] = grad_out
        return
        self.current_activation_mean += [mean.cpu().numpy()]
        n_samples = grad_out.flatten().shape[0]
        self.current_activation_var += [((grad_out - mean)**2).mean().cpu().numpy() * n_samples / (n_samples - 1)]

        # TODO use patch act coords instead of max
        # save max gradients
        if len(grad_out.shape) > 2:
            max_grad = grad_out.max(dim=2)[0].max(dim=2)[0].cpu().numpy()  # [batch X features]
        else:
            max_grad = grad_out.cpu().numpy()

        # group grads by class
        for c in range(self.n_classes):
            class_idxs = np.where(self.batch_labels.cpu().numpy() == c)
            self.max_acts_by_class[layer][c] += [max_grad[class_idxs]]

    def gather(self, model, curr_class=[0], loader=None):
        """
        Gather class-feature activation p_values for the network layer specified at initialization
        :return: None
        """
        if not self.loader:
            assert loader, 'DataLoader was not specified'
            self.loader = loader

        self.model = model
        self.setup_hooks()

        for indices, images, labels in tqdm(self.loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # set self.batch_labels
            self.batch_labels = labels

            context = NullContext()
            if not self.grad:
                context = torch.no_grad()
            with context:
                out = self.model(images)

                if self.grad:
                    for c in range(self.n_classes):
                        out[:, c].sum().backward(retain_graph=c + 1 < self.n_classes)
                        model.zero_grad()

        self.save(curr_class)
        self.iters += 1
        self.remove_hook()
        self.model = None

    def remove_hook(self):
        for l in self.layers:
            self.hook_var[l].remove()
            self.hook_var[l] = None


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

