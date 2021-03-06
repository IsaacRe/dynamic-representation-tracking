import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.models as models
from utils.loader_utils import CustomRandomSampler
from utils.loader_utils import CustomBatchSampler
from utils.model_utils import kaiming_normal_init
from utils.model_utils import MultiClassCrossEntropyLoss
import os
import pickle

import pdb

def sort_fc(model, classes_seen, model_classes_seen):
    fc = model.fc
    out_dim = fc.out_features

    classes_map = get_classes_map(out_dim, classes_seen, model_classes_seen)
    w = fc.weight.data
    fc.weight.data = w[classes_map]


def get_classes_map(total_classes, classes_seen, model_classes_seen):
    classes = set()
    classes_map = np.zeros(total_classes)
    for c, mc in zip(classes_seen.flatten(), model_classes_seen.flatten()):
        if c in classes:
            continue
        classes = classes.union({c})
        classes_map[c] = mc

    return classes_map


class IncrNet(nn.Module):
    def __init__(self, args, device, cifar=False):
        self.debug = args.debug
        #Task
        self.cifar = cifar
        # Hyper Parameters
        self.init_lr = args.init_lr
        self.init_lr_ft = args.init_lr_ft
        self.num_epoch = args.num_epoch
        self.num_epoch_ft = args.num_epoch_ft
        self.batch_size = args.batch_size
        self.lr_dec_factor = args.lrd
        self.llr_freq = args.llr_freq
        self.weight_decay = args.wd
        self.should_prune = args.should_prune
        self.final_prune = args.final_prune
        self.prune_save_all_dir = args.prune_save_all_dir
        self.prune_final_iter = args.prune_final_iter
        self.save_all_dir = args.save_all_dir
        self.mask_dict = None

        # Number of exemplars
        self.num_explrs = args.num_exemplars

        # Hardcoded
        if not self.cifar:
            self.lower_rate_epoch = [
                int(0.7 * self.num_epoch), int(0.9 * self.num_epoch)]
        else:
            self.lower_rate_epoch = []
        self.momentum = 0.9

        self.pretrained = args.pretrained
        self.dist = args.dist
        self.algo = args.algo
        self.epsilon = 1e-16
        self.aug = args.aug

        # whether to load model from pretrained model
        self.file_path = args.file_path
        # Whether to use fixed (loaded from pretrained model) exemplar set
        self.fixed_ex = args.fixed_ex
        # Whether to use a pretrained model from the same task
        #self.ptr_model = args.ptr_model
        # Network architecture
        super(IncrNet, self).__init__()
        if len(self.file_path) == 0:
            self.model = models.resnet34(pretrained=self.pretrained)
        else:
            model_path = "%s-model.pth.tar" %os.path.splitext(self.file_path)[0]
            classes_path = "%s-classes.npz" %os.path.splitext(self.file_path)[0]
            print('Loading pretrained model from: ', self.file_path)
            mdl = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model = mdl.model
            if self.fixed_ex:
                self.exemplar_sets_full = mdl.exemplar_sets
                self.exemplar_bbs_full = mdl.exemplar_bbs
                self.eset_le_maps_full = mdl.eset_le_maps
                self.ex_class_id_map = {cl: idx for (cl, idx) in zip(np.load(classes_path)['classes_seen'], np.load(classes_path)['model_classes_seen'])}
                # Train a new model with random weights
                #if not self.ptr_model:
                self.model = models.resnet34(pretrained=self.pretrained)

        if not self.pretrained and not self.file_path:
            self.model.apply(kaiming_normal_init)
        feat_size = self.model.fc.in_features
        self.model.fc = nn.Linear(feat_size, 1, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-1])
        # GPU device for the model
        self.device = device

        # n_classes incremented before processing new data in an iteration.
        # n_known is set to n_classes after all data for a learning exposure
        # has been processed
        self.n_classes = 0
        self.n_known = 0
        # map for storing index into self.classes
        self.classes_map = {}
        # stores classes in the order in which they are seen without repetitions
        self.classes = []
        self.n_occurrences = []

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        # N = number of exemplars per class
        # C = num channels
        # H = image height
        # W = image width
        self.exemplar_sets = []
        # for each exemplar store which learning exposure it came from
        self.eset_le_maps = []
        if not self.cifar:
            # store bounding boxes for all exemplars
            self.exemplar_bbs = []        
        # Boolean to store whether exemplar means need to be recomputed
        self.compute_means = True
        # Means of exemplars
        self.exemplar_means = []

        # sampling option

        self.sample = args.sample

        # Cross Entropy Loss functions
        self.loss = args.loss
        if self.sample == 'wg':
            self.bce_loss = nn.BCELoss(reduction='none')
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.bce_loss = nn.BCELoss()
            self.ce_loss = nn.CrossEntropyLoss()

        # Temperature for cross entropy distillation loss
        self.T = 2

        # std for gradient noise adding
        self.std = np.array([np.sqrt(0.3/(epoch+2)**0.55) 
            for epoch in range(np.max([self.num_epoch, self.num_epoch_ft]))])

        # random exemplar option
        self.random_exemplar = True
        self.batch_pt = False

        # for saving exposure learning stats
        self.exposure_stats = []

    def children(self):
        """
        For compatibility with CorrelationTracker
        :return: Generator<torch.module> returned from model.children()
        """
        return self.model.children()

    def from_resnet(self, model_file, load_fc=False):
        self.batch_pt = True
        model = torch.load(model_file,
                           map_location=lambda storage, loc: storage)
        if type(model) == IncrNet:
            # if we've loaded an IncrNet model, just get the resnet model
            model = model._modules['model']
        self.model = model
        if load_fc:
            self.fc = self.model.fc
        else:
            self.model.fc = self.fc
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def increment_classes(self, new_classes):
        '''
        Add new output nodes when new classes are seen and make changes to
        model data members
        '''
        n = len(new_classes)
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        if self.n_known == 0:
            new_out_features = n
        else:
            new_out_features = out_features + n

        self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
        self.fc = self.model.fc

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='sigmoid')
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

        for i, cl in enumerate(new_classes):
            self.classes_map[cl] = self.n_known + i
            self.classes.append(cl)

    def compute_exemplar_means(self, mean_image, img_size):
        '''
        Compute exemplar means from exemplar_sets
        '''
        exemplar_means = []
        for y, P_y in enumerate(self.exemplar_sets):
            # normalize images
            if not self.cifar:
                bbs = self.exemplar_bbs[y]
                data_means = np.array([cv2.resize(mean_image[b[2]:b[3], b[0]:b[1]],\
                    (img_size, img_size)).transpose(2, 0, 1) for b in bbs])
            elif self.cifar:
                
                data_means = np.array([mean_image]*P_y.shape[0])
            P_y = (np.float32(P_y) - data_means)/255.

            # Concatenate with horizontally flipped images
            all_imgs = np.concatenate((P_y, P_y[:, :, :, ::-1]), axis=0)
            all_features = []

            # Batch up all_imgs to fit on the GPU
            # all_features on GPU
            for i in range(0, len(all_imgs), self.batch_size):
                # Extract image features
                img_tensors = Variable(
                    torch.FloatTensor(
                        all_imgs[i:min(i+self.batch_size, len(all_imgs))]))\
                            .cuda(device=self.device)
                features = self.feature_extractor(img_tensors)
                del img_tensors

                # Normalize features
                features_norm = features.data.norm(p=2, dim=1) + self.epsilon
                features_norm = features_norm.unsqueeze(1)
                features.data = features.data.div(
                    features_norm.expand_as(features))  # Normalize

                all_features.append(features)

            # compute mean feature vector and renormalize
            features = torch.cat(all_features)
            mu_y = features.mean(dim=0).squeeze().detach()
            mu_y.data = mu_y.data / \
                (mu_y.data.norm() + self.epsilon)  # Normalize
            exemplar_means.append(mu_y.cpu())

            del features

        self.exemplar_means = exemplar_means
        self.compute_means = False

    def classify(self, x):
        '''
        Args:
            x: input images
        Returns:
            preds: Tensor of size (x.shape[0],)
        '''
        return self.classify_network(x)

    def classify_ncm(self, x, mean_image, img_size):
        # if we have no exemplars we cannot conduct ncm classification
        if self.exemplar_sets[0].shape[0] == 0:
            return torch.Tensor([-1] * x.shape[0]).cuda(self.device)
        batch_size = x.size(0)

        if self.compute_means:
            self.compute_exemplar_means(mean_image, img_size)

        # Expand means to compute distance to each mean 
        # for each feature vector
        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means).cuda(
            device=self.device)  # (n_classes, feature_size)
        # (batch_size, n_classes, feature_size)
        means = torch.stack([means] * batch_size)
        # (batch_size, feature_size, n_classes)
        means = means.transpose(1, 2)

        # Expand features to find distance from each mean
        feature = self.feature_extractor(x)  # (batch_size, feature_size)
        feature_norm = feature.data.norm(p=2, dim=1) + self.epsilon
        feature_norm = feature_norm.unsqueeze(1)
        feature.data = feature.data.div(feature_norm.expand_as(feature))
        feature = feature.squeeze(3)  # (batch_size, feature_size, 1)
        # (batch_size, feature_size, n_classes)
        feature = feature.expand_as(means)

        # (batch_size, n_classes)
        dists = (feature - means).pow(2).sum(1).squeeze()

        if len(dists.data.shape) == 1:  # Only one output node right now
            preds = Variable(torch.LongTensor(
                np.zeros(dists.data.shape, 
                         dtype=np.int64)).cuda(device=self.device))
        else:
            # predict class based on closest exemplar mean
            _, preds = dists.min(1)

        return preds

    def classify_network(self, x):
        _, preds = torch.max(torch.softmax(self.forward(x), dim=1), 
                                 dim=1, keepdim=False)
        return preds

    def train_fc(self, args, loader):
        optimizer = optim.SGD(self.fc.parameters(), lr=args.ft_fc_lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        with tqdm(total=len(loader)*args.ft_fc_epochs) as pbar:  # with tqdm(total=args.n*args.ft_fc_epochs) as pbar:
            for e in range(args.ft_fc_epochs):
                for k, (i, x, y) in enumerate(loader):
                    x, y = Variable(x).cuda(device=self.device), Variable(y).cuda(device=self.device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        x = self.feature_extractor(x)
                    x = x.view(x.size(0), -1)
                    out = self.fc(x)

                    loss = self.ce_loss(out, y)
                    loss = torch.mean(loss)

                    loss.backward(retain_graph=False)
                    optimizer.step()

                    """
                    loss_ = float(loss.data.item())
                    length = len(loader)
                    tqdm.write('FT-FC Epoch [%d/%d], Minibatch [%d/%d] Loss: %.4f'
                               % (e, args.ft_fc_epochs,
                                  i % length + 1,
                                  length, loss_))
                    """
                    pbar.update(1)

    def construct_exemplar_sets(self, images, image_means, le_maps,
                                image_bbs, m, cl, curr_iter, overwrite=False):
        '''
        Construct exemplar sets for each class given images
        Args:
            images: list containing np.array of images for each class
            image_means: cropped out parts of the dataset mean image for
                         image patches
            le_maps: list containing np.array of learning exposure indices and
                     frame indices within them of images for each class
            image_bbs: list containing np.array of image bounding boxes
                       for each class
            m: number of images in each exemplar set
            cl: class indices in self.classes of for each exemplar set
            curr_iter: learning exposure index
            overwrite: Boolean to keep track of whether the exemplar set is
                       being constructed after finetuning (not a repeated
                       exposure)
        '''
        if image_bbs is None:
            image_bbs = [None] * len(images)
        for c, image, le_map, image_bb in zip(cl, images, le_maps, image_bbs):
            self.construct_exemplar_set(image, image_means, le_map,
                                        image_bb, m, c, curr_iter, overwrite=overwrite)

    def construct_exemplar_set(self, images, image_means, le_maps, 
                               image_bbs, m, cl, curr_iter, overwrite=False):
        '''
        Construct an exemplar set given images
        Args:
            images: np.array containing images of a class
            image_means: cropped out parts of the dataset mean image for
                         image patches
            le_maps: np.array containing learning exposure indices and 
                     frame indices within them of images
            image_bbs: np.array containing image bounding boxes
            m: number of images in the exemplar set
            cl: class index in self.classes of the current exemplar set
            curr_iter: learning exposure index
            overwrite: Boolean to keep track of whether the exemplar set is 
                       being constructed after finetuning (not a repeated 
                       exposure) 
        '''
        if not self.random_exemplar:
            num_new_imgs = np.sum(le_maps[:, 0] == curr_iter)
            num_old_imgs = len(images) - num_new_imgs
            all_features = []

            # Normalize images
            normalized_images = (np.float32(images) - image_means)/255.

            # Batch up images to fit on GPU memory
            for i in range(0, len(images), self.batch_size):
                with torch.no_grad():
                    img_tensors = Variable(torch.FloatTensor(
                        normalized_images[i:min(i+self.batch_size, len(images))])
                                           ).cuda(device=self.device)

                # Get features
                features = self.feature_extractor(img_tensors)
                del img_tensors
                
                # Normalize features
                features_norm = features.data.norm(p=2, dim=1) + self.epsilon
                features_norm = features_norm.unsqueeze(1)
                features.data = features.data.div(
                    features_norm.expand_as(features))  # Normalize
                features.data = features.data.squeeze(3)
                features.data = features.data.squeeze(2)
                features = features.data.cpu().numpy()

                all_features.append(features)

            features = np.concatenate(all_features, axis=0)

            # Weight of images for computing mean while herding
            # New (images from new learning exposure) and 
            # old images (from old exemplars) are weighed 
            # in the inverse ratio of their numbers 
            weights = np.zeros((len(features), 1))
            weights[le_maps[:, 0] == curr_iter] = float(
                num_old_imgs + 1)/(num_old_imgs + num_new_imgs + 1)
            weights[le_maps[:, 0] != curr_iter] = float(
                num_new_imgs + 1)/(num_old_imgs + num_new_imgs + 1)

            class_mean = np.sum(weights * features, axis=0)/np.sum(weights)
            class_mean = class_mean / \
                (np.linalg.norm(class_mean) + self.epsilon)  # Normalize

            indices_remaining = np.arange(0, len(images))
            indices_selected = []

            # Herding procedure : Algorithm 4 in 
            # https://arxiv.org/pdf/1611.07725.pdf (Rebuffi et al.)
            for k in range(m):
                if len(indices_remaining) == 0:
                    break

                if len(indices_selected) > 0:
                    S = np.sum(features[np.array(indices_selected)], axis=0)
                else:
                    S = 0

                phi = features[indices_remaining]
                mu = class_mean
                mu_p = 1.0/(k+1) * (phi + S)
                mu_p = mu_p / (np.linalg.norm(mu_p) + self.epsilon)
                i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

                indices_selected.append(indices_remaining[i])
                indices_remaining = np.delete(indices_remaining, i, axis=0)
        elif self.random_exemplar:
            num_new_imgs = np.sum(le_maps[:, 0] == curr_iter)
            num_old_imgs = len(images) - num_new_imgs
            weights = np.zeros(len(images), dtype=np.float32)
            weights[le_maps[:, 0] == curr_iter] = float(
                num_old_imgs + 1)/(num_old_imgs + num_new_imgs + 1)
            weights[le_maps[:, 0] != curr_iter] = float(
                num_new_imgs + 1)/(num_old_imgs + num_new_imgs + 1)
            # Make sure the weights sum to 1
            weights /= np.sum(weights)
            indices_selected = np.random.choice(len(images),min(len(images),m), p=weights, replace=False)


        if cl < len(self.exemplar_sets) or overwrite:
            # Repeated exposure, or balanced finetuning for E2EIL
            self.exemplar_sets[cl] = np.array(images[indices_selected])
            self.eset_le_maps[cl] = np.array(le_maps[indices_selected])
            if not self.cifar:
                self.exemplar_bbs[cl] = np.array(image_bbs[indices_selected])
            if not overwrite:
                self.n_occurrences[cl] += 1
        else:
            # New object exemplar set to be created
            self.exemplar_sets.append(np.array(images[indices_selected]))
            self.eset_le_maps.append(np.array(le_maps[indices_selected]))
            if not self.cifar:
                self.exemplar_bbs.append(np.array(image_bbs[indices_selected]))
            self.n_occurrences.append(1)

    def reduce_exemplar_sets(self, m):
        '''
        Shrink each exemplar set to size m, keeping only the 
        top m ranked by herding
        '''
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
            self.eset_le_maps[y] = self.eset_le_maps[y][:m]
            if not self.cifar:
                self.exemplar_bbs[y] = self.exemplar_bbs[y][:m]

    def reduce_exemplar_sets_full_explrs(self, n_explrs_class):
        '''
        Shrink each exemplar set to size m, keeping only the 
        top m ranked by herding
        '''
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:n_explrs_class[y]]
            self.eset_le_maps[y] = self.eset_le_maps[y][:n_explrs_class[y]]
            if not self.cifar:
                self.exemplar_bbs[y] = self.exemplar_bbs[y][:n_explrs_class[y]]

    def combine_dataset_with_exemplars(self, dataset):
        '''
        Add exemplars to dataset for training
        '''
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            if not self.cifar:
                dataset.append(exemplar_images, exemplar_labels,
                           self.exemplar_bbs[y], self.eset_le_maps[y])
            else:
                dataset.append(exemplar_images, exemplar_labels, self.eset_le_maps[y])

    def fetch_hyper_params(self):
        return {'num_epoch': self.num_epoch,
                'batch_size': self.batch_size,
                'lower_rate_epoch': self.lower_rate_epoch,
                'lr_dec_factor': self.lr_dec_factor,
                'init_lr': self.init_lr,
                'pretrained': self.pretrained,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay}
                
    def populate_with_previous_init(self):
        assert(self.final_prune)
        prev_model = "%s/model_iter_0.pth.tar" % self.prune_save_all_dir
        # prev_model = load_model(0, self.prune_save_all_dir)
        self.from_resnet(prev_model, load_fc=False)

    def get_final_mask_dict(self):
        if self.mask_dict is None:
            path = "%s/model_iter_%d.npz" % (self.prune_save_all_dir, self.prune_final_iter)
            md = {}
            mask_dict = np.load(path)
            for name, mask in mask_dict.items():
                reshaped_mask = mask.reshape(self.state_dict()[name].shape)
                reshaped_mask = np.invert(reshaped_mask)
                reshaped_mask = torch.from_numpy(reshaped_mask).to(self.device)
                md[name] = reshaped_mask
            self.mask_dict = md
            
        return self.mask_dict


        
    def get_mask_dict(self):
        if self.mask_dict is None:
            # get the dict
            mask_dict = np.load("%s/model_iter_%d.npz" % (self.save_all_dir, 0))
            for name, mask in mask_dict.items():
                reshaped_mask = mask.reshape(self.state_dict()[name].shape)
                reshaped_mask = np.invert(reshaped_mask)
                reshaped_mask = torch.from_numpy(reshaped_mask).to(self.device)
                mask_dict[name] = reshaped_mask
            self.mask_dict = mask_dict

        return self.mask_dict

    def update_representation_icarl(self, dataset, prev_model, 
                                    curr_class_idxs, num_workers):
        '''
        Update feature representation for icarl/lwf
        Args:
            dataset: torch.utils.data.Dataset object, the dataset to train on
            prev_model: model before backprop updates for obtaining distillation 
                        labels
            curr_class_idxs: indices in [0, self.n_known] of new classes.
                             also contains an index if its a repeated exposure 
                             to a class 
            num_workers: number of data loader threads to use
        '''
        torch.backends.cudnn.benchmark = True
        self.compute_means = True
        lr = self.init_lr

        # Form combined training set
        if self.algo == 'icarl':
            self.combine_dataset_with_exemplars(dataset)
        if self.aug == "e2e_full":
            dataset.get_augmented_set()


        # if self.sample == 'minibatch_sampling':
        #     weights = dataset.get_class_weights()
        # else:
        #     weights = None

        if self.sample == 'minibatch_sampling_inflate':
        	dataset.inflate_dataset()

        if self.sample == "wg":
            if not (self.cifar and self.n_classes == 1):
                print("UPDATING =>>>>")
                if self.loss == "BCE":
                    dataset.update_class_weights_bce()
                elif self.loss == "CE":
                    dataset.update_class_weights_ce()

        sampler = CustomRandomSampler(dataset, self.num_epoch, num_workers)
        batch_sampler = CustomBatchSampler(
            sampler, self.batch_size, drop_last=False, epoch_size=len(dataset))
        num_batches_per_epoch = batch_sampler.num_batches_per_epoch
        
        # Run network training
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=0 if self.debug else num_workers,
            pin_memory=True)

        class Namespace:
            pass
        args = Namespace()
        args.ft_fc_epochs = 1
        args.ft_fc_lr = 0.002
        args.n = num_batches_per_epoch
        #self.train_fc(args, loader)

        print("Length of current data + exemplars : ", len(dataset))
        optimizer = optim.SGD(self.parameters(), lr=lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)

        mean_loss = 0
        total = 0

        # label matrix
        q = Variable(torch.zeros(self.batch_size, self.n_classes)
                     ).cuda(device=self.device)
        with tqdm(total=num_batches_per_epoch*self.num_epoch) as pbar:
            for i, (indices, images, labels, weights) in enumerate(loader):
                epoch = i//num_batches_per_epoch

                if ((epoch+1) in self.lower_rate_epoch 
                        and i % num_batches_per_epoch == 0):
                    lr = lr * 1.0/self.lr_dec_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                images = Variable(images).cuda(device=self.device)

                optimizer.zero_grad()
                g = self.forward(images)
                if self.loss == 'BCE' or self.n_classes == 1:
                    g = torch.sigmoid(g)
                    q[:, :] = 0

                    if self.dist:
                        labels = labels.cuda(device=self.device)
                        if self.n_known > 0:
                            # Store network outputs with pre-update parameters
                            # for distillation
                            q_prev = torch.sigmoid(prev_model.forward(images))
                            q.data[:len(labels), :self.n_known] = q_prev.data[:, :self.n_known]

                        # For new classes use label 1
                        for curr_class_idx in curr_class_idxs:
                            q.data[:len(labels), curr_class_idx] = 0
                            q.data[:len(labels), curr_class_idx].masked_fill_(
                                labels == curr_class_idx, 1)
                    else:
                        labels = labels.cuda(device=self.device)
                        pos_labels = labels
                        pos_indices = torch.arange(0, g.data.shape[0], 
                            out=torch.LongTensor()).cuda(device=self.device)[pos_labels != -1]
                        pos_labels = pos_labels[pos_labels != -1]

                        if len(pos_indices) > 0:
                            q[pos_indices, pos_labels] = 1

                    loss = self.bce_loss(g, q[:len(labels)])
                elif self.loss == 'CE':
                    labels = labels.cuda(device=self.device)
                    loss = self.ce_loss(g, labels)

                if self.sample == 'wg':
                    # loss has reduction='none'
                    if self.loss == 'BCE':
                        loss = torch.mean(loss, dim=1)
                    weights = weights.float().cuda(device=self.device)
                    loss = torch.mean(loss * weights)

                    if len(labels) == loader.batch_sampler.batch_size:
                        mean_loss += loss.cpu().item()
                        total += 1

                loss.backward()
                # if self.n_classes > 1:
                #     print('GRADS: ')
                #     for i, p in enumerate(self.fc.parameters()):
                #         print(p.grad.data)

                optimizer.step()
                if self.final_prune:
                    for name, mask in self.get_final_mask_dict().items():
                        self.state_dict()[name].data[mask] = 0
                elif self.should_prune:
                    for name, mask in self.get_mask_dict().items():
                        self.state_dict()[name].data[mask] = 0

                tqdm.write('Epoch [%d/%d], Minibatch [%d/%d] Loss: %.4f' 
                           % (epoch, self.num_epoch, 
                              i % num_batches_per_epoch+1, 
                              num_batches_per_epoch, loss.data))
                pbar.update(1)

        mean_loss /= total
        return mean_loss


    def update_representation_e2e(self, dataset, prev_model, 
                                  num_workers, bft=False):
        '''
        Update feature representation for E2EIL
        Args:
            dataset: torch.utils.data.Dataset object, the dataset to train on
            prev_model: model before backprop updates for obtaining distillation 
                        labels
            num_workers: number of data loader threads to use
            bft: boolean indicating whether its the balanced finetuning stage
        '''
        torch.backends.cudnn.benchmark = True
        self.compute_means = True

        if not bft:
            num_epoch = self.num_epoch
            lr = self.init_lr
        else:
            num_epoch = self.num_epoch_ft
            lr = self.init_lr_ft


        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)

        sampler = CustomRandomSampler(dataset, num_epoch, num_workers)
        batch_sampler = CustomBatchSampler(sampler, self.batch_size, 
                                           drop_last=False, 
                                           epoch_size=len(dataset))
        num_batches_per_epoch = batch_sampler.num_batches_per_epoch
        # Run network training
        loader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler, 
                                             num_workers=0 if self.debug else num_workers,
                                             pin_memory=True)

        print("Length of current data + exemplars : ", len(dataset))
        
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=self.momentum, 
                              weight_decay=self.weight_decay)
        if self.loss == "BCE":
            q = Variable(torch.zeros(\
                self.batch_size, self.n_classes))\
                .cuda(device=self.device)
        
        with tqdm(total=num_batches_per_epoch*num_epoch) as pbar:
            for i, (indices, images, labels) in enumerate(loader):
                epoch = i//num_batches_per_epoch
                if ((epoch+1) % self.llr_freq == 0 
                        and i % num_batches_per_epoch == 0):
                    tqdm.write('Lowering Learning rate at epoch %d' %
                               (epoch+1))
                    lr = lr * 1.0/self.lr_dec_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                images = Variable(images).cuda(device=self.device)
                labels = Variable(labels).cuda(device=self.device)

                optimizer.zero_grad()

                logits = self.forward(images)

                # Get classification loss
                # if n_classes = 1, classification loss is BCE
                if self.n_classes == 1:
                    logits = torch.sigmoid(logits)
                    # q of shape (batch_size, 1) same as logits
                    q_0 = Variable(torch.zeros(len(images), 1)
                                ).cuda(device=self.device)
                    if torch.any(labels!=-1):
                        q_0[labels != -1, 0] = 1
                    cls_loss = self.bce_loss(logits, q_0) 
                    del q_0
                else:
                    if self.loss == "BCE":
                        g = torch.sigmoid(logits)
                        q[:, :] = 0
                        pos_labels = labels
                        pos_indices = torch.arange(0, g.data.shape[0], 
                            out=torch.LongTensor()).cuda(device=self.device)[pos_labels != -1]
                        pos_labels = pos_labels[pos_labels != -1]

                        if len(pos_indices) > 0:
                            q[pos_indices, pos_labels] = 1

                        cls_loss = self.bce_loss(g, q[:len(labels)])

                    elif self.loss == "CE":
                        cls_loss = self.ce_loss(logits, labels)


                # Get distillation loss
                if self.dist and (bft or self.n_classes > 1):
                    logits_dist = logits[:, :self.n_known]
                    dist_target = prev_model.forward(images)

                    if logits_dist.shape[1] == 1:
                        logits_dist = torch.sigmoid(logits_dist)
                        dist_target = torch.sigmoid(dist_target)
                        dist_loss = self.bce_loss(logits_dist, 
                                                  Variable(dist_target.data,
                                                           requires_grad=False
                                                           ).cuda(device=self.device))
                    else:
                        dist_loss = MultiClassCrossEntropyLoss(
                            logits_dist, dist_target, self.T, device=self.device)

                    loss = dist_loss*self.T*self.T + cls_loss 
                else:
                    loss = cls_loss

                if self.sample == 'gradient_iid':
                    # loss has reduction='none'
                    if self.loss == 'BCE' or self.n_classes == 1:
                        loss = torch.mean(loss, dim=1)
                    if self.n_classes == 1:
                        freqs = torch.bincount(labels+1) #So -1 gets to 0
                    else:
                        freqs = torch.bincount(labels)
                    # batch_size/weight instead of 1/weight so gradients don't 
                    # get too small
                    # print(freqs)
                    weights = self.batch_size/(freqs[labels].float())
                    loss = torch.mean(loss * weights)

                loss.backward()
                # loss.backward(retain_graph=True)
                # get the grads for each layer (dL/dW)
                # grads = torch.autograd.grad(loss, self.parameters())
                # # add noise to the grads
                # # dL/dW += N(0, std_e)
                # for p in grads:
                #     p.data.add_(torch.cuda.FloatTensor(
                #         p.shape, device=self.device).normal_(std=self.std[epoch]))

                optimizer.step()
                if self.should_prune:
                    for name, mask in self.get_mask_dict().items():
                        self.state_dict()[name].data[mask] = 0
                tqdm.write('Epoch [%d/%d], Minibatch [%d/%d] Loss: %.4f'
                           % ((epoch+1), num_epoch, i % num_batches_per_epoch+1, 
                              num_batches_per_epoch, loss.data))

                pbar.update(1)
