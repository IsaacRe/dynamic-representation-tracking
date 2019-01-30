# NOTE : import orders are important
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import time

import torchvision.models as models
import torchvision.transforms as transforms

from utils.loader_utils import CustomRandomSampler, CustomBatchSampler


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class IncrNet(nn.Module):
    def __init__(self, args, device):
        # Hyper Parameters
        self.init_lr = args.init_lr
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.lr_dec_factor = args.lrd
        self.weight_decay = args.wd
        # Hardcoded
        self.lower_rate_epoch = [
            int(0.7 * self.num_epoch), int(0.9 * self.num_epoch)]
        self.momentum = 0.9

        self.pretrained = args.pretrained
        self.dist = args.dist
        self.algo = args.algo
        self.epsilon = 1e-16

        # Network architecture
        super(IncrNet, self).__init__()
        self.model = models.resnet34(pretrained=self.pretrained)
        if not self.pretrained:
            self.model.apply(kaiming_normal_init)

        self.device = device
        feat_size = self.model.fc.in_features
        self.model.fc = nn.Linear(feat_size, 1, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-1])

        # n_classes incremented before processing new data in an iteration
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
        # for each exemplar store which data unit it came from
        self.eset_du_maps = []
        # store bounding boxes for all exemplars
        self.exemplar_bbs = []
        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

        # Loss functions
        self.cls_loss = nn.BCELoss()
        self.dist_loss = nn.BCELoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def increment_classes(self, new_classes):
        '''
        Make network changes when new classes are seen
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

    def compute_exemplar_means(self):
        '''
        Compute exemplar means from exemplar_sets
        '''
        exemplar_means = []
        for P_y in self.exemplar_sets:

            # Concatenate with horizontally flipped images
            all_imgs = np.concatenate((P_y, P_y[:, :, :, ::-1]), axis=0)
            all_features = []

            # Batch up all_imgs to fit on the GPU
            # all_features on GPU
            for i in range(0, len(all_imgs), self.batch_size):
                img_tensors = Variable(
                    torch.FloatTensor(
                        all_imgs[i:min(i+self.batch_size, len(all_imgs))])).cuda(device=self.device)
                features = self.feature_extractor(img_tensors)
                del img_tensors

                # Normalize along dimension 1
                features_norm = features.data.norm(p=2, dim=1) + self.epsilon
                features_norm = features_norm.unsqueeze(1)
                features.data = features.data.div(
                    features_norm.expand_as(features))  # Normalize

                all_features.append(features)

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
        Classify images by nearest-mean-of-exemplars
        Args:
            x: input images
        Returns:
            preds: Tensor of size (x.shape[0],)
        '''
        batch_size = x.size(0)

        if self.algo == 'icarl':
            if self.compute_means:
                self.compute_exemplar_means()

            exemplar_means = self.exemplar_means
            means = torch.stack(exemplar_means).cuda(
                device=self.device)  # (n_classes, feature_size)
            # (batch_size, n_classes, feature_size)
            means = torch.stack([means] * batch_size)
            # (batch_size, feature_size, n_classes)
            means = means.transpose(1, 2)

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
                _, preds = dists.min(1)

        elif self.algo == 'hybrid1' or self.algo == 'lwf':
            sigmoids = torch.sigmoid(self.forward(x))
            _, preds = sigmoids.max(dim=1)

        return preds

    def construct_exemplar_set(self, images, du_maps, 
                               image_bbs, m, cl, curr_iter):
        '''
        Construct an exemplar set given images
        Args:
            images: np.array containing images of a class
            du_maps: np.array containing data units and frame indices of images
            image_bbs: np.array containing image bounding boxes
            m: number of images in the exemplar set
            cl: class index in self.classes of the current exemplar set
            curr_iter: learning exposure index
        '''
        num_new_imgs = np.sum(du_maps[:, 0] == curr_iter)
        num_old_imgs = len(images) - num_new_imgs
        all_features = []

        # Batch up images so that not all needs to be fit on the GPU memory at once
        for i in range(0, len(images), self.batch_size):
            with torch.no_grad():
                img_tensors = Variable(torch.FloatTensor(
                    images[i:min(i+self.batch_size, len(images))])).cuda(device=self.device)

            features = self.feature_extractor(img_tensors)
            del img_tensors
            # Normalize along dimension 1
            features_norm = features.data.norm(p=2, dim=1) + self.epsilon
            features_norm = features_norm.unsqueeze(1)
            features.data = features.data.div(
                features_norm.expand_as(features))  # Normalize
            features.data = features.data.squeeze(3)
            features.data = features.data.squeeze(2)
            features = features.data.cpu().numpy()

            all_features.append(features)

        features = np.concatenate(all_features, axis=0)

        weights = np.zeros((len(features), 1))
        weights[du_maps[:, 0] == curr_iter] = float(
            num_old_imgs + 1)/(num_old_imgs + num_new_imgs + 1)
        weights[du_maps[:, 0] != curr_iter] = float(
            num_new_imgs + 1)/(num_old_imgs + num_new_imgs + 1)

        class_mean = np.sum(weights * features, axis=0)/np.sum(weights)
        class_mean = class_mean / \
            (np.linalg.norm(class_mean) + self.epsilon)  # Normalize

        indices_remaining = np.arange(0, len(images))
        indices_selected = []

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

        if cl < self.n_known:
            self.exemplar_sets[cl] = np.array(images[indices_selected])
            self.eset_du_maps[cl] = np.array(du_maps[indices_selected])
            self.exemplar_bbs[cl] = np.array(image_bbs[indices_selected])
            self.n_occurrences[cl] += 1
        else:
            self.exemplar_sets.append(np.array(images[indices_selected]))
            self.eset_du_maps.append(np.array(du_maps[indices_selected]))
            self.exemplar_bbs.append(np.array(image_bbs[indices_selected]))
            self.n_occurrences.append(1)

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
            self.eset_du_maps[y] = self.eset_du_maps[y][:m]
            self.exemplar_bbs[y] = self.exemplar_bbs[y][:m]

    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels,
                           self.exemplar_bbs[y], self.eset_du_maps[y])

    def fetch_hyper_params(self):
        return {'num_epoch': self.num_epoch,
                'batch_size': self.batch_size,
                'lower_rate_epoch': self.lower_rate_epoch,
                'lr_dec_factor': self.lr_dec_factor,
                'init_lr': self.init_lr,
                'pretrained': self.pretrained,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay}

    def update_representation(self, dataset, prev_model, 
                              curr_class_idxs, num_workers):
        '''
        Update feature representation
        Args:
            dataset: torch.utils.data.Dataset object, the dataset to train on
            prev_model: model before backprop updates for obtaining distillation labels
            curr_class_idxs: indices in [0, self.n_known] of new classes.
                             also contains an index if its a repeated exposure 
                             to a class 
            num_workers: number of data loader threads to use
        '''
        torch.backends.cudnn.benchmark = True
        self.compute_means = True
        self.lr = self.init_lr

        # Form combined training set
        if self.algo == 'icarl' or self.algo == 'hybrid1':
            self.combine_dataset_with_exemplars(dataset)

        # NOTE : see if there are synchronization issues
        sampler = CustomRandomSampler(dataset, self.num_epoch, num_workers)
        batch_sampler = CustomBatchSampler(
            sampler, self.batch_size, drop_last=False, epoch_size=len(dataset))
        num_batches_per_epoch = batch_sampler.num_batches_per_epoch
        # Run network training
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers, 
            pin_memory=True)

        print("Length of current data + exemplars : ", len(dataset))
        optimizer = optim.SGD(self.parameters(), lr=self.lr,
                              momentum=self.momentum, 
                              weight_decay=self.weight_decay)

        q = Variable(torch.zeros(self.batch_size, self.n_classes)
                     ).cuda(device=self.device)
        with tqdm(self.num_epoch) as pbar:
            for i, (indices, images, labels) in enumerate(loader):
                epoch = i//num_batches_per_epoch

                if ((epoch+1) in self.lower_rate_epoch 
                        and i % num_batches_per_epoch == 0):
                    self.lr = self.lr * 1.0/self.lr_dec_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr

                images = Variable(images).cuda(device=self.device)

                optimizer.zero_grad()
                g = self.forward(images)
                g = torch.sigmoid(g)
                q[:, :] = 0

                if self.dist:
                    labels = labels.cuda(device=self.device)
                    if self.n_known > 0:
                        # Store network outputs with pre-update parameters
                        q_prev = torch.sigmoid(prev_model.forward(images))
                        q.data[:len(labels), :self.n_known] = q_prev.data[:, :self.n_known]

                    # For new classes use label 1
                    for new_class_idx in curr_class_idxs:
                        q.data[:len(labels), new_class_idx] = 0
                        q.data[:len(labels), new_class_idx].masked_fill_(
                            labels == new_class_idx, 1)
                else:
                    labels = labels.cuda(device=self.device)
                    pos_labels = labels
                    pos_indices = torch.arange(0, g.data.shape[0], 
                        out=torch.LongTensor()).cuda(device=self.device)[pos_labels != -1]
                    pos_labels = pos_labels[pos_labels != -1]

                    if len(pos_indices) > 0:
                        q[pos_indices, pos_labels] = 1

                loss = self.cls_loss(g, q[:len(labels)])
                loss.backward()
                optimizer.step()

                if i%num_batches_per_epoch == 0:
                    tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                               % (epoch, self.num_epoch, 
                                  i % num_batches_per_epoch+1, 
                                  num_batches_per_epoch, loss.data))
                    pbar.update(1)
