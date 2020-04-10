import torch
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2


class ImageNet(Dataset):
    def __init__(self, classes,
                 root='/home/cgu45/data/',
                 train=True,
                 mean_image=None,
                 transform=None,
                 target_transform=None):
        self.classes = classes;
        self.train = train
        # TODO: mean_image
        self.transform = transform
        self.target_transform = target_transform

        if train:
            train_dir = os.path.join(root, "train/")
            dataset = ImageFolder(train_dir, transform=transform, target_transform=target_transform)
            self.train_samples = dataset.samples
            self.num_train = len(self.train_samples)

            self.img_size = 224
            self.train_data = \
                np.zeros((self.num_train, 3, self.img_size, self.img_size), dtype=np.float32)
            self.train_labels = np.zeros(self.num_train, dtype=np.int)
            for i, (train_image, train_label) in enumerate(self.train_samples):
                img = np.asarray(Image.open(train_image).convert("RGB"))
                img = cv2.resize(img, (self.img_size, self.img_size)).transpose(2, 0, 1)
                self.train_data[i] = img
                self.train_labels[i] = train_label

            train_data = []
            train_labels = []
            np_labels = self.train_labels
            for i in classes:
                idx = np.where(np_labels == i)
                train_data += [self.train_data[idx]]
                train_labels += list(np_labels[idx])
            self.train_data = np.concatenate(train_data)

            self.train_labels = train_labels

            # resize images
            # train_data = self.train_data
            # self.train_data = np.zeros((len(self.train_labels), self.img_size, self.img_size, 3))
            # for i, img in enumerate(train_data):
                # img = cv2.resize(img, (self.img_size, self.img_size))
                # self.train_data[i, :] = img

        else:
            test_dir = os.path.join(root, "test/")
            dataset = ImageFolder(test_dir, transform=transform, target_transform=target_transform)
            self.test_samples = dataset.samples
            self.num_test = len(self.test_samples)

            self.img_size = 224
            self.test_data = \
                np.zeros((self.num_test, 3, self.img_size, self.img_size), dtype=np.float32)
            self.test_labels = np.zeros(self.num_test, dtype=np.int)
            for i, (test_image, test_label) in enumerate(self.test_samples):
                img = np.asarray(Image.open(test_image).convert("RGB"))
                img = cv2.resize(img, (self.img_size, self.img_size)).transpose(2, 0, 1)
                self.test_data[i] = img
                self.test_labels[i] = test_label

            test_data = []
            test_labels = []
            np_labels = self.test_labels
            for i in classes:
                idx = np.where(np_labels == i)
                test_data += [self.test_data[idx]]
                test_labels += list(np_labels[idx])
            self.test_data = np.concatenate(test_data)

            self.test_labels = test_labels

            # resize images
            # test_data = self.test_data
            # self.test_data = np.zeros((len(self.test_labels), self.img_size, self.img_size, 3))
            # for i, img in enumerate(test_data):
                # img = cv2.resize(img, (self.img_size, self.img_size))
                # self.test_data[i, :] = img


        self.mean_image = mean_image



    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index]
            img = Image.fromarray(np.uint8(img))
            img = np.array(self.transform(img))
            # img = img.transpose(2,0,1)

            if self.mean_image is not None:
                img = (img - self.mean_image)/255.
            else:
                img /= 255.
            # Augment : Random crops and horizontal flips
            random_cropped = np.zeros(img.shape, dtype=np.float32)
            padded = np.pad(img, ((0, 0), (4, 4), (4, 4)),
                            mode="constant")
            crops = np.random.random_integers(0, high=8, size=(1, 2))
            if (np.random.randint(2) > 0):
                random_cropped[:, :, :] = padded[:,
                    crops[0, 0]:(crops[0, 0] + self.img_size),
                    crops[0, 1]:(crops[0, 1] + self.img_size)]
            else:
                random_cropped[:, :, :] = padded[:,
                    crops[0, 0]:(crops[0, 0] + self.img_size),
                    crops[0, 1]:(crops[0, 1] + self.img_size)][:, :, ::-1]

            img = torch.FloatTensor(random_cropped)

            target = self.train_labels[index]

            return index, img, target

        else:
            # img = self.test_data[index].transpose(2,0,1)
            img = torch.FloatTensor((img - self.mean_image)/255.)

            return index, img, self.test_labels[index]
