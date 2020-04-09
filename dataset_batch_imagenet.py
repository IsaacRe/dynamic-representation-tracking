import torch
import os
# from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageFolder
from torchvision import transforms
# from torchvision.datasets import ImageNet
from PIL import Image
import numpy as np
import cv2

import time

class bImageNet(ImageFolder):
    def __init__(self,
                 root='/home/cgu45/data/',
                 train=True,
                 transform=None,
                 target_transform=None):
        if train:
            root_dir = os.path.join(root, "train/")
            super(bImageNet, self).__init__(root_dir,
                                            transform=transform,
                                            target_transform = target_transform)
            if os.path.exists("data_generator/imagenet_train_data.npy"):
                self.train_data = np.load("data_generator/imagenet_train_data.npy")
                print("bImageNet data shape: ", self.train_data.shape)
            else:
                train_data = []
                train_labels = []
                temp_arr = []
                for i in range(len(self.samples)):
                    train_labels.append(super().__getitem__(i)[1])
                    img = np.array(super().__getitem__(i)[0])
                    temp_arr.append(img)
                    if i % 100 == 0:
                        train_data += temp_arr
                        temp_arr = []
                train_data += temp_arr
                train_data = np.array(train_data)
                np.save("data_generator/imagenet_train_data.npy", train_data)
                self.train_data = train_data
            self.train_labels = [d[1] for d in self.samples]
        else:
            test_dir = os.path.join(root, "test/")
            super(bImageNet, self).__init__(test_dir,
                                            transform=transform,
                                            target_transform = target_transform)
            if os.path.exists("data_generator/imagenet_test_data.npy"):
                self.test_data = np.load("data_generator/imagenet_test_data.npy")
            else:
                test_data = []
                test_labels = []
                temp_arr = []
                for i in range(len(self.samples)):
                    test_labels.append(super().__getitem__(i)[1])
                    img = np.array(super().__getitem__(i)[0])
                    temp_arr.append(img)
                    if i % 100 == 0:
                        test_data += temp_arr
                        temp_arr = []
                test_data += temp_arr
                test_data = np.array(test_data)
                np.save("data_generator/imagenet_test_data.npy", test_data)
                self.test_data = test_data
            self.test_labels = [d[1] for d in self.samples]

class ImageNet(bImageNet):
    def __init__(self, classes,
                 root='/home/cgu45/data/',
                 train=True,
                 mean_image=None,
                 transform=None,
                 target_transform=None,
                 download=False):

        self.train = train
        super(ImageNet, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform)
        if train:
            # print("image net train data 0: ", self.train_data)
            # print("image net train labels: ", len(self.train_labels))
            print("ImageNet initial data shape: ", self.train_data.shape)
            print("classes: ", classes)
            print("root: ", root)
            train_data = []
            train_labels = []
            np_labels = np.array(self.train_labels)
            # print("np labels shape: {}".format(np_labels.shape))
            for i in classes:
                idx = np.where(np_labels == i)
                train_data += [self.train_data[idx]]
                train_labels += list(np_labels[idx])
            self.train_data = np.concatenate(train_data)
            print("ImageNet concat data shape: ", self.train_data.shape)
            self.img_size = 224

            self.train_labels = train_labels

            # resize images
            # print("image net train data 1: ", self.train_data)
            train_data = self.train_data
            self.train_data = np.zeros((len(self.train_labels), self.img_size, self.img_size, 3))
            for i, img in enumerate(train_data):
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.train_data[i, :] = img
            print("ImageNet reshaped data shape: ", self.train_data.shape)

        else:
            test_data = []
            test_labels = []
            np_labels = np.array(self.test_labels)
            for i in classes:
                idx = np.where(np_labels == i)
                test_data += [self.test_data[idx]]
                test_labels += list(np_labels[idx])
            self.test_data = np.concatenate(test_data)
            self.img_size = 224

            self.test_labels = test_labels

            # resize images
            test_data = self.test_data
            self.test_data = np.zeros((len(self.test_labels), self.img_size, self.img_size, 3))
            for i, img in enumerate(test_data):
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.test_data[i, :] = img

        self.mean_image = mean_image



    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index]
            img = Image.fromarray(np.uint8(img))
            img = np.array(self.transform(img))
            img = img.transpose(2,0,1)

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
            img = self.test_data[index].transpose(2,0,1)
            img = torch.FloatTensor((img - self.mean_image)/255.)

            return index, img, self.test_labels[index]
