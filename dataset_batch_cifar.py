import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np
import cv2


class CIFAR20(CIFAR100):
    def __init__(self, args, classes,
                 root='./data',
                 train=True,
                 mean_image=None,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR20, self).__init__(root,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)

        # only used for training set
        train_data = []
        train_labels = []
        np_labels = np.array(self.train_labels)
        for i in classes:
            idx = np.where(np_labels == i)
            train_data += [self.train_data[idx]]
            train_labels += list(np_labels[idx])
        self.train_data = np.concatenate(train_data)
        self.img_size = args.img_size

        self.train_labels = train_labels

        # resize images
        train_data = self.train_data
        self.train_data = np.zeros((len(self.train_labels), self.img_size, self.img_size, 3))
        for i, img in enumerate(train_data):
            img = cv2.resize(img, (self.img_size, self.img_size))
            self.train_data[i, :] = img

        self.mean_image = mean_image

    def __getitem__(self, index):
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
