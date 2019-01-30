import numpy as np
import torch
import cv2
import random
import sys
from color_jitter.jitter import jitter
from utils.get_samples import get_samples


class iToys(torch.utils.data.Dataset):
    def __init__(self, args, mean_image, data_generators, max_data_size,
                 classes=None, class_map=None, job=None, du_idx=-1):
        self.job = job
        self.jitter = args.jitter
        self.img_size = args.img_size

        self.datax = np.empty(
            (max_data_size,
             3,
             self.img_size,
             self.img_size),
            dtype=np.float32)
        self.datay = np.empty(max_data_size, dtype=int)
        self.bb = np.empty((max_data_size, 4), dtype=int)
        self.curr_len = 0

        self.h_ch, self.s_ch, self.l_ch = args.h_ch, args.s_ch, args.l_ch

        # 3 x rendered_img_size x rendered_img_size
        self.mean_image = mean_image

        # Data z stores which dataunit an image comes from and its frame index,
        # populated only when job=='train'
        self.dataz = np.empty((max_data_size, 2), dtype=int)

        # Can also initialize an empty set and expand later
        if len(data_generators) > 0:
            assert len(data_generators) == len(classes), \
                'Number of datagenerators and classes passed is different'
            self.expand(args, data_generators, classes, class_map, job, du_idx)

    # To allow using the same memory locations for datax
    def pseudo_init(self, args, data_generators, classes=None,
                    class_map=None, job=None, du_idx=-1):
        self.curr_len = 0
        if len(data_generators) > 0:
            assert len(data_generators) == len(classes), \
                'Number of datagenerators and classes passed is different'
            self.expand(args, data_generators, classes, class_map, job, du_idx)

    def __getitem__(self, index):
        image = self.datax[index]
        bb = self.bb[index]

        if self.job == 'train':
            # Augment : Color jittering
            if self.jitter:
                x_min = int(bb[0])
                x_max = int(bb[1])
                y_min = int(bb[2])
                y_max = int(bb[3])
                cropped_mean = cv2.resize(
                    self.mean_image[y_min:y_max, x_min:x_max], 
                    (self.img_size, self.img_size))
                # cropped_mean final shape : 3ximg_sizeximg_size
                cropped_mean = cropped_mean.transpose(2, 0, 1)
                image = np.array(
                    image * 255. + cropped_mean,
                    dtype=np.float32,
                    order='C')
                jitter(image, self.h_ch, self.s_ch, self.l_ch)
                image = (image - cropped_mean) / 255.

            # Augment : Random crops and horizontal flips
            random_cropped = np.zeros(image.shape, dtype=np.float32)
            padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode='constant')
            crops = np.random.random_integers(0, high=8, size=(1, 2))
            if (np.random.randint(2) > 0):
                random_cropped[:, :, :] = padded[:, 
                    crops[0, 0]:(crops[0, 0] + self.img_size), 
                    crops[0, 1]:(crops[0, 1] + self.img_size)]
            else:
                random_cropped[:, :, :] = padded[:, 
                    crops[0, 0]:(crops[0, 0] + self.img_size), 
                    crops[0, 1]:(crops[0, 1] + self.img_size)][:, :, ::-1]

            image = torch.FloatTensor(random_cropped)

        elif self.job == 'test':
            image = torch.FloatTensor(image)

        target = self.datay[index]

        return index, image, target

    def __len__(self):
        return self.curr_len

    def get_image_class(self, label):
        curr_data_y = self.datay[:self.curr_len]
        return (self.datax[:self.curr_len][curr_data_y == label], 
                self.dataz[:self.curr_len][curr_data_y ==label], 
                self.bb[:self.curr_len][curr_data_y == label])

    def append(self, images, labels, bb, dataunits):
        '''
        Append dataset with images, labels, and other metadata
        '''
        if self.curr_len + len(images) > len(self.datax):
            raise Exception("Dataset max length exceeded")

        self.datax[self.curr_len:self.curr_len + len(images)] = images
        self.datay[self.curr_len:self.curr_len + len(labels)] = labels
        self.bb[self.curr_len:self.curr_len + len(bb)] = bb
        self.dataz[self.curr_len:self.curr_len + len(dataunits)] = dataunits

        self.curr_len += len(images)

    def expand(self, args, data_generators,
               classes, class_map, job, du_idx=-1):
        '''
        Call data generator to get images and append to dataset
        '''
        if job == 'train':
            assert du_idx >= 0, 'Needs to have positive learning exposure index'

        for i, cl in enumerate(classes):
            if job == 'train':
                data_generator = data_generators[i]
                images, bboxes = data_generator.getDataUnit()
            elif job == "test":
                data_generator = data_generators[i]
                images, bboxes = data_generator.getRandomPoints()
            else:
                raise Exception(
                    "Operation should be either 'test' or 'train'")

            images = (images - self.mean_image) / \
                255.  # 3 x rendered_img_size x rendered_img_size
            [datax, datay, bb] = get_samples(
                images, bboxes, cl, self.img_size, class_map, job)

            if self.curr_len + len(datax) > len(self.datax):
                raise Exception("Dataset max length exceeded")

            self.datax[self.curr_len:self.curr_len + len(datax)] = datax
            self.datay[self.curr_len:self.curr_len + len(datay)] = datay
            self.bb[self.curr_len:self.curr_len + len(bb)] = bb
            self.dataz[self.curr_len:self.curr_len + len(datax), 0] = du_idx
            
            # numbering the frames that come in. Each +ve frame is followed 
            # by a -ve background image
            # TODO: change if no negative images are passed (use args.algo)
            self.dataz[self.curr_len:self.curr_len +
                       len(datax), 1] = np.arange(len(datax)) // 2
            self.curr_len += len(datax)
