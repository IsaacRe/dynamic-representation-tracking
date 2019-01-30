import numpy as np
import torch
import cv2
import random
import sys
from color_jitter.jitter import jitter
from utils.get_samples import get_samples
from torch.utils.data.sampler import Sampler
from multiprocessing.dummy import Pool as ThreadPool
import itertools

class iToys(torch.utils.data.Dataset):
	def __init__(self, args, img_size, mean_image, data_generators, max_data_size, classes = None, class_map = None, job = None, size_test = None, du_idx = -1):
		self.job = job
		self.rgb = args.rgb
		self.jitter = args.jitter
		self.img_size = img_size

		self.datax = np.empty((max_data_size, 3, img_size, img_size), dtype=np.float32)
		self.datay = np.empty(max_data_size, dtype=int)
		self.bb = np.empty((max_data_size, 4), dtype=int)
		self.curr_len = 0

		self.h_ch, self.s_ch, self.l_ch = args.h_ch, args.s_ch, args.l_ch


		# Grayscaling mean_image
		if args.rgb == False:
			mean_image = cv2.cvtColor(np.uint8(mean_image), cv2.COLOR_RGB2GRAY)
		
		#rendered_img_size x rendered_img_size
		self.mean_image = mean_image 

		# Note : du_idx is the index of the "batch of classes" seen, if multiple classes are passed in a batch
		# Data z stores which dataunit an image comes from, populated only when job=='train' 
		self.dataz = np.empty((max_data_size, 2), dtype=int)

		# Can also initialize an empty set and expand later
		if len(data_generators) > 0:
			assert len(data_generators) == len(classes), "Number of datagenerators and classes passed is different"
			self.expand(args, img_size, data_generators, classes, class_map, job, size_test, du_idx)
	
	# To allow using the same memory locations for datax
	def pseudo_init(self, args, img_size, data_generators, classes = None, class_map = None, job = None, size_test = None, du_idx = -1):
		self.curr_len = 0
		if len(data_generators) > 0:
			assert len(data_generators) == len(classes), "Number of datagenerators and classes passed is different"
			self.expand(args, img_size, data_generators, classes, class_map, job, size_test, du_idx)


	def __getitem__(self, index):
		image = self.datax[index]
		bb = self.bb[index]

		if self.job == 'train':
			#### Augment : Color jittering
			if self.jitter and self.rgb:
				# Perform color jittering only under these conditions
				x_min, x_max, y_min, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
				# Reconstructing image by adding mean image to do jittering, image is in range (0,1)
				cropped_mean = cv2.resize(self.mean_image[y_min:y_max,x_min:x_max],(self.img_size, self.img_size))
				# Cropped mean needs to be 3ximg_sizeximg_size
				cropped_mean = cropped_mean.transpose(2,0,1)
				image = np.array(image*255. + cropped_mean, dtype = np.float32, order = 'C')
				jitter(image, self.h_ch, self.s_ch, self.l_ch)
				image = (image - cropped_mean)/255.

			##### Augment : Random crops and horizontal flips
			random_cropped = np.zeros(image.shape, dtype=np.float32)
			padded = np.pad(image,((0,0),(4,4),(4,4)),mode='constant')
			crops = np.random.random_integers(0,high=8,size=(1,2))
			# Cropping and possible flipping
			if (np.random.randint(2) > 0):
				random_cropped[:,:,:] = padded[:,crops[0,0]:(crops[0,0]+self.img_size),crops[0,1]:(crops[0,1]+self.img_size)]
			else:
				random_cropped[:,:,:] = padded[:,crops[0,0]:(crops[0,0]+self.img_size),crops[0,1]:(crops[0,1]+self.img_size)][:,:,::-1]
			image = torch.FloatTensor(random_cropped)
		elif self.job == 'test':
			image = torch.FloatTensor(image)

		target = self.datay[index]
		
		return index, image, target

	def __len__(self):
		return self.curr_len

	def get_image_class(self, label):
		curr_data_y = self.datay[:self.curr_len]
		return self.datax[:self.curr_len][curr_data_y == label], self.dataz[:self.curr_len][curr_data_y == label], self.bb[:self.curr_len][curr_data_y == label]

	def append(self, images, labels, bb, dataunits):
		"""Append dataset with images and labels
		Args:
			images: Tensor of shape (num_images, channels, H, W)
			labels: list of labels
		"""
		if self.curr_len + len(images) > len(self.datax):
			raise Exception("Dataset max length exceeded")

		self.datax[self.curr_len:self.curr_len+len(images)] = images
		self.datay[self.curr_len:self.curr_len+len(labels)] = labels
		self.bb[self.curr_len:self.curr_len+len(bb)] = bb
		self.dataz[self.curr_len:self.curr_len+len(dataunits)] = dataunits

		self.curr_len += len(images)

	def expand(self, args, img_size, data_generators, classes, class_map, job, size_test, du_idx = -1):
		# If job is train, different data generators are passed for different classes
		if job == 'train':
			assert len(classes) == len(data_generators) 
			assert du_idx >= 0
		
		for i, cl in enumerate(classes):
			if job == 'train':
				data_generator = data_generators[i]
				images, bboxes = data_generator.getDataUnit()
			elif job == "test":
				data_generator = data_generators[i]
				images, bboxes = data_generator.getRandomPoints()
			else:
				raise Exception("Operation not supported, should be either 'test' or 'train'")

			images = (images - self.mean_image)/255. #3xrendered_img_size x rendered_img_size
			[datax, datay, bb] = get_samples(args, images, bboxes, cl, img_size, class_map, job)
			
			if self.curr_len + len(datax) > len(self.datax):
				raise Exception("Dataset max length exceeded")

			self.datax[self.curr_len:self.curr_len+len(datax)] = datax
			self.datay[self.curr_len:self.curr_len+len(datay)] = datay
			self.bb[self.curr_len:self.curr_len+len(bb)] = bb
			self.dataz[self.curr_len:self.curr_len+len(datax), 0] = du_idx 
			# numbering the frames that come in. Both the +ve and -ve frames get the same number
			# TODO: change if no negative images are passed
			self.dataz[self.curr_len:self.curr_len+len(datax), 1] = np.arange(len(datax))//2 
			self.curr_len += len(datax)


class CustomRandomSampler(Sampler):
	"""
	Samples elements randomly, without replacement. 
	This sampling only shuffles within epoch intervals of the dataset 
	Arguments:
		data_source (Dataset): dataset to sample from
		num_epochs (int) : Number of epochs in the train dataset
		num_workers (int) : Number of workers to use for generating iterator
	"""

	def __init__(self, data_source, num_epochs, num_workers):
		self.data_source = data_source
		self.num_epochs = num_epochs
		self.num_workers = num_workers
		self.datalen = len(data_source)

	def __iter__(self):
		iter_array = []
		pool = ThreadPool(self.num_workers)
		def get_randperm(i):
			return torch.randperm(self.datalen).tolist()
		iter_array = list(itertools.chain.from_iterable(pool.map(get_randperm, range(self.num_epochs))))
		pool.close()
		pool.join()
		return iter(iter_array)

	def __len__(self):
		return len(self.data_source)

from torch._six import int_classes as _int_classes
class CustomBatchSampler(object):
	"""
	Wraps another custom sampler with epoch intervals to yield a mini-batch of indices.

	Args:
		sampler (Sampler): Base sampler.
		batch_size (int): Size of mini-batch.
		drop_last (bool): If ``True``, the sampler will drop the last batch if
			its size would be less than ``batch_size``
		epoch_size : Number of items in an epoch
	"""

	def __init__(self, sampler, batch_size, drop_last, epoch_size):
		if not isinstance(sampler, Sampler):
			raise ValueError("sampler should be an instance of "
							 "torch.utils.data.Sampler, but got sampler={}"
							 .format(sampler))
		if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
				batch_size <= 0:
			raise ValueError("batch_size should be a positive integeral value, "
							 "but got batch_size={}".format(batch_size))
		if not isinstance(drop_last, bool):
			raise ValueError("drop_last should be a boolean value, but got "
							 "drop_last={}".format(drop_last))
		self.sampler = sampler
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.epoch_size = epoch_size
		self.num_epochs = len(self.sampler)/self.epoch_size
		
		if self.drop_last:
			self.num_batches_per_epoch = self.epoch_size // self.batch_size
		else:
			self.num_batches_per_epoch = (self.epoch_size + self.batch_size - 1) // self.batch_size

	def __iter__(self):
		batch = []
		epoch_ctr = 0 
		for idx in self.sampler:
			epoch_ctr += 1
			batch.append(int(idx))
			if len(batch) == self.batch_size or epoch_ctr == self.epoch_size:
				yield batch
				batch = []
				if epoch_ctr == self.epoch_size:
					epoch_ctr = 0
			
		if len(batch) > 0 and not self.drop_last:
			yield batch

	def __len__(self):
		return self.num_epochs * self.num_batches_per_epoch