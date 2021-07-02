from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from scipy.stats import truncnorm
#from torchnet.meter import AUCMeter

			
def unpickle(file):
	import _pickle as cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo, encoding='latin1')
	return dict

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=0)

class CutoutDefault(object):
	"""
	Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
	"""
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		if self.length <= 0:
			return img
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img

class cifar_dataset(Dataset): 
	def __init__(self, dataset, r, root_dir, transform, mode, log=''): 
		
		self.dataset = dataset
		self.transform = transform
		self.set_transform()
		self.mode = mode  
		self.r = r	
		self.transform_cutout = CutoutDefault(2)

		if self.mode=='test':
			if dataset=='cifar10':                
				test_dic = unpickle('%s/test_batch'%root_dir)
				self.test_data = test_dic['data']
				self.test_data = self.test_data.reshape((10000, 3, 32, 32))
				self.test_data = self.test_data.transpose((0, 2, 3, 1))  
				self.test_label = test_dic['labels']
			elif dataset=='cifar100':
				test_dic = unpickle('%s/test'%root_dir)
				self.test_data = test_dic['data']
				self.test_data = self.test_data.reshape((10000, 3, 32, 32))
				self.test_data = self.test_data.transpose((0, 2, 3, 1))  
				self.test_label = test_dic['fine_labels']                            
		else:    
			train_data=[]
			train_label=[]
			if dataset=='cifar10': 
				for n in range(1,6):
					dpath = '%s/data_batch_%d'%(root_dir,n)
					data_dic = unpickle(dpath)
					train_data.append(data_dic['data'])
					train_label = train_label+data_dic['labels']
				train_data = np.concatenate(train_data)
			elif dataset=='cifar100':    
				train_dic = unpickle('%s/train'%root_dir)
				train_data = train_dic['data']
				train_label = train_dic['fine_labels']
			train_data = train_data.reshape((50000, 3, 32, 32))
			train_data = train_data.transpose((0, 2, 3, 1))

			if dataset == 'cifar100':
				num_classes = 100
			elif dataset == 'cifar10':
				num_classes = 10
			
			num_samples = 50000
			class_num = np.zeros(num_classes, dtype=np.int)
			index = list(range(50000))
			random.shuffle(index)
			labeled_num = int(self.r*50000)
			labeled_idx = []
			unlabeled_idx = []

			for i in index:
				label = train_label[i]
				if class_num[label]<int(num_samples/num_classes) and len(labeled_idx) < labeled_num:
					labeled_idx.append(i)
					class_num[label] += 1
				else:
					unlabeled_idx.append(i)
					
					  
			if self.mode == "labeled":
				pred_idx = labeled_idx
					
			elif self.mode == "unlabeled":
				pred_idx = unlabeled_idx
				
			self.train_data = train_data[pred_idx]
			self.train_label = [train_label[i] for i in pred_idx]
			print("labeled data %d, unlabeled data %d"%(len(labeled_idx),len(unlabeled_idx)))
				
	def __getitem__(self, index):
		if self.mode=='labeled':
			img, target = self.train_data[index], self.train_label[index]
			img = Image.fromarray(img)
			img1 = self.transform_train(img)
			img1 = self.transform_cutout(img1)
			img2 = self.transform_train(img)
			img2 = self.transform_cutout(img2)
			return img1, img2, target
		elif self.mode=='unlabeled':
			img, target = self.train_data[index], self.train_label[index]
			img = Image.fromarray(img)
			img1 = self.transform_test(img)
			img2 = self.transform_train(img)
			img2 = self.transform_cutout(img2)
			return img1, img2, target
		elif self.mode=='test':
			img, target = self.test_data[index], self.test_label[index]
			img = Image.fromarray(img)
			img = self.transform_test(img)
			return img, target

	def __len__(self):
		if self.mode!='test':
			return len(self.train_data)
		else:
			return len(self.test_data)         

	def set_transform(self):
		if self.dataset=='cifar10':
			mean = (0.4914, 0.4822, 0.4465)
			std = (0.2023, 0.1994, 0.2010)
		elif self.dataset=='cifar100':
			mean = (0.507, 0.487, 0.441)
			std = (0.267, 0.256, 0.276)

		self.transform_train = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean,std),
			])
		self.transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean,std),
			])

class cifar_dataloader():  
	def __init__(self, dataset, r, batch_size, num_workers, root_dir, log):
		self.dataset = dataset
		self.r = r
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.root_dir = root_dir
		self.log = log
		
	def run(self, mode):        
		if mode=='train':
			labeled_dataset = cifar_dataset(dataset=self.dataset, r=self.r, root_dir=self.root_dir, transform=None, mode="labeled", log=self.log)
			labeled_trainloader = DataLoader(
				dataset=labeled_dataset, 
				batch_size=self.batch_size,
				shuffle=True,
				num_workers=self.num_workers,
				drop_last=True)   
			
			unlabeled_dataset = cifar_dataset(dataset=self.dataset, r=self.r, root_dir=self.root_dir, transform=None, mode="unlabeled", log=self.log)
			unlabeled_trainloader = DataLoader(
				dataset=unlabeled_dataset, 
				batch_size=self.batch_size,
				shuffle=True,
				num_workers=self.num_workers,
				drop_last=True)

			return labeled_trainloader, unlabeled_trainloader
		
		elif mode=='test':
			test_dataset = cifar_dataset(dataset=self.dataset, r=self.r, root_dir=self.root_dir, transform=None, mode='test', log=self.log)
			test_loader = DataLoader(
				dataset=test_dataset, 
				batch_size=self.batch_size,
				shuffle=False,
				num_workers=self.num_workers)          
			return test_loader

