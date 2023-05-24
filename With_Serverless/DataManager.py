# -*- coding: utf-8 -*-
"""
This work was done by Amine Barrak and Ranim Trabelsi

"""

import pathlib 
from random import Random 
import torch 
from torchvision import datasets, transforms
import torch.optim as optim 
import torch.utils.data as data 
import torch.nn.init as init


class Partition(object):
    def __init__(self,data,index): 
        self.data = data 
        self.index = index 
    def __len__(self):
        return len(self.index) 
    def __get_data_partition__(self):
        data_partition = [self.data[i] for i  in self.index]
        return data_partition
    def __get_item__(self,index):
        data_idx = self.index[index]
        data_item = self.data[data_idx] 
        return data_item

class DataPartitioner(object): 
    def __init__(self,data,num_workers, seed=1234): 
        self.data = data 
        self.num_workers= num_workers 
        self.partitions = [] 
        rng = Random() 
        rng.seed(seed) 
        len_data = len(data) 
        indexes = [index for index in range(0,len_data)] 
        rng.shuffle(indexes) 
        partition_len= int(len_data/(self.num_workers))
        for i in range (self.num_workers):
            self.partitions.append(indexes[0:partition_len]) 
            indexes = indexes[partition_len:] 
            
    def use(self,partition): 
        
        return Partition(self.data,self.partitions[partition])

class DataManager(object): 
    def __init__(self,dataset,batch,num_workers,rank): 
        self.dataset = dataset 
        self.batch = batch 
        self.num_workers = num_workers 
        self.rank = rank 
    def fetch_dataset(self,dataset,train=True):
        homedir = str(pathlib.Path.home()) 
        if dataset == "mnist":
          if train:
            return datasets.MNIST(
              homedir+'/data',
              train=train,
              download=True,
              transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, )),
                 transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                 transforms.Resize(21) #squeeznet1_1
                 #  transforms.Resize(32) vgg11
                 # transforms.Resize(224) mobilenetv2
                 #   transforms.Resize(224)  mnasnet0_5
                 #   transforms.Resize(29)  densenet121 
                 #transforms.Resize(29) mobilenet_v3_small
              ]))
          else:
            return datasets.MNIST(
              homedir+'/data',
              train=train,
              download=False,
              transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, )),
                 transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                 transforms.Resize(21) #squeeznet1_1
                 #  transforms.Resize(32) vgg11
                 # transforms.Resize(224) mobilenetv2
                 #   transforms.Resize(224)  mnasnet0_5
                 #   transforms.Resize(29)  densenet121 
                 #transforms.Resize(21) mobilenet_v3_small
              ]))



        if dataset == "cifar":
            if train:
                transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(21), #squeeznet1_1
                 #  transforms.Resize(32) vgg11
                 # transforms.Resize(224) mobilenetv2
                 #   transforms.Resize(224)  mnasnet0_5
                 #   transforms.Resize(29)  densenet121 
                 #transforms.Resize(29) mobilenet_v3_small
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
                return datasets.CIFAR10(
               homedir+'/data',
               train=True,
               download=True,
               transform=transforms_train)
            else:
                transforms_test = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize(21), #squeeznet1_1
                 #  transforms.Resize(32) vgg11
                 # transforms.Resize(224) mobilenetv2
                 #   transforms.Resize(224)  mnasnet0_5
                 #   transforms.Resize(29)  densenet121 
                 #transforms.Resize(29) mobilenet_v3_small
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
                return datasets.CIFAR10(
                homedir+'/data',
                train=False,
                download=True,
                transform=transforms_test) 
            
    def get_train_set(self): 
        train_set = self.fetch_dataset(self.dataset,train=True) 
        train_set_size = int(len(train_set) * 0.9)
        valid_set_size = len(train_set) - train_set_size
        train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])
        size = self.num_workers 
        partition = DataPartitioner(train_set,size) 
        partition = partition.use(self.rank) 
        data_partition = partition.__get_data_partition__() 
        len_partition = len(data_partition)
        train_set = torch.utils.data.DataLoader(data_partition, batch_size=self.batch,shuffle=False)
        valid_set = torch.utils.data.DataLoader(valid_set, batch_size=self.batch,shuffle=False)

        return  train_set,len_partition,valid_set
    
    def get_test_set(self):
        test_set = self.fetch_dataset(self.dataset, train=False)
        test_set = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False )

        return test_set

