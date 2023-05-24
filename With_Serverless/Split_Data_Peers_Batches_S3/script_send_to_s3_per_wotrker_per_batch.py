# -*- coding: utf-8 -*-
"""
This work was done by Amine Barrak and Ranim Trabelsi

"""

import boto3
import dill as pickle
import gzip
import numpy as np
from urllib.request import urlretrieve
import torch 
from torchvision import datasets, transforms
import torch.optim as optim 
import torch.utils.data as data 
import pathlib 
from random import Random 
import sys





def fetch_dataset(dataset,train=True):
        homedir = str(pathlib.Path.home()) 
        if dataset == "MNIST":
          if train:
            return datasets.MNIST(
             homedir+'/data',
              train=train,
              download=True,
              transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, )),
                 transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                 #transforms.Resize(21) #squeeznet1_1
                 #transforms.Resize(32) #vgg11
                 transforms.Resize(224) #mobilenetv2
                 #transforms.Resize(224)  #mnasnet0_5
                 #transforms.Resize(29)  #densenet121 
                 #transforms.Resize(29) #mobilenet_v3_small
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
                 #transforms.Resize(21) #squeeznet1_1
                 #transforms.Resize(32)#vgg11
                 transforms.Resize(224) #mobilenetv2 
                 #transforms.Resize(224)  #mnasnet0_5 
                 #transforms.Resize(29)  #densenet121 
                 #transforms.Resize(29) #mobilenet_v3_small
              ]))



        if dataset == "CIFAR10":
            if train:
                transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(21), #squeeznet1_1
                  #transforms.Resize(32) ,  #vgg11
                  #transforms.Resize(224) , #mobilenetv2
                    #transforms.Resize(224) ,  #mnasnet0_5
                 #transforms.Resize(29) ,  #densenet121 
                 #transforms.Resize(29) ,  #mobilenet_v3_small
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
                #transforms.Resize(32) , #vgg11
                 #transforms.Resize(224) , mobilenetv2
                  #transforms.Resize(224) ,  #mnasnet0_5
                   #transforms.Resize(29) ,   #densenet121 
                 #transforms.Resize(29) , #mobilenet_v3_small
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
                return datasets.CIFAR10(
                homedir+'/data',
                train=False,
                download=True,
                transform=transforms_test) 
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
    
    


def create_bucket(bucket_name, region_name):
    s3_client = boto3.client("s3", region_name=region_name)
    response = s3_client.list_buckets()
    existing_buckets = [bucket["Name"] for bucket in response["Buckets"]]

    if bucket_name not in existing_buckets:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region_name},
        )
        print(f"Bucket {bucket_name} created.")
    else:
        print(f"Bucket {bucket_name} already exists.")



def save_to_s3(dataset, bucket, key):
    s3_client = boto3.client("s3")
    dataset_pkl = pickle.dumps(dataset)
    s3_client.put_object(Bucket=bucket, Key=key, Body=dataset_pkl)

if __name__ == "__main__": 

  num_workers = int(sys.argv[1]) 
  num_batch = int(sys.argv[3])
  bucket = int(sys.argv[2]) 
  dataset = int(sys.argv[4]) 
  

 
 
  folder_name = 'batch/'

  for i in range(num_workers):
        data  = fetch_dataset(dataset)
        partition = DataPartitioner(data ,num_workers) 
        partition = partition.use(i) 
        data_partition = partition.__get_data_partition__() 
        print(len(data_partition)) 
        batch_size = len(data_partition)//num_batch
        train_set = torch.utils.data.DataLoader(data_partition, batch_size=batch_size,shuffle=False)
        worker_key = f'w-{i}'
        region_name = "us-east-1"  
        create_bucket(bucket, region_name) 
        for batch_idx, data in enumerate(train_set):
               key = folder_name+bucket+"_"+worker_key+"_"+str(batch_idx)+".pkl"
               save_to_s3(data, bucket, key)
