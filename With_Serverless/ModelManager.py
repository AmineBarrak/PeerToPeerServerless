# -*- coding: utf-8 -*-
"""
This work was done by Amine Barrak and Ranim Trabelsi

"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets 
from torch.distributed.rpc import RRef, remote, rpc_async 
from time import sleep, time
import sys

def select_loss(loss_fn):

  losses = {"NLL":nn.NLLLoss,"cross_entropy":nn.CrossEntropyLoss} 
  if loss_fn in losses.keys(): 
    return losses[loss_fn]() 
  else: 
    print("The selected loss function is undefined, available losses are: ", losses.keys())

def select_model(dataset,model,device):
  models={          'vgg11':torchvision.models.vgg11,
	            	'squeeznet11':torchvision.models.squeezenet1_1,
	             	'mobilenetv2':torchvision.models.mobilenet_v2,
	             	'mnasnet05':torchvision.models.mnasnet0_5,
		             'densenet121': torchvision.models.densenet121, 
		             'mobilenet-v3-small': torchvision.models.mobilenet_v3_small
		  
		            
      
  } 
  datasets= { "cifar":10,  "mnist":10} 
  if dataset in datasets.keys():
    num_classes = datasets[dataset] 
  else:
    print("The specified dataset is undefined, available datasets are: ", datasets.keys())


  if model in models.keys():
    model = models[model](num_classes=num_classes) 
  else:
    print("The specified model is undefined, available models are: ", models.keys())

  model = model.to(device) 
  return model

def select_optimizer(model,optimizer,lr):
  optimizers={'sgd': optim.SGD,
		'adam': optim.Adam,
		'adamw':optim.AdamW,
		'rmsprop': optim.RMSprop,
		'adagrad': optim.Adagrad} 
  if optimizer in optimizers.keys():
    return optimizers[optimizer](model.parameters(),lr=lr)