"""
This work was done by Amine Barrak and Ranim Trabelsi

"""


import uuid
from random import Random
import pika
import ssl
import torch 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import boto3 
import dill as pickle
import json

def select_loss(loss_fn):

  losses = {"NLL":nn.NLLLoss,"cross_entropy":nn.CrossEntropyLoss} 
  if loss_fn in losses.keys(): 
    return losses[loss_fn]() 
  else: 
    print("The selected loss function is undefined, available losses are: ", losses.keys())

def select_model(dataset,model):
  models={ 'resnet18':torchvision.models.resnet18,
                'resnet34':torchvision.models.resnet34,
                'resnet50':torchvision.models.resnet50,
                'resnet152':torchvision.models.resnet152,
                    'inception':torchvision.models.inception_v3,
                     'vgg16':torchvision.models.vgg16,
                     'vgg19':torchvision.models.vgg19, 
                    'vgg11':torchvision.models.vgg11,
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
    print("The specifiedmodel is undefined, available models are: ", models.keys())

  return model
def select_optimizer(model,optimizer,lr):
  optimizers={'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw':optim.AdamW,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad} 
  if optimizer in optimizers.keys():
    return optimizers[optimizer](model.parameters(),lr=lr)

def load_data_from_s3(bucket,dataset,rank,model_str,batch_rank):
           s3 = boto3.client('s3')
           worker_key = f'w-{rank}'
           folder_name = 'worker12/batch1024/'
           key=f"{folder_name}{dataset}-{model_str}_{worker_key}_{batch_rank}.pkl"
           obj = s3.get_object(Bucket=bucket, Key=key)
           data = pickle.loads(obj['Body'].read())
           return data
def get_train_set(data_partition,batch):

        train_set = torch.utils.data.DataLoader(data_partition, batch_size=batch,shuffle=False)
        len_partition = len(data_partition)
        print(len_partition)


        return  train_set,len_partition


def send_grads_to_peer(queue_name, connection_parameters, grads): 
        def upload_to_s3(bucket_name, object_key, data):
            s3 = boto3.client('s3')
            s3.put_object(Bucket=bucket_name, Key=object_key, Body=data)

        connection = pika.BlockingConnection(connection_parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)
        s3_bucket = 's3-bucket-gradient'
        object_key = f"tensors2/{uuid.uuid4()}.pt"

        # Serialize the tensor into memory
        data = pickle.dumps(grads)

        # Upload the serialized tensor to S3
        try:
            upload_to_s3(s3_bucket, object_key, data)
        except Exception as e:
            print(f"Error uploading tensor to S3: {e}")
          # Send the S3 object key in the message body
        channel.basic_publish(exchange='', routing_key=queue_name, body=object_key)

        connection.close()


def compute_gradients(rank,batch_data,model,loss,optimizer):

        #compute the number of batches
        loss_store= []
        dict_peers = {}
        model.train()
        data, target = batch_data
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        loss_result = loss(output,target)

        # Backward pass
        loss_result.backward()
        # Quantize and compress the gradients
        grad = [torch.reshape(p.grad,(-1,)) for p in model.parameters()]
        gradients= torch.cat(grad).to("cpu")
        



        return loss_result.item(),gradients

def save_to_s3(dataset, bucket, key):
    s3_client = boto3.client("s3")
    dataset_pkl = pickle.dumps(dataset)
    s3_client.put_object(Bucket=bucket, Key=key, Body=dataset_pkl)


def lambda_handler(event, context):

    # Parse input parameters from the event
    rank = event['rank']
    batch_rank = event ['batch_rank']
    dataset = event['dataset']
    model_str = event['model_str']
    optimiser = event['optimiser']
    optimiser_lr = event['lr']
    loss = event['loss']
    queue_name = f"lambda-vgg11-queue-{rank}"
    bucket = f"{dataset}-{model_str}"
    
    credentials = pika.PlainCredentials('************', '************')
    ssl_context = ssl.create_default_context(cafile='**********.pem')
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connection_parameters = pika.ConnectionParameters('*****************************.mq.us-east-1.amazonaws.com',
                                                    5671,
                                                    '/',
                                                    credentials,
                                                    ssl_options=pika.SSLOptions(ssl_context))



    model = select_model(dataset, model_str)
    optimizer =select_optimizer(model,optimiser,optimiser_lr)
    loss = select_loss(loss)
    #Prepare the response
    
    data_batch = load_data_from_s3(bucket,dataset,rank,model_str,batch_rank)
    


    #the retrun of compute_gradients contain the loss and the batch_gradient
    loss_result, gradients_data = compute_gradients(rank,data_batch,model,loss,optimizer)
    #size_gradients = gradients_data[1].numel() * gradients_data[1].element_size()

    #tab = [gradients_data, gradients_data.shape, loss_result]
    
    #send_grads_to_peer(queue_name, connection_parameters, tab)


    response = {
        'statusCode': 200,
        'body': f"gradient was usccessfully saved to the rabbitMQ"
    }

    return response
