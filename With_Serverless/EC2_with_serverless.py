# -*- coding: utf-8 -*-
"""
This work was done by Amine Barrak and Ranim Trabelsi

"""

import ssl
import pika
import torch 
import ModelManager 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import sleep, time 
import torch.utils.data as data
import sys 
import boto3 
import os 
import numpy as np
import dill as pickle 
import copy 
import compressors 
from time import perf_counter
import psutil 
import json 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import tracemalloc 
import concurrent.futures
import cProfile 
import pstats 
from collections.abc import Iterable 
from concurrent.futures import wait 
import requests
import uuid
import io












class worker(object): 

    
    def __init__(self,bucket,rank,num_workers,base_name,batch_size,dataset,model_str,loss,optimizer,lr,evaluation,compression,print_necessary,sync_total):  

        
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        self.device = torch.device("cpu:0")

        self.rank = rank 
        self.loss_store=[] 
        self.dict_peers= {}
        self.gradient_store = [] 
        self.model =ModelManager.select_model(dataset,model_str,self.device)
        self.base_name = base_name 
        self.num_workers = num_workers 
        self.batch_size = batch_size 
        self.loss = ModelManager.select_loss(loss)
        self.optimizer = ModelManager.select_optimizer(self.model,optimizer,lr) 
        self.dict_metrics={} 
        self.evaluation = evaluation 
        self.compression = compression 
        self.print_necessary = print_necessary 
        self.filename =  "worker_" + str(self.rank) + ".txt" 
        self.sync_total = sync_total 
        self.memory_send_dict = {}
        self.memory_compte_gradients_per_batch_dict = {} 
        self.memory_receive_dict = {} 
        self.memory_update_dict = {} 
        self.time_update_dict = {} 
        self.update_dict_cpu = {}
 
  
     
        self.bucket = bucket 
       
        self.torch_to_numpy_dtype_dict = {
                  torch.float32: np.float32,
                  torch.float64: np.float64,
                  torch.float16: np.float16,
                  torch.int8: np.int8,
                  torch.int16: np.int16,
                  torch.int32: np.int32,
                  torch.int64: np.int64,
                  torch.uint8: np.uint8
                  }

     
        print('worker has been created')


    


    
    
    




    def compute_gradients(self,num_epoch, connection_parameters, queue_name, batch_number): 
        def check_execution_status(execution_arn):
            client = boto3.client(
                "stepfunctions",
            )
            while True:
                response = client.describe_execution(
                    executionArn=execution_arn
                )

                status = response["status"]
                print(f"Execution status: {status}")

                if status in ["SUCCEEDED", "FAILED", "TIMED_OUT"]:
                    final = perf_counter()
                    break
                else:
                    sleep(2)  # Adjust the sleep time as needed

            return status , final

        def invoke_aws_setp_function(input, arn):
            state_machine_arn = arn
            # Initialize the boto3 client
            client = boto3.client(
                "stepfunctions",
            )

            # Define the input for your state machine
            

            # Invoke the Step Function
            response = client.start_execution(
                stateMachineArn=state_machine_arn,
                input=json.dumps(input),
            )

            # Print the response
            print(response)
        

            execution_arn = response["executionArn"]

        

            status , final = check_execution_status(execution_arn)

            return status, final

        
        def download_from_s3(bucket_name, object_key):
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body'].read()


        connection = pika.BlockingConnection(connection_parameters)
        channel = connection.channel()

        channel.queue_purge(queue=queue_name) 


        print('calcul gradients') 
        # Prepare the model for quantization

        state_machine_arn = "arn:aws:states:us-east-1:************:stateMachine:batch_processing"
        input = {    
        "worker": self.rank,
        "batch_number":batch_number
        } 

        start = perf_counter()

        status, final = invoke_aws_setp_function(input, state_machine_arn)

        print (f"the total training batches is :{final - start}")
        if status == "SUCCEEDED":

            pass

        else:
            os.exit() 

         
        # Declare the queue to make sure it exists
        channel.queue_declare(queue=queue_name)

        # Callback function to process messages
        def callback(ch, method, properties, body):
            print(f"Received message: {body.decode('utf-8')}")
            object_key = body.decode()
            s3_bucket = 's3-bucket-gradient'
            raw_data = download_from_s3(s3_bucket, object_key)

            tab = pickle.loads(raw_data)

            return tab
        tab_gradients = []


        # Read messages until the queue is empty
        while True:
            method_frame, properties, body = channel.basic_get(queue=queue_name, auto_ack=True)
            if method_frame:
                tab = callback(channel, method_frame, properties, body)
                print(f"the shape of tab received is :{tab[0].shape}")
                print(f"the shape of tab[1] received is :{tab[1]}")
                tab_gradients.append(tab[0])


            else:
                print("Queue is empty. Finalizing connection.")
                break


        sum_of_grad = torch.zeros_like(tab_gradients[0])
            
        for grad in tab_gradients: 
            sum_of_grad += grad 
        print(f"the shape of sum_of_grad is :{sum_of_grad.shape}")
        avg_gradients = (sum_of_grad)/(len(tab_gradients))

        # Close the connection
        connection.close()


        print(f"the shape is :{avg_gradients.shape}")

        self.dict_peers[self.rank]= avg_gradients 

        return avg_gradients 



    def send_grads_to_peer(self, connection_parameters, grads,epoch): 
        def upload_to_s3(bucket_name, object_key, data):
            s3 = boto3.client('s3')
            s3.put_object(Bucket=bucket_name, Key=object_key, Body=data)
        process = psutil.Process()

        connection = pika.BlockingConnection(connection_parameters)
        channel = connection.channel()
        queue_name = "peer{}-queue_vgg11".format(self.rank) 
        channel.queue_declare(queue=queue_name)
        memory_info = process.memory_info()
        print(f" Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB") 
        self.memory_send_dict[epoch] =  memory_info.rss / 1024 / 1024 
        s3_bucket = 's3-bucket-gradient'
        object_key = f"tensors2/{uuid.uuid4()}.pt"

        # Serialize the tensor into memory
        buffer = io.BytesIO()
        torch.save(grads, buffer)
        data = buffer.getvalue()
        # Upload the serialized tensor to S3
        try:
            upload_to_s3(s3_bucket, object_key, data)
        except Exception as e:
            print(f"Error uploading tensor to S3: {e}")
          # Send the S3 object key in the message body
        channel.queue_purge(queue=queue_name) 
        channel.basic_publish(exchange='', routing_key=queue_name, body=object_key)

        connection.close()
       

    def receive_grads_from_peer(self, connection_parameters,dest_ranks,grads_shape):
        Receive_time_per_peer=0
        process = psutil.Process()
       # initialize a status object to check for received messages
        def download_from_s3(bucket_name, object_key):
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body'].read()
        
        for i in range(dest_ranks):
            # sleep(5)
            if i != self.rank:

              process = psutil.Process()

              print("start receiving") 
              connection = pika.BlockingConnection(connection_parameters)
              channel = connection.channel()
              queue_name = "peer{}-queue_vgg11".format(i) 
              start = perf_counter() 
              channel.queue_declare(queue=queue_name)
              method_frame, properties, body = channel.basic_get(queue=queue_name)
              if method_frame:
                  object_key = body.decode()
                  s3_bucket = 's3-bucket-gradient'
                  raw_data = download_from_s3(s3_bucket, object_key)
                  # Deserialize the tensor
                  buffer = io.BytesIO(raw_data)
                  received_gradients= torch.load(buffer)
                  # Print the deserialized tensor
                  print("Tensor received from queue", queue_name) 
                  memory_info = process.memory_info()
                  print(f" Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB") 
                  self.memory_receive_dict[i] =  memory_info.rss / 1024 / 1024 
                  # store the gradients received in dict_peers 
                  self.dict_peers[i]= received_gradients 
                  print(f"end receiving from {i}")
                  print(f"the dict len is becoming {len(self.dict_peers)}") 
              connection.close()


              
              
                
        


    def avg_grad(self):
        print("computing average gradients") 
        sum_of_grad = torch.zeros_like(self.dict_peers[0])
        for rank, grad in self.dict_peers.items(): 
           sum_of_grad += grad 
           avg_gradients = (sum_of_grad)/(len(self.dict_peers))
        self.dict_peers.clear()
        return avg_gradients 
       


    
    def compute_accuracy(self,test_set): 
       
        correct = 0
        total = 0
        model_cpy = copy.deepcopy(self.model)
        # if torch.cuda.is_available():
        #     model_cpy = model_cpy.cuda()

        model_cpy.eval()
        with torch.no_grad():
            val_running_correct = 0 

            for  batch_idx, (data, target) in enumerate(test_set):
                      total += target.size(0)
                      outputs = self.model(data) 
                      pred_y = torch.max(outputs, 1)[1].data.squeeze() 
                      val_running_correct += (pred_y == target).sum().item() 
            test_accuracy = 100 * val_running_correct / total 
            return test_accuracy 

    
             
 
      
    def update_model(self,avg_gradients,model,epoch):    
             process = psutil.Process()
             update_time_start  = perf_counter()


             cur_pos = 0 
             process_update = psutil.Process()

             cpu_update_percent_before = process_update.cpu_percent()

             for param in self.model.parameters():
                   param.grad = torch.reshape(torch.narrow(avg_gradients,0,cur_pos,param.nelement()), param.size()).detach()
                   cur_pos = cur_pos+param.nelement()  
             self.optimizer.step() 
             memory_info = process.memory_info()
             print(f" Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB") 
             self.memory_update_dict[epoch] =  memory_info.rss / 1024 / 1024 
             update_time_finish = perf_counter() -  update_time_start  
             self.time_update_dict[epoch] =   update_time_finish 
             cpu_update_percent_after= process_update.cpu_percent() 
             cpu_convergence_percent = cpu_update_percent_after - cpu_update_percent_before 
             self.update_dict_cpu[epoch] = cpu_convergence_percent 



    def validate(self):
          # Settings
          self.model.eval()
          loss_total = 0
          counter = 0
          # Test validation data
          with torch.no_grad():
            for inputs,targets in self.valid_set:
               counter += 1
               outputs = self.model(inputs)
               loss = self.loss(outputs,targets)
               loss_total += loss.item()

            return loss_total / counter 


def measure_memory_usage(function, *args, **kwargs):
    # Get current memory usage

    process = psutil.Process()

    cpu_percentage_before = process.cpu_percent()  

    tracemalloc.start()  # Start tracing memory allocations

    # Run the function and measure the memory usage
    result = function(*args, **kwargs) 
    # Your code goes here 

    current, peak = tracemalloc.get_traced_memory() 
    mem_percentage = (peak / psutil.virtual_memory().total) * 100
    print(f"Current memory usage: {current / 1048576:.2f} MiB")
    print(f"Peak memory usage: {peak / 1048576:.2f} MiB") 
    current = current / 1048576 
    peak = peak / 1048576 
    cpu_percentage_after = process.cpu_percent()  
    cpu_diff = cpu_percentage_after - cpu_percentage_before 
    # Get memory usage after the function execution
    tracemalloc.stop()  # Stop tracing memory allocations


    # Calculate memory usage
    print(mem_percentage)
    print("{}: consumed memory: {:,} by MiB ({:.4f}% of total system memory), cpu_percentage:{}".format(
                function.__name__,
                 peak,
                mem_percentage, 
                cpu_diff 
               

            )) 
    return  result,peak,cpu_diff
def main():
  # Initialize MPI


  # assert comm.size == 3


  parser = argparse.ArgumentParser()
  parser.add_argument('--size',type=int)
  parser.add_argument('--rank',type=int) 
  parser.add_argument('--batch_size',type=int) 

  parser.add_argument('--dataset',type=str)
  parser.add_argument('--model',type=str) 
  parser.add_argument('--optimizer',type=str)
  parser.add_argument('--loss',type=str) 
  parser.add_argument('--evaluation',type=bool)
  parser.add_argument('--compression',type=bool)
  FLAGS = parser.parse_args(sys.argv[1:])
  size  = FLAGS.size 
  rank = FLAGS.rank 
  dataset = FLAGS.dataset
  model  = FLAGS.model
  optimizer   = FLAGS.optimizer
  loss  = FLAGS.loss 
  rank  = FLAGS.evaluation 
  size  = FLAGS.compression 
 
 
  print("the worker rank is :", rank)
  print(size) 
  epochs= 4 
  computation_time_per_training_step_dict = {}
  cpu_usage_per_training_step_dict = {}
  memory_usage_per_training_step_dict = {}
  send_time_dict = {}  
  convergence_dict_metrics = {}  
  convergence_dict_metrics_memory_usage = {}  
  convergence_dict_metrics_cpu_usage = {}  

  receive_time_dict = {} 
  bandwith_per_training_step_dict = {} 
  total_cpu_per_training_step_dict = {} 
  memory_profile_dict_compute = {}
  cpu_profile_dict_compute = {}
  memory_profile_dict_send = {}
  cpu_profile_dict_send={}
  memory_profile_dict_receive = {}
  cpu_profile_dict_receive = {} 
  time_dict_compute = {} 
  loss_dict = {} 

  best_val_loss = None 
  min_delta = 0 
  counter = 0 
  early_stop = False 
  patience = 10 
  bucket = "mnist-vgg11"






  # Define your credentials
  credentials = pika.PlainCredentials('********', '**********')
  # Create an SSL context 
  url = 'https://www.amazontrust.com/repository/AmazonRootCA1.pem'
  response = requests.get(url)

  if response.status_code == 200:
      with open('AmazonRootCA1.pem', 'wb') as f:
          f.write(response.content)
          print("AmazonRootCA1.pem file downloaded successfully.")
  else:
      print("Failed to download the file. Status code:", response.status_code)
  ssl_context = ssl.create_default_context(cafile='AmazonRootCA1.pem')
  ssl_context.check_hostname = False
  ssl_context.verify_mode = ssl.CERT_NONE

  # Set up connection parameters with the SSL context
  connection_parameters = pika.ConnectionParameters('b-************************************.mq.us-east-1.amazonaws.com',
                                                    5671,
                                                    '/',
                                                    credentials,
                                                    ssl_options=pika.SSLOptions(ssl_context))




  # Establish a connection
  connection = pika.BlockingConnection(connection_parameters)
  channel = connection.channel() 
  queue_name = "peer{}-queue_vgg11".format(rank)  
  queue_name_lambda = f"lambda-vgg11-queue-{rank}"
  channel.queue_delete(queue= queue_name)  
  channel.queue_declare(queue=queue_name) 
  channel.queue_declare(queue=queue_name_lambda) 
  connection.close()
  worker_instance = worker(bucket,rank, size, "worker"+str(rank),batch_size, dataset, model, loss,optimizer,lr = 0.02,evaluation=evaluation,compression=compression,print_necessary=True,sync_total = size) 
  comp = compressors.QSGDWECModCompressor(worker_instance.device) 
  scheduler = ReduceLROnPlateau(worker_instance.optimizer,mode='min', factor=0.1, patience=5, verbose=True,min_lr= 1e-6)
        

  
  tracemalloc.start()
  profiler = cProfile.Profile() 

  for epoch in range(epochs): 
    profiler.enable() 
    start_training_epoch_time = perf_counter()
    process = psutil.Process()

    psutil.cpu_percent(interval=None, percpu=False)
    start_compute_gradients_time = perf_counter() 
    gradients,memory_percentage_compute_gradients,cpu_percentage_compute_gradients = measure_memory_usage( worker_instance.compute_gradients,epoch, connection_parameters, queue_name_lambda, batch_number) 
    compute_gradients_time = perf_counter() - start_compute_gradients_time 
    time_dict_compute[epoch]  = compute_gradients_time 
  


    memory_profile_dict_compute[epoch]  = memory_percentage_compute_gradients 
    cpu_profile_dict_compute[epoch]  = cpu_percentage_compute_gradients 


    print("gradient was successufully computed") 
    #start recording send time 
    
    send_start = perf_counter() 
    buffer_shape, memory_percentage_send_gradients, cpu_percentage_send_gradients= measure_memory_usage( worker_instance.send_grads_to_peer ,  connection_parameters,gradients,epoch) 
    memory_profile_dict_send[epoch]  = memory_percentage_send_gradients 
    cpu_profile_dict_send[epoch]  = cpu_percentage_send_gradients 



    SendTime = perf_counter()-send_start   
    send_time_dict[epoch] = SendTime 

    print("gradient sent") 

    while(len(worker_instance.dict_peers)<worker_instance.sync_total):
        receive_start = perf_counter() 
        print(f"le synch total is {worker_instance.sync_total} and {len(worker_instance.dict_peers)}")
    
        result,memory_percentage_receive_gradients,cpu_percentage_receive_gradients = measure_memory_usage( worker_instance.receive_grads_from_peer, connection_parameters,size,buffer_shape) 
        memory_profile_dict_receive[epoch]  = memory_percentage_receive_gradients 
        cpu_profile_dict_receive[epoch]  = cpu_percentage_receive_gradients 


        receive_total_time = perf_counter()-receive_start  
        receive_time_dict[epoch] = receive_total_time  
        print("gradient received")




        sleep(2)
        print("still waiting for gradients from other peers")
        print(f"the dict len is {len(worker_instance.dict_peers)}")

    avg_gradients  = worker_instance.avg_grad() 
    worker_instance.update_model(avg_gradients,worker_instance.model,epoch) 
    
    profiler.disable()

    if worker_instance.evaluation ==True: 
            stats  = pstats.Stats(profiler)
            total_cpu_per_training_step = stats.total_tt 
            total_cpu_per_training_step_dict[epoch] = total_cpu_per_training_step 
            bandwith_network = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            bandwith_per_training_step_dict[epoch] = bandwith_network 
            print(perf_counter()-start_training_epoch_time)
            computation_time_per_training_step =  perf_counter()-start_training_epoch_time  
            computation_time_per_training_step_dict[epoch] =  computation_time_per_training_step   
            cpu_usage_per_training_step = psutil.cpu_percent(interval=None, percpu=False)
            cpu_usage_per_training_step_dict[epoch] =  cpu_usage_per_training_step 
            current, peak = tracemalloc.get_traced_memory()
            print("Memory usage of iteration {}: {:.2f} MB".format(epoch, peak / 1024 ** 2))
            memory_usage_per_training_step_dict[epoch] =  peak / 1024 ** 2  
            worker_instance.dict_metrics["send_time"] = send_time_dict 
            worker_instance.dict_metrics["receive_time"] =receive_time_dict 
            worker_instance.dict_metrics["computation_time_per_training_stept"] =   computation_time_per_training_step_dict
            worker_instance.dict_metrics["cpu_usage_per_training_step"] = cpu_usage_per_training_step_dict 
            worker_instance.dict_metrics["memory_usage_per_training_step"] = memory_usage_per_training_step_dict 
            worker_instance.dict_metrics["bandwith_per_training_step"] =  bandwith_per_training_step_dict 
            worker_instance.dict_metrics[" total_cpu_per_training_step"] =  total_cpu_per_training_step_dict 
            worker_instance.dict_metrics[" total_memory_per_training_step_compute_profile"] =    memory_profile_dict_compute 
            worker_instance.dict_metrics[" total_cpu_per_training_step_compute_profile"] =   cpu_profile_dict_compute 


            worker_instance.dict_metrics["total_memory_per_training_step_send_profile"] =  memory_profile_dict_send  
            worker_instance.dict_metrics["total_cpu_per_training_step_send_profile"] =  cpu_profile_dict_send  

 
            worker_instance.dict_metrics[" total_memory_per_training_step_receive_profile"] =  memory_profile_dict_receive 
            worker_instance.dict_metrics[" total_cpu_per_training_step_receive_profile"] =  cpu_profile_dict_receive  
            worker_instance.dict_metrics["compute_gradients_time"] =  time_dict_compute 
            worker_instance.dict_metrics["loss"] = loss_dict 
            worker_instance.dict_metrics["ram gradients per batch "] = worker_instance.memory_compte_gradients_per_batch_dict 
            worker_instance.dict_metrics["ram send "] = worker_instance.memory_send_dict 
            worker_instance.dict_metrics["ram receive "] = worker_instance.memory_receive_dict 
            worker_instance.dict_metrics["ram update   "] = worker_instance.memory_update_dict 
            worker_instance.dict_metrics["time update   "] = worker_instance.time_update_dict  
            worker_instance.dict_metrics["cpu update   "] = worker_instance.update_dict_cpu  
 

            







            

  

    tracemalloc.stop()

    process_convergence = psutil.Process()
    cpu_convergence_percent_before = process_convergence.cpu_percent()
    convergence_detection_time_start  = perf_counter()
    

    val_loss = worker_instance.validate() 
    memory_info = process.memory_info()
    print(f" Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


    if best_val_loss == None:
            best_val_loss = val_loss
    elif val_loss - best_val_loss < min_delta: 
            best_val_loss = val_loss
            # reset counter if validation loss improves
            counter = 0 
    elif val_loss - best_val_loss >= min_delta:
            counter += 1
            print(f"INFO: Early stopping counter {counter} of {patience}")
            if counter >= patience:
                print('INFO: Early stopping')
                early_stop = True 

    if early_stop: 
         break  
    cpu_convergence_percent_after= process_convergence.cpu_percent() 
    cpu_convergence_percent = cpu_convergence_percent_after - cpu_convergence_percent_before 
    convergence_dict_metrics_cpu_usage[epoch] = cpu_convergence_percent 
    worker_instance.dict_metrics[" convergence_cpu_usage_per_training_step"] = convergence_dict_metrics_cpu_usage 


    convergence_memory_usage = memory_info.rss / 1024 / 1024 
    convergence_dict_metrics_memory_usage[epoch] = convergence_memory_usage  
    worker_instance.dict_metrics[" convergence_memory_usage_per_training_step"] =    convergence_dict_metrics_memory_usage 
    #print(worker_instance.dict_metrics) 
    convergence_detection_time =     perf_counter() -  convergence_detection_time_start 
    convergence_dict_metrics[epoch] = convergence_detection_time  
    worker_instance.dict_metrics[" convergence_time_per_training_step"] =  convergence_dict_metrics 
  #print(worker_instance.dict_metrics) 
  # Open the file for each worker
  with open(worker_instance.filename, 'a') as f:
     # Write the metrics dictionary  to the file
     f.write(json.dumps(worker_instance.dict_metrics,indent=2)) 
     f.write('\n')

  # Synchronize all processes








main()
