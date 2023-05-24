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
from DataManager import DataManager 
import argparse


 













class worker(object): 

    
    def __init__(self,bucket,rank,num_workers,base_name,batch_size,dataset,model_str,loss,optimizer,lr,evaluation,compression,print_necessary,sync_total):  


        """ Constructor of worker Object
        Args
        rank           unique ID of this worker node in the deployment
        num_workers    total number of workers in the deployment 
        base_name      base name of the workers 
        batch_size     size of the batch to be used for training
        model_str      the name of the NN model to be used   
        dataset        the name of the dataset to be used for training
        loss           the name of the loss function to be applied 
        optimizer           the name of the optimizer algorithm  to be applied 

        """
        if torch.cuda.is_available():
                self.device = torch.device("cuda")
        else:
                self.device = torch.device("cpu:0")

        self.rank = rank 
        self.loss_store=[] 
        self.dict_peers= {}
        self.gradient_store = [] 
        self.dataset = dataset 
        self.model_str = model_str 
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
        data_train,data_val = self.load_data_from_s3(model_str,dataset)
        self.train_set,self.len_data_partition = self.get_train_set(data_train) 
        self.valid_set, self.len_data_partition_val = self.get_train_set(data_val) 
        self.data_manager= DataManager(dataset, batch_size, num_workers, rank) 
        self.test_set = self.data_manager.get_test_set()
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

    # get  the peer partition  from the s3 
    def load_data_from_s3(self,model_str,dataset):
           s3 = boto3.client('s3') 
           folder_name = 'worker4/dataloader/' 
           worker_key = f'w-{self.rank}'
           key=f"{folder_name}{dataset}-{model_str}_{worker_key}.pkl"
           print(key)
           obj = s3.get_object(Bucket=self.bucket, Key=key)
           data_set = pickle.loads(obj['Body'].read())
           train_set_size = int(len(data_set) * 0.9)
           valid_set_size = len(data_set) - train_set_size
           train_set_data, valid_set_data = data.random_split(data_set, [train_set_size, valid_set_size])
           return train_set_data,valid_set_data 
    def get_train_set(self,data_partition): 
        
        train_set = torch.utils.data.DataLoader(data_partition, batch_size=self.batch_size,shuffle=True)
        len_partition = len(data_partition)
        return train_set,len_partition 



    def remove_object_from_s3(self,model_str,dataset):
          s3 = boto3.client('s3') 
          folder_name = 'worker4/dataloader/' 
          worker_key = f'w-{self.rank}'
          key=f"{folder_name}{dataset}-{model_str}_{worker_key}.pkl"
          obj = s3.get_object(Bucket=self.bucket, Key = key) 
          obj.delete()



    
    
  

   

    def compute_gradients(self,num_epoch): 
        self.model = self.model.to(self.device) 

   
        print('calcul gradients') 
        
        # Prepare the model for quantization
        computation_time_per_iteration=0 
        compute_gradients_per_batch_memory_dict={}
        
        #compute the number of batches
        number_batches = (self.len_data_partition)//(self.batch_size) 
        print(number_batches)
        print("data for worker {} is : {}".format(self.rank, self.len_data_partition))  
        if not self.print_necessary:
            print(number_batches)
        sum_of_grad=0 
        self.model.train() 
        loss_total=0 
        self.loss_dict={} 
        counter = 0 
        max_norm = 1.0
        computation_time_per_iteration_dict={} 
        cpu_total_time_per_iteration_dict={}
        for batch_idx, (data, targets) in enumerate(self.train_set): 
            with torch.autograd.profiler.profile(enabled=False) as prof:
               

                process = psutil.Process()

                start_computation_time= perf_counter() 
                counter += 1

                if torch.cuda.is_available():
                        data = data.cuda()
                        targets = targets.cuda() 
              
                # Zero the gradients
                self.optimizer.zero_grad()     
                
                # Forward pass
                output = self.model(data) 

                
                loss_result = self.loss(output,targets)  
                self.loss_store.append(loss_result.item())
                # Backward pass
                loss_result.backward()  
               


                loss_total += loss_result.item()
                if (batch_idx+1) % 10 == 0: 
                 print (f' rank {self.rank} Epoch [{num_epoch+1}], Step [{batch_idx+1}/{len(self.train_set)}], Loss: {loss_result.item():.4f}')

                # Quantize and compress the gradients
                grad = [torch.reshape(p.grad,(-1,)) for p in self.model.parameters()]
                gradients= torch.cat(grad).to("cpu")
                print("worker of rank {} gradient of batch{}".format(self.rank,batch_idx))
                sum_of_grad =sum_of_grad+gradients 

                if self.evaluation: 
                    computation_time_per_iteration =  perf_counter()-start_computation_time  
                    computation_time_per_iteration_dict[batch_idx] = computation_time_per_iteration 
                    memory_info = process.memory_info()
                    print(f"Iteration {batch_idx}: Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB") 
                    self.memory_compte_gradients_per_batch_dict[batch_idx] =  memory_info.rss / 1024 / 1024 
 
      
        avg_gradients = sum_of_grad/counter 
        print(avg_gradients)
        self.loss_per_epoch = loss_total / counter 
        self.dict_peers[self.rank]= avg_gradients 
        self.loss_store.append(self.loss_per_epoch) 
       
      

        if self.evaluation: 
                 self.dict_metrics["computation_time_per_iteration"] = computation_time_per_iteration_dict 
                 self.dict_metrics[" compute_gradients_per_batch_memory"] =   compute_gradients_per_batch_memory_dict 

        return avg_gradients 
    def send_grads_to_peer(self, connection_parameters, grads,epoch,comp): 
        process = psutil.Process()

        connection = pika.BlockingConnection(connection_parameters)
        channel = connection.channel()
        queue_name = "peer{}-queue-{}-{}".format(self.rank,self.dataset,self.model_str) 

       
        if self.compression:
                  # compress the gradients before sending 
                  norm, sign_xi_array = comp.compress(grads)
                  norm = norm.unsqueeze(0)
                  buf =  torch.cat((norm,sign_xi_array)) 
                  grads_array = buf.numpy()
        else:
                grads_array = grads.numpy() 

          
        channel.queue_declare(queue=queue_name)
        #channel.queue_purge(queue=queue_name) 
        memory_info = process.memory_info()
        print(f" Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB") 
        self.memory_send_dict[epoch] =  memory_info.rss / 1024 / 1024 


        
        channel.basic_publish(exchange='',
                      routing_key=queue_name,
                      body= grads_array.tobytes()) 
        connection.close() 
        if self.compression:
          return buf.shape
        else:
          return grads.shape 

    def receive_grads_from_peer(self, connection_parameters,dest_ranks,grads_shape,comp):
        Receive_time_per_peer=0
       # initialize a status object to check for received messages
        
        for i in range(dest_ranks):
            if i != self.rank:

              process = psutil.Process()

              print("start receiving") 
              connection = pika.BlockingConnection(connection_parameters)
              channel = connection.channel()
              queue_name = "peer{}-queue-{}-{}".format(i,self.dataset,self.model_str) 
              start = perf_counter() 
              channel.queue_declare(queue=queue_name)
              method_frame, properties, body = channel.basic_get(queue=queue_name)
              if method_frame:

                      
                    # Convert the received byte array back to a NumPy array
                    gradients_array = np.frombuffer(body, dtype=np.float32)
                    received_grads = torch.from_numpy( gradients_array.reshape(grads_shape)) 

                    if self.compression:
                                norm = received_grads[0] 
                                sign_xi_array = received_grads[1:] 
                                #decompress to have the gradients 
                                grads_decompressed = comp.decompress(norm,sign_xi_array) 
                                received_gradients = grads_decompressed  
                                print("finish decompression")
    
                    else:
                      received_gradients =  received_grads 
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
        if torch.cuda.is_available():
             model_cpy = model_cpy.cuda()

        model_cpy.eval()
        with torch.no_grad():
            val_running_correct = 0 

            for  batch_idx, (data, target) in enumerate(test_set):
                      total += target.size(0)
                      outputs = self.model(data) 
                      pred_y = torch.max(outputs, 1)[1].data.squeeze() 
                      val_running_correct += (pred_y == target).sum().item() 
            test_accuracy = 100 * val_running_correct / total 
            with open('accuracy_here'+str(self.rank)+'.txt','a') as file:
            	line = 'acc_sync' + str(test_accuracy) + '\n'
            	file.write(line) 
            return test_accuracy 

    
             
 
    def update_model(self,avg_gradients,model,epoch):    
             process = psutil.Process() 
             process_update = psutil.Process() 
             update_time_start  = perf_counter()

             cpu_update_percent_before = process_update.cpu_percent()

             cur_pos = 0 
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
  epochs = 400   
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
  bucket = "mnist-mobilenet"






  # Define your credentials
  credentials = pika.PlainCredentials('**********', '*********')
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
  connection_parameters = pika.ConnectionParameters('b-*****************************.mq.us-east-1.amazonaws.com',
                                                    5671,
                                                    '/',
                                                    credentials,
                                                    ssl_options=pika.SSLOptions(ssl_context))




  # Establish a connection 
  worker_instance = worker(bucket,rank, size, "worker"+str(rank),batch_size, dataset, model, loss,optimizer,lr = 0.02,evaluation=evaluation,compression=compression,print_necessary=True,sync_total = size) 
  # Initialize the weights of the model
  for m in  worker_instance.modules():
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight, mean=1, std=0.02)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0, std=0.01)
        init.constant_(m.bias, 0)

  connection = pika.BlockingConnection(connection_parameters)
  channel = connection.channel() 
  queue_name ="peer{}-queue-{}-{}".format(rank,worker_instance.dataset,worker_instance.model_str) 

  channel.queue_delete(queue= queue_name)  
  channel.queue_declare(queue=queue_name)  
  channel.queue_delete(queue= "sync_queue_name")  
  channel.queue_declare(queue="sync_queue_name") 

  connection.close()
  comp = compressors.QSGDWECModCompressor(worker_instance.device) 
  scheduler = ReduceLROnPlateau(worker_instance.optimizer,mode='min', factor=0.1, patience=5, verbose=True,min_lr= 1e-6)
        

  
  tracemalloc.start()

  for epoch in range(epochs): 
    start_training_epoch_time = perf_counter()

    psutil.cpu_percent(interval=None, percpu=False)
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel() 
    channel.queue_delete(queue= queue_name)  
    channel.queue_declare(queue=queue_name) 
	  connection.close()

    start_compute_gradients_time = perf_counter()  
    gradients,memory_percentage_compute_gradients,cpu_percentage_compute_gradients = measure_memory_usage( worker_instance.compute_gradients,epoch) 
    compute_gradients_time = perf_counter() - start_compute_gradients_time 
    time_dict_compute[epoch]  = compute_gradients_time 
    loss_dict[epoch] = worker_instance.loss_per_epoch 
  


    memory_profile_dict_compute[epoch]  = memory_percentage_compute_gradients 
    cpu_profile_dict_compute[epoch]  = cpu_percentage_compute_gradients 


    print("gradient was successufully computed") 
    
    #start recording send time 
    send_start = perf_counter() 
    buffer_shape, memory_percentage_send_gradients, cpu_percentage_send_gradients= measure_memory_usage( worker_instance.send_grads_to_peer ,  connection_parameters,gradients,epoch,comp) 
    memory_profile_dict_send[epoch]  = memory_percentage_send_gradients 
    cpu_profile_dict_send[epoch]  = cpu_percentage_send_gradients 
    SendTime = perf_counter()-send_start   
    send_time_dict[epoch] = SendTime 

    print("gradient sent") 

    while(len(worker_instance.dict_peers)<worker_instance.sync_total):
        receive_start = perf_counter() 
        print(f"le synch total is {worker_instance.sync_total} and {len(worker_instance.dict_peers)}")
    
        result,memory_percentage_receive_gradients,cpu_percentage_receive_gradients = measure_memory_usage( worker_instance.receive_grads_from_peer, connection_parameters,size,buffer_shape,comp) 
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
            bandwith_network = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            bandwith_per_training_step_dict[epoch] = bandwith_network 
            print(perf_counter()-start_training_epoch_time)
            computation_time_per_training_step =  perf_counter()-start_training_epoch_time  
            computation_time_per_training_step_dict[epoch] =  computation_time_per_training_step   
            cpu_usage_per_training_step = psutil.cpu_percent(interval=None, percpu=False)
            cpu_usage_per_training_step_dict[epoch] =  cpu_usage_per_training_step 
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
    cpu_convergence_percent_before =   process_convergence.cpu_percent()
    convergence_detection_time_start  = perf_counter()

    val_loss = worker_instance.validate() 
    memory_info =   process_convergence.memory_info()
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
         
    cpu_convergence_percent_after=   process_convergence.cpu_percent() 
    cpu_convergence_percent = cpu_convergence_percent_after - cpu_convergence_percent_before 
    convergence_dict_metrics_cpu_usage[epoch] = cpu_convergence_percent 
    worker_instance.dict_metrics[" convergence_cpu_usage_per_training_step"] = convergence_dict_metrics_cpu_usage 


    convergence_memory_usage = memory_info.rss / 1024 / 1024 
    convergence_dict_metrics_memory_usage[epoch] = convergence_memory_usage  
    worker_instance.dict_metrics[" convergence_memory_usage_per_training_step"] =    convergence_dict_metrics_memory_usage 
    convergence_detection_time =     perf_counter() -  convergence_detection_time_start 
    convergence_dict_metrics[epoch] = convergence_detection_time  
    worker_instance.dict_metrics[" convergence_time_per_training_step"] =  convergence_dict_metrics 
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel() 
    channel.basic_publish(exchange='', routing_key="sync_queue_name", body="rank")

    # sync barrier
    sleep(1)
    queue_sync_value = channel.queue_declare(queue="sync_queue_name", passive=True)
    message_count = queue_sync_value.method.message_count
    while message_count != size :

        queue_sync_value = channel.queue_declare(queue="sync_queue_name", passive=True)
        message_count = queue_sync_value.method.message_count
        print(f"still synchronising {message_count} ....")
        sleep(1)



    sleep(3)
    channel.queue_purge(queue="sync_queue_name") 

  # Open the file for each worker
  with open(worker_instance.filename, 'a') as f:
     # Write the metrics dictionary  to the file
     f.write(json.dumps(worker_instance.dict_metrics,indent=2)) 
     f.write('\n')

  # Synchronize all processes








main()
