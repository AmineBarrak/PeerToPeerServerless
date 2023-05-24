<h1>Exploring the Impact of Serverless Computing on Peer-To-Peer Training for Machine Learning</h1>
<p>The primary objective of this project is to compare and evaluate the potential impact of incorporating serverless computing into peer-to-peer (P2P) training for machine learning. We aim to demonstrate the benefits of utilizing serverless architectures in comparison to traditional P2P training approaches. Our proposed study comprises two core architectures:
<ul>
  <li><b> P2P Training without Serverless Computing:</b> This represents the traditional P2P training process, where peers communicate and synchronize with each other directly, without the integration of serverless computing.</li>
  <li> <b>P2P Training with Serverless Computing: </b>This architecture integrates serverless computing into the P2P training process, leveraging its dynamic resource allocation capabilities to enhance training efficiency and address the challenges posed by varying peer capabilities.</li>
</ul> 
<p>Our work leverages the dynamic resource allocation capability of serverless computing, which allows it to adapt to real-time requirements and effectively address the challenges presented by the increasing number of peers with varying capabilities.</p>
</p>





<h2>P2P Training without Serverless Computing</h2>
The following figure show the architecture to make running a p2p training without serverless:

![P2P](P2P.jpg)


<p> In the following, we will show all the steps to make a replication of our study</p>

1. Prepare the EC2 instances according to the needed number of peers.
2. Copy the script of P2P distributed training to the different EC2 instances.
3. Configure RabbitMQ using amazon or configure a local one with a public IP address.
4. Prepare the dataset inside the S3 buckets.
5. Start the different peers inside each EC2.

<h3>1. Prepare the EC2 instances</h3> 
To set up the EC2 instances for distributed computing, follow these steps:

1. Launch the required number of EC2 instances based on the desired number of peers. Choose an instance type that meets the computational requirements of your workload.For instance, the t2.medium instance type is suitable for MobileNetV3 small, while the t2.large instance type is recommended for VGG11.

2. Configure the security group for the EC2 instances to allow necessary inbound and outbound network traffic. This involves opening specific ports and enabling communication between instances within the same security group.

3. Install the necessary dependencies and libraries on each EC2 instance. Ensure that all required software, frameworks, and libraries are installed to support your distributed computing workload.
4. Set up SSH access to the EC2 instances for remote management. 
5. Generate SSH key pairs, associate them with the instances, and securely store the private key for authentication.
6. Test the connectivity between the EC2 instances to verify effective communication. Ensure that the required ports are open and that the instances can discover and connect to each other. 

<h3>2. Copy the script of P2P to EC2</h3> 
To distribute scripts to each machine, you can utilize the following command:
<pre> scp -i myAmazonKey.pem script.py ubuntu@ip-xx-xxx-xx-xxx.compute-1.amazonaws.com  </pre>  

Replace <strong> myAmazonKey.pem </strong> with the path to your Amazon EC2 key file and <strong> script.py </strong> with the name of the script you want to send. Additionally, replace <strong> ip-xx-xxx-xx-xxx.compute-1.amazonaws.com </strong> with the actual public IP or DNS address of the target EC2 instance. 


<h3>3. Configure RabbitMQ</h3> 
You can configure and set up the environment locally on your machine by installing the RabbitMQ on your local machine and provide a public IP adress to have a public credentials that will be used by the Peer to send and receive data.

To install RabbitMQ on Ubuntu, follow these steps:

1.  Open the terminal on your Ubuntu machine.<br>  

2.  Update the package list by running the following command: 
<pre>sudo apt update</pre> 

3. Install RabbitMQ by running the following command: 
<pre>sudo apt install rabbitmq-server</pre> 
4. Once the installation is complete, start the RabbitMQ server by running the following command: <pre>sudo systemctl start rabbitmq-server</pre> 
5. To enable the RabbitMQ server to start automatically at boot time, run the following command: <pre>sudo systemctl enable rabbitmq-server</pre> 
6. Check the status of the RabbitMQ server by running the following command: 
<pre class="commande">sudo systemctl status rabbitmq-server</pre>  

Make sure to have the following credentials to replace them inside the source code:

<pre>  rabbitmq_host: xxx.xxx.xxx.xxx
  rabbitmq_port: xxxxx
  rabbitmq_username: xxxxxxxx
  rabbitmq_password: xxxxxxxx</pre>

Another option is to set up the environment on Amazon Web Services (AWS) by creating a RabbitMQ instance. This process entails both creating the RabbitMQ instance itself and configuring it in a manner that aligns with your distinct requirements. By specifying the desired instance type, storage options, and additional settings, you can effectively fine-tune the environment to optimize its functionality and suit your specific needs. 
<h3>4. Prepare the data inside the S3</h3> 
1. Create and configure an S3 (Simple Storage Service) bucket. This involves setting up the necessary permissions, access control, and region settings for the bucket.

2. Create the required buckets to accommodate different data partitions based on your dataset and the model being used. Each bucket should be named and organized in a way that aligns with your data partitioning strategy.

3. Partition and load the data into the respective S3 buckets. To achieve this, execute the split_worker_send_to_s3 script using the following command: 

<pre>python3 split_worker_send_to_s3.py [--size SIZE] [--dataset DATASET] [--model_str MODEL]<br>  
<strong>Arguments:</strong> 
<strong>--size</strong> SIZE                  Total number of workers in the deployment 
<strong>--dataset</strong> DATASET            Dataset to be used, e.g., mnist, cifar10.
<strong>--model_str</strong> MODEL            Model to be trained, e.g., squeeznet1.1, vgg11, mobilenet v3 small. 

</pre> 




<h3>5. Start the process</h3>  
Deployment requires running `EC2_without_serverless.py` on multiple machines. 
<pre>
<strong>Usage:</strong>

EC2_without_serverless.py [--size SIZE] [--rank RANK] [--batch BATCH]
[--dataset DATASET] [--model_str MODEL] [--optimizer OPTIMIZER]
[--loss LOSS] [--evaluation EVALUATION] [--compression COMPRESSION] 


<strong>Arguments:</strong> 
  
<strong>--size</strong> SIZE                  Total number of workers in the deployment 
<strong>--rank</strong> RANK                  Unique ID of the worker node in the distributed setup.
<strong>--batch_size</strong> BATCH           Size of the batch to be employed by each node.
<strong>--dataset</strong> DATASET            Dataset to be used, e.g., mnist, cifar10.
<strong>--model_str</strong> MODEL            Model to be trained, e.g., squeeznet1.1, vgg11, mobilenet v3 small.
<strong>--loss </strong> LOSS                 Loss function to optimize.
<strong>--optimizer</strong> OPTIMIZER        Optimizer to use.
<strong>--evaluation </strong> EVALUATION     If True, metrics are recorded.
<strong>--compression </strong> COMPRESSION   If True, gradients are compressed to minimize communication overhead.
</pre> 



<h2>P2P Training with Serverless Computing</h2>
The following figure show the architecture to make running a p2p training without serverless:

<img src="markdownmonstericon.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<p> In the following, we will show all the steps to make a replication of P2P using serverless</p>

1. Prepare the EC2 instances according to the needed number of peers.
2. Copy the script of P2P distributed training to the different EC2 instances.
3. Configure RabbitMQ using amazon or configure a local one with a public IP address.
4. Prepare the dataset inside the S3 buckets.
5. Create Lambda serverless to make a batch training
6. Create AWS Step function for managing the flow
5. Start the different peers inside each EC2.

<h3>1. Prepare the EC2 instances</h3>
--Similar to the previosu Section--
<h3>2. Copy the script of P2P to EC2</h3>
--Similar to the previosu Section--
<h3>3. Configure RabbitMQ</h3>
--Similar to the previosu Section--
<h3>4. Prepare the dataset inside the S3</h3>
--Similar to the previosu Section--

Execute the split_worker_batches_send_to_s3.py script using the following command: 

<pre>python3 split_worker_batches_send_to_s3.py [--size SIZE] [--dataset DATASET] [--model_str MODEL]<br>  
<strong>Arguments:</strong> 
<strong>--size</strong> SIZE                  Total number of workers in the deployment 
<strong>--dataset</strong> DATASET            Dataset to be used, e.g., mnist, cifar10.
<strong>--model_str</strong> MODEL            Model to be trained, e.g., squeeznet1.1, vgg11, mobilenet v3 small. 

</pre> 

This script will split the data into workers and batches, each batch will be processed by a lambda function.


<h3>5. Create Lambda serverless</h3>

1. Step1: Navigate to the AWS Lambda Console
https://console.aws.amazon.com/lambda/

2. Step2: Configure the function:

- Give a name to the function called "compute_gradient"
- Select the Python3 as language, and make the necessary modification to the accorded ressources and the timelimit that will be set for lambda function (default: 3 seconds, maximum 15 minutes).

3. Use the source code we prepared in the folder package, we put all the packages that the lambda function will need to train a model for the assigned data batch.

You can modify the lambda_function.py code if you need specific requirements.

Create a zip file with all the package including the lambda_function.py and the different requirements in package folder. name the zip file as "train_batch.zip"

<pre> zip -r ../train_batch.zip . </pre>

Use the following command to deploy the function:

<pre>
aws lambda update-function-code --function-name compute_gradient --zip-file fileb://train_batch.zip
</pre>

4. Since we are going to create a parallel processing, we need to use another lambda function that will trigger the parallel batch processing functions.

The code of this function is located on the file "ProcessInputFunction.py". Deploy this function as lambda function called "ProcessInputFunction"

<h3>6. Create AWS Step function</h3>
1. Navigate to the AWS Step Functions Console: https://console.aws.amazon.com/states/.

2. Create a new state machine that you call it "batch_processing"


3. Use the code "generate_state_machine.py" to generate a machine step function that you will deploy it into the AWS Step Function. You need to give it the number of batches as input.

<pre>python3 generate_state_machine.py [--nbr_batches] </pre>

- A similar file to the "state_machine_definition.json" will be generated. You need to deploy this file to the step function configured before.  

<pre> aws stepfunctions update-state-machine --state-machine-arn arn:aws:states:us-east-1:account_id:stateMachine:batch_processing --definition file://state_machine_definition.json </pre>



<h3>7. Start the process</h3>
In each EC2 machine, you need to run the source code as following:

<pre>
<strong>Usage:</strong>

EC2_with_serverless.py [--size SIZE] [--rank RANK] [--batch BATCH]
[--dataset DATASET] [--model_str MODEL] [--optimizer OPTIMIZER]
[--loss LOSS] [--evaluation EVALUATION] [--compression COMPRESSION] 


<strong>Arguments:</strong> 
  
<strong>--size</strong> SIZE                  Total number of workers in the deployment 
<strong>--rank</strong> RANK                  Unique ID of the worker node in the distributed setup.
<strong>--batch_size</strong> BATCH           Size of the batch to be employed by each node.
<strong>--dataset</strong> DATASET            Dataset to be used, e.g., mnist, cifar10.
<strong>--model_str</strong> MODEL            Model to be trained, e.g., squeeznet1.1, vgg11, mobilenet v3 small.
<strong>--loss </strong> LOSS                 Loss function to optimize.
<strong>--optimizer</strong> OPTIMIZER        Optimizer to use.
<strong>--evaluation </strong> EVALUATION     If True, metrics are recorded.
<strong>--compression </strong> COMPRESSION   If True, gradients are compressed to minimize communication overhead.
</pre> 



Please note that we removed all the credentials from the source code, where you need to replace it with your proper credentials.

<h3>References</h3>





