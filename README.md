
# DA6401 - Assignment 1

## Author: PRATEEK PANDEY

### Link to the project report:

https://wandb.ai/cs24m034-indian-institute-of-technology-madras/DA6401_Assignment-1/reports/Prateek-s-DA6401-Assignment-1--VmlldzoxMTcxNDcyOA?accessToken=9gzb81ebi37bru6s8jwqe4th7ae4u7djoog4w7284y82wqzzl1jm4a1frczoo2y4

### Steps to follow:

1. Install wandb by command *pip install wandb* to your system
2. Run train.py file with the appropriate arguments given below
3. Your run will be logged to wandb to my project "DA6401_Assignment-1" and entity="cs24m034"
4. You can view the logs that is the model training accuracy, validation accuracy, testing accuracy and respective losses

### Explanation of the project:

**1. NN.ipynb** is the jupyter notebook which contains all the necessary functions for training the Neural Network model used in questions 1-10. It contains the optimiser functions sgd, momentum based GD, nesterov accelerated GD, RMSProp, Adam, NAdam as asked in question 3. This code tests also tests on the MNIST data.

**2. NN.py** is the python script for the same above jupyter notebook I created this as my train.py script calls the required functions from this python file.

**3. train.py** a python script which you need to call that accepts the following command line arguments with the specified values -

We will check your code for implementation and ease of use. We will also verify your code works by running the following command and checking wandb logs generated -
| Name | Default Value | Description |
|------|---------------|-------------|
| -wp, --wandb_project | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity | myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| -d, --dataset | fashion_mnist | choices: ["mnist", "fashion_mnist"] |
| -e, --epochs | 1 | Number of epochs to train neural network. |
| -b, --batch_size | 4 | Batch size used to train neural network. |
| -l, --loss | cross_entropy | choices: ["mean_squared_error", "cross_entropy"] |
| -o, --optimizer | sgd | choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] |
| -lr, --learning_rate | 0.1 | Learning rate used to optimize model parameters |
| -m, --momentum | 0.5 | Momentum used by momentum and nag optimizers. |
| -beta, --beta | 0.5 | Beta used by rmsprop optimizer |
| -beta1, --beta1 | 0.5 | Beta1 used by adam and nadam optimizers. |
| -beta2, --beta2 | 0.5 | Beta2 used by adam and nadam optimizers. |
| -eps, --epsilon | 0.000001 | Epsilon used by optimizers. |
| -w_d, --weight_decay | .0 | Weight decay used by optimizers. |
| -w_i, --weight_init | random | choices: ["random", "Xavier"] |
| -nhl, --num_layers | 1 | Number of hidden layers used in feedforward neural network. |
| -sz, --hidden_size | 4 | Number of hidden neurons in a feedforward layer. |
| -a, --activation | sigmoid | choices: ["identity", "sigmoid", "tanh", "ReLU"] |

The NN training framework:

My code are based on a procedural framework and make no use of classes for NN models like keras does for the simplicity of understanding as well as the code. 
My code works only for classification tasks and by default assumes that the activation function for the last layer is softmax. 
This was done for simplicity as because the tasks involved in the assignment did not call for a different output layer activation. 

**1. learning_params(...)**

The learning_params(...) function takes the training data, the validation data and the hyperparameters and Trains a NN specified by num_neurons and num_hidden. 


* **eta**: the learning rate 


* **activation**: activation functions for all the layers except the last layer which is softmax  $\epsilon$ (sigmoid, ReLU, tanh)                          


* **init_mode**: initialization mode $\epsilon$ (random_uniform, random_normal, xavier)


* **learning_algorithm**: optimization routine $\epsilon$ (sgd, momentum, nesterov, RMSprop, Adam, nadam)


* **bach_size**: minibatch size


* **loss**: loss function $\epsilon$ (MSE, Categorical Crossentropy)


* **epochs**: number of epochs to be used


* **L2_lamb**: lambda for L2 regularisation of weights


* **num_neurons**: number of neurons in every hidden layer


* **num_hidden**: number of hidden layers


The function returns 

* weights and biases: two list containing weights and biases

* epoch costs: a list containing Cost function values vs epochs
* epoch_train_loss: a list containing Cost function values vs epochs
* epoch_train_loss: a list containing train loss values vs epochs
* epoch_val_loss: a list containing val loss values vs epochs
* acc_train: a list containing training accuracy vs epochs
* acc_val: a list containing validation data accuracy values vs epochs

The function contains 2 loops, one epoch loop and one batch loop. Note the optimizers are not implemented with those loops instead they are called as parameter updates hence the nomenclature update_params_sgd etc. Hence to include a new optimiser routine, we simply need to include a function that can update the parameters in each epoch in each batch.

