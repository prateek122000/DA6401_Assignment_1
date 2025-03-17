

from sklearn.model_selection import train_test_split
import wandb
from keras.datasets import fashion_mnist,mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_images():
  wandb.login()
  # start a new wandb run to track this script
  wandb.init(project="DA6401_Assignment-1")
  # simulate training
  # x_train is a matrix of shape 60000x28x28 containing image pixels for training purposes.
  # y_train is a matrix of shape (60000, 1), containing the corresponding labels.

  # x_train is a matrix of shape 10000x28x28 containing image pixels for training purposes.
  # y_train is a matrix of shape (10000, 1), containing the corresponding labels.
  (x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()
  labels=set()
  i=0
  fig,ax=plt.subplots(2,5,figsize=(10,5))
  row=0
  col=0
  for pixels in x_train:
    #The matplotlib function imshow() creates an image from a 2-dimensional numpy array
    #pixels is (28,28) 2-D array
    #l is the current label of image
    l=y_train[i]
    if(not(l in labels)):
      if(col>=5):
        col=0
        row+=1
      ax[row][col].imshow(pixels,cmap="gray")
      ax[row][col].set_title("Label {}".format(l))
      ax[row][col].axis(False)
      labels.add(l)
      col+=1
    #if we get all our 10 labels just break the loop
    if(len(labels)==10):
      break;
    i+=1
  wandb.log({"plot":plt})
  wandb.run.name = "Sample_Images"
  wandb.run.save()
  wandb.run.finish()
  # finish the wandb run, necessary in notebooks
  wandb.finish()

def data_preprocess(dataset="fashion_mnist"):
    if dataset=="fashion_mnist":
        (x_train, y_train), (x_test, y_test) =fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) =mnist.load_data()
    #NORMALIZING THE DATASET
    x_train=x_train/255.0
    x_test=x_test/255.0
    #RESHAPING THE TRAIN_IMAGE DATASET FROM (60000,28x28) TO (60000,784) AND SAME FOR TEST_IMAGE
    num_inputs=784
    num_outputs=10
    x_train=x_train.reshape(x_train.shape[0],784)
    x_test=x_test.reshape(x_test.shape[0],784)


    #SPLITTING THE TRAINING DATA FOR VALIDATION AND TESTING
    train_x,val_x,train_y,val_y=train_test_split(x_train,y_train)
    train_x=np.transpose(train_x)
    train_y=np.transpose(train_y)
    val_x=np.transpose(val_x)
    val_y=np.transpose(val_y)
    #RESHAPING MY DATA TO COLUMN-WISE IMAGES
    x_train=x_train.T
    x_test=x_test.T
    return x_train,y_train,train_x,train_y,val_x,val_y,x_test, y_test

#ONE-HOT ENCODING FOR Y_TRAIN AND Y_TEST:
def one_hot_encoding(y):
    exp_y=np.zeros((10,y.shape[0]))
    for i in range(0,y.shape[0]):
        exp_y[y[i]][i]=1
    return exp_y

def softmax(x):
    max_val = np.max(x, axis=0, keepdims=True)
    exp_val= np.exp(x-max_val)
    sum_exp = np.sum(exp_val, axis=0, keepdims=True)
    f_x = exp_val/sum_exp
    return f_x

def sigmoid(x):
    x = np.clip(x, -500, 500)
    a = np.exp(-x)
    f_x = 1.0/(1.0+a)
    return f_x

def sigmoid_derivative(x):
    x = np.clip(x, -500, 500)
    s = 1 / (1 + np.exp(-x))
    f_x = s * (1 - s)
    return f_x

def Relu(x):
    return np.maximum(0,x)

def identity(x):
    return x

def Relu_derivative(x):
    return 1*(x>0)

def tanh(x):
    x = np.clip(x, -500, 500)
    f_x = np.tanh(x)
    return f_x

def tanh_derivative(x):
    x = np.clip(x, -500, 500)
    t = np.tanh(x)
    f_x = 1 - t**2
    return f_x

def initialize_params(hidden_layers,neurons,method):
  #USING XAVIER INITIALIZATION TO INITIALIZE WEIGHTS AND BIAS MATRIX

  #INDEXING DONE FROM 1
  L=hidden_layers+1 #number of layers excluding hidden layer
  weights=[0]*(hidden_layers+2)
  biases=[0]*(hidden_layers+2)
  previous_updates_W=[0]*(hidden_layers+2)
  previous_updates_B=[0]*(hidden_layers+2)
  np.random.seed(42)
  for i in range(1,hidden_layers+1):
    n=neurons[i]
    # appending the weight and bias matrix for the ith layer
    if(i==1):
      if method=='xavier':
        weights[i]=(np.random.randn(n,784)*np.sqrt(2/(n+784)))
      if method=='random':
        weights[i]=(np.random.randn(n,784))*0.01
      biases[i]=(np.zeros((n,1)))
      previous_updates_W[i]=np.zeros((n,784))
      previous_updates_B[i]=np.zeros((n,1))
      # biases[i]=(np.random.randn(n,1))
    else:
      if method=='xavier':
        weights[i]=(np.random.randn(n,neurons[i-1])*np.sqrt(2/(n+neurons[i-1])))
      if method=='random':
        weights[i]=(np.random.randn(n,neurons[i-1]))*0.01
      biases[i]=(np.zeros((n,1)))
      previous_updates_W[i]=np.zeros((n,neurons[i-1]))
      previous_updates_B[i]=np.zeros((n,1))
      # biases[i]=(np.random.randn(n,1))
  weights[L]=(np.random.randn(10,neurons[hidden_layers])*np.sqrt(2/(10+neurons[hidden_layers-1])))
  biases[L]=(np.zeros((10,1)))
  previous_updates_W[L]=np.zeros((10,neurons[hidden_layers]))
  previous_updates_B[L]=np.zeros((10,1))
  weights=np.array(weights,dtype=object)
  biases=np.array(biases,dtype=object)
  previous_updates_W=np.array(previous_updates_W,dtype=object)
  previous_updates_B=np.array(previous_updates_B,dtype=object)
  return weights,biases,previous_updates_W,previous_updates_B

def apply_activation(x, activation):
    activations = {
        'sigmoid': sigmoid,
        'ReLU': Relu,
        'tanh': tanh,
        'identity': identity
    }
    return activations.get(activation, lambda x: x)(x)

def FeedForwardNetwork(weights, biases, L, data, activation):
    h = [None] * (L + 1)
    a = [None] * (L + 1)
    h[0] = data

    for i in range(1, L):
        a[i] = np.add(np.dot(weights[i], h[i - 1]), biases[i])
        h[i] = apply_activation(a[i], activation)

    a[L] = np.add(np.dot(weights[L], h[L - 1]), biases[L])
    h[L] = softmax(a[L])

    return h[L], h, a

def compute_loss_gradient(y_hat, exp_Y, A_L, loss):
    if loss == "cross_entropy":
        return -(exp_Y - y_hat)
    elif loss == "mean_squared_error":
        return -(exp_Y - y_hat) * softmax(A_L) * (1 - softmax(A_L))
    return None

def apply_derivative(activation, grad_H, A):
    derivatives = {
        'sigmoid': sigmoid_derivative,
        'ReLU': Relu_derivative,
        'tanh': tanh_derivative,
        'identity': lambda x: np.ones_like(x)
    }
    return np.multiply(grad_H, derivatives.get(activation, lambda x: x)(A))

def BackPropogation(weights, L, H, A, exp_Y, y_hat, activation, loss="cross_entropy"):
    gradients_A = [None] * (L + 1)
    gradients_W = [None] * (L + 1)
    gradients_B = [None] * (L + 1)

    gradients_A[L] = compute_loss_gradient(y_hat, exp_Y, A[L], loss)

    for k in range(L, 0, -1):
        gradients_W[k] = np.dot(gradients_A[k], H[k - 1].T)
        gradients_B[k] = np.sum(gradients_A[k], axis=1, keepdims=True)

        if k > 1:
            grad_H = np.dot(weights[k].T, gradients_A[k])
            gradients_A[k - 1] = apply_derivative(activation, grad_H, A[k - 1])

    return gradients_W, gradients_B

def calc_loss(weights, y, exp_y, loss, data_size, L2_lamb):
    """
    Computes the loss value based on the specified loss function and applies L2 regularization.

    Parameters:
    - weights: List of weight matrices
    - y: Model predictions
    - exp_y: One-hot encoded actual class labels
    - loss: Loss function type ('cross_entropy' or 'mean_squared_error')
    - data_size: Number of data samples
    - L2_lamb: L2 regularization strength

    Returns:
    - Computed loss value
    """
    if loss == 'cross_entropy':
        loss_val = -np.sum(exp_y * np.log(y)) / data_size
    elif loss == 'mean_squared_error':
        loss_val = 0.5 * np.sum((y - exp_y) ** 2) / data_size
    else:
        raise ValueError("Invalid loss function. Choose 'cross_entropy' or 'mean_squared_error'.")

    # Compute L2 regularization term (excluding bias weights at index 0)
    l2_penalty = sum(np.sum(np.square(w)) for w in weights[1:])
    loss_val += (L2_lamb / (2 * data_size)) * l2_penalty

    return loss_val

def calc_accuracy(y,predicted_y):
    correct=0
    for i in range(len(y)):
        if(y[i]==predicted_y[i]):
            correct+=1
    return (correct/len(y))*100

def sgd_params_update(weights, biases, gradients_W, gradients_B, eta, L, L2_lamb):
    gradients_B = np.asarray(gradients_B, dtype=object)
    gradients_W = np.asarray(gradients_W, dtype=object)

    for i in range(1, L + 1):
        weight_update = eta * (gradients_W[i] + L2_lamb * weights[i])
        bias_update = eta * gradients_B[i]

        weights[i] -= weight_update
        biases[i] -= bias_update

    return weights, biases



def update_parameters_momentum(weights, biases, gradients_B, gradients_W, beta, prev_W_update, prev_B_update, eta, L, L2_lamb):
    gradients_B = np.asarray(gradients_B, dtype=object)
    gradients_W = np.asarray(gradients_W, dtype=object)

    for i in range(1, L + 1):
        prev_W_update[i] = beta * prev_W_update[i] + (1 - beta) * gradients_W[i]
        prev_B_update[i] = beta * prev_B_update[i] + (1 - beta) * gradients_B[i]

        weights[i] -= (eta * prev_W_update[i] + eta * L2_lamb * weights[i])
        biases[i] -= eta * prev_B_update[i]
    return weights,biases,prev_W_update,prev_B_update



def update_parameters_adam(weights, biases, gradients_B, gradients_W, eta, m_W, m_B, v_W, v_B, t, L, L2_lamb, beta1, beta2, epsilon):
    gradients_B = np.asarray(gradients_B, dtype=object)
    gradients_W = np.asarray(gradients_W, dtype=object)

    beta1_t = 1 - beta1**t
    beta2_t = 1 - beta2**t

    for i in range(1, L + 1):
        # Compute biased first moment estimates
        m_W[i] = beta1 * m_W[i] + (1 - beta1) * gradients_W[i]
        m_B[i] = beta1 * m_B[i] + (1 - beta1) * gradients_B[i]

        # Compute biased second raw moment estimates
        v_W[i] = beta2 * v_W[i] + (1 - beta2) * gradients_W[i]**2
        v_B[i] = beta2 * v_B[i] + (1 - beta2) * gradients_B[i]**2

        # Compute bias-corrected first moment estimate
        m_W_hat = m_W[i] / beta1_t
        m_B_hat = m_B[i] / beta1_t

        # Compute bias-corrected second raw moment estimate
        v_W_hat = v_W[i] / beta2_t
        v_B_hat = v_B[i] / beta2_t

        # Update parameters
        weights[i] -= eta * (m_W_hat / (np.sqrt(v_W_hat) + epsilon) + L2_lamb * weights[i])
        biases[i] -= eta * m_B_hat / (np.sqrt(v_B_hat) + epsilon)

    return weights, biases, m_W, m_B, v_W, v_B, t + 1



def rmsprop_params_update(weights, biases, gradients_B, gradients_W, beta, eta, W_v, B_v, L, L2_lamb):
    gradients_B = np.asarray(gradients_B, dtype=object)
    gradients_W = np.asarray(gradients_W, dtype=object)
    epsilon = 1e-4

    for i in range(1, L + 1):
        # Update moving averages of squared gradients
        W_v[i] = beta * W_v[i] + gradients_W[i] ** 2
        B_v[i] = beta * B_v[i] + gradients_B[i] ** 2

        # Compute adaptive learning rate
        adjusted_eta_W = eta / np.sqrt(W_v[i] + epsilon)
        adjusted_eta_B = eta / np.sqrt(B_v[i] + epsilon)

        # Update weights and biases
        weights[i] -= adjusted_eta_W * gradients_W[i] + eta * L2_lamb * weights[i]
        biases[i] -= adjusted_eta_B * gradients_B[i]

    return weights, biases, W_v, B_v

'''def learning_params(hidden_layers,neuron,x_train,y_train,x_val,y_val,learning_algorithm,eta,epochs,batch_size,activation,init_method,L2_lamb,momentum=0.9 ,beta=0.9 ,beta1=0.9 ,beta2=0.99 ,epsilon=0.00001,loss="cross_entropy"):
  count=1
  predicted_y=[]
  L=hidden_layers+1
  neurons=[0]*(L)
  for i in range(1,L):
    neurons[i]=neuron
  exp_y=one_hot_encoding(y_train)
  exp_y_val=one_hot_encoding(y_val)
  i
  weights,biases,previous_updates_W,previous_updates_B=initialize_params(hidden_layers,neurons,init_method)
  epoch_train_loss=[]
  epoch_val_loss=[]
  acc_val=[]
  acc_train=[]
  t=1
  v_W = previous_updates_W.copy()
  m_W = previous_updates_W.copy()
  v_B = previous_updates_B.copy()
  m_B = previous_updates_B.copy()
  while count<=epochs:
      for i in range(0,x_train.shape[1],batch_size):
        mini_batch=x_train[:,i:i+batch_size]
        if learning_algorithm=='nag':
          W_look_ahead=weights-(beta)*previous_updates_W
          B_look_ahead=biases-(beta)*previous_updates_B
          output,post_act,pre_act=FeedForwardNetwork(W_look_ahead,B_look_ahead,L,mini_batch,activation)
          gradients_W,gradients_B=BackPropogation(W_look_ahead,L,post_act,pre_act,exp_y[:,i:i+batch_size],output,activation,loss)
          weights,biases,previous_updates_W,previous_updates_B=update_parameters_momentum(weights,biases, gradients_B,gradients_W, beta, previous_updates_W,previous_updates_B,eta,L,L2_lamb)
        elif learning_algorithm=='nadam':
          W_look_ahead=weights-(beta)*previous_updates_W
          B_look_ahead=biases-(beta)*previous_updates_B
          output,post_act,pre_act=FeedForwardNetwork(W_look_ahead,B_look_ahead,L,mini_batch,activation)
          gradients_W,gradients_B=BackPropogation(W_look_ahead,L,post_act,pre_act,exp_y[:,i:i+batch_size],output,activation,loss)
          weights,biases,m_W,m_B,v_W,v_B,t= update_parameters_adam(weights, biases, gradients_B,gradients_W,eta, m_W,m_B,v_W,v_B, t,L,L2_lamb,beta1,beta2,epsilon)
        elif learning_algorithm=='momentum':
            output,post_act,pre_act=FeedForwardNetwork(weights,biases,L,mini_batch,activation)
            gradients_W,gradients_B=BackPropogation(weights,L,post_act,pre_act,exp_y[:,i:i+batch_size],output,activation,loss)
            weights,biases,previous_updates_W,previous_updates_B=update_parameters_momentum(weights, biases, gradients_B,gradients_W, momentum, previous_updates_W,previous_updates_B,eta,L,L2_lamb)
        elif learning_algorithm=='sgd':
            output,post_act,pre_act=FeedForwardNetwork(weights,biases,L,mini_batch,activation)
            gradients_W,gradients_B=BackPropogation(weights,L,post_act,pre_act,exp_y[:,i:i+batch_size],output,activation,loss)
            weights,biases=sgd_params_update(weights,biases,gradients_W,gradients_B,eta,L,L2_lamb)
        elif learning_algorithm=='adam':
            output,post_act,pre_act=FeedForwardNetwork(weights,biases,L,mini_batch,activation)
            gradients_W,gradients_B=BackPropogation(weights,L,post_act,pre_act,exp_y[:,i:i+batch_size],output,activation,loss)
            weights,biases,m_W,m_B,v_W,v_B,t= update_parameters_adam(weights, biases, gradients_B,gradients_W,eta, m_W,m_B,v_W,v_B, t,L,L2_lamb,beta1,beta2,epsilon)
        elif learning_algorithm=='rmsprop':
            output,post_act,pre_act=FeedForwardNetwork(weights,biases,L,mini_batch,activation)
            gradients_W,gradients_B=BackPropogation(weights,L,post_act,pre_act,exp_y[:,i:i+batch_size],output,activation,loss)
            weights,biases,previous_updates_W,previous_updates_B = rmsprop_params_update(weights, biases, gradients_B,gradients_W, beta,eta, previous_updates_W,previous_updates_B,L,L2_lamb)
        else:
            break;
      full_output_train,_,_=FeedForwardNetwork(weights,biases,L,x_train,activation)
      full_output_val,_,_=FeedForwardNetwork(weights,biases,L,x_val,activation)
      loss_train=calc_loss(weights,full_output_train,exp_y,loss,full_output_train.shape[1],L2_lamb)
      loss_val=calc_loss(weights,full_output_val,exp_y_val,loss,full_output_val.shape[1],L2_lamb)
      acc_train.append(calc_accuracy(y_train,np.argmax(full_output_train,axis=0)))
      acc_val.append(calc_accuracy(y_val,np.argmax(full_output_val,axis=0)))
      epoch_train_loss.append(loss_train)
      epoch_val_loss.append(loss_val)
      count+=1
  return weights,biases,epoch_train_loss,epoch_val_loss,acc_train,acc_val'''


def learning_params(
    hidden_layers, neuron, x_train, y_train, x_val, y_val, learning_algorithm,
    eta, epochs, batch_size, activation, init_method, L2_lamb,
    momentum=0.9, beta=0.9, beta1=0.9, beta2=0.99, epsilon=1e-5, loss="cross_entropy"
):
    L = hidden_layers + 1
    neurons = [neuron] * L
    exp_y, exp_y_val = one_hot_encoding(y_train), one_hot_encoding(y_val)

    weights, biases, prev_updates_W, prev_updates_B = initialize_params(hidden_layers, neurons, init_method)

    epoch_train_loss, epoch_val_loss, acc_train, acc_val = [], [], [], []
    t = 1

    v_W, m_W = prev_updates_W.copy(), prev_updates_W.copy()
    v_B, m_B = prev_updates_B.copy(), prev_updates_B.copy()

    for epoch in range(epochs):
        for i in range(0, x_train.shape[1], batch_size):
            mini_batch = x_train[:, i:i + batch_size]
            batch_y = exp_y[:, i:i + batch_size]

            if learning_algorithm in {"nag", "nadam"}:
                W_look_ahead, B_look_ahead = weights - beta * prev_updates_W, biases - beta * prev_updates_B
            else:
                W_look_ahead, B_look_ahead = weights, biases

            output, post_act, pre_act = FeedForwardNetwork(W_look_ahead, B_look_ahead, L, mini_batch, activation)
            gradients_W, gradients_B = BackPropogation(W_look_ahead, L, post_act, pre_act, batch_y, output, activation, loss)

            if learning_algorithm == "nag":
                weights, biases, prev_updates_W, prev_updates_B = update_parameters_momentum(
                    weights, biases, gradients_B, gradients_W, beta, prev_updates_W, prev_updates_B, eta, L, L2_lamb
                )
            elif learning_algorithm == "nadam" or learning_algorithm == "adam":
                weights, biases, m_W, m_B, v_W, v_B, t = update_parameters_adam(
                    weights, biases, gradients_B, gradients_W, eta, m_W, m_B, v_W, v_B, t, L, L2_lamb, beta1, beta2, epsilon
                )
            elif learning_algorithm == "momentum":
                weights, biases, prev_updates_W, prev_updates_B = update_parameters_momentum(
                    weights, biases, gradients_B, gradients_W, momentum, prev_updates_W, prev_updates_B, eta, L, L2_lamb
                )
            elif learning_algorithm == "sgd":
                weights, biases = sgd_params_update(weights, biases, gradients_W, gradients_B, eta, L, L2_lamb)
            elif learning_algorithm == "rmsprop":
                weights, biases, prev_updates_W, prev_updates_B = rmsprop_params_update(
                    weights, biases, gradients_B, gradients_W, beta, eta, prev_updates_W, prev_updates_B, L, L2_lamb
                )
            else:
                raise ValueError("Invalid learning algorithm specified")

        full_output_train, _, _ = FeedForwardNetwork(weights, biases, L, x_train, activation)
        full_output_val, _, _ = FeedForwardNetwork(weights, biases, L, x_val, activation)

        loss_train = calc_loss(weights, full_output_train, exp_y, loss, full_output_train.shape[1], L2_lamb)
        loss_val = calc_loss(weights, full_output_val, exp_y_val, loss, full_output_val.shape[1], L2_lamb)

        epoch_train_loss.append(loss_train)
        epoch_val_loss.append(loss_val)
        acc_train.append(calc_accuracy(y_train, np.argmax(full_output_train, axis=0)))
        acc_val.append(calc_accuracy(y_val, np.argmax(full_output_val, axis=0)))

    return weights, biases, epoch_train_loss, epoch_val_loss, acc_train, acc_val

def run_sweeps(train_x,train_y,val_x,val_y):

    config = {
        "project":"DA6401_Assignment-1",
        "method": 'random',
        "metric": {
        'name': 'acc',
        'goal': 'maximize'
        },
        'parameters' :{
        "hidden_layers": {"values":[3,4,5]},
        "neurons": {"values": [32,64,128]},
        "learning_algorithm": {"values":["momentum","sgd","nag","rmsprop","nadam","adam"]},
        "eta": {"values":[1e-3,1e-4]},
        "epoch": {"values":[5,10]},
        "batch_size": {"values":[16,32,64]},
        "activation": {"values":["tanh","ReLU","sigmoid"]},
        "weight_init":{"values":["random","xavier"]},
        "L2_lamb":{"values":[0,0.0005,0.5]}
        }
    }

    def trainn():
        wandb.init()
        name='_h1_'+str(wandb.config.hidden_layers)+"_SL_"+str(wandb.config.neurons)+"_BS_"+str(wandb.config.batch_size)+"_OPT_"+str(wandb.config.learning_algorithm)
        _,_,epoch_train_loss,epoch_val_loss,acc_train,acc_val=learning_params(wandb.config.hidden_layers,wandb.config.neurons,train_x,train_y,val_x,val_y,wandb.config.learning_algorithm,wandb.config.eta,wandb.config.epoch,wandb.config.batch_size,wandb.config.activation,wandb.config.weight_init,wandb.config.L2_lamb)
        for i in range(len(epoch_train_loss)):
            wandb.log({"loss":epoch_train_loss[i]})
            wandb.log({"val_loss":epoch_val_loss[i]})
            wandb.log({"accuracy":acc_train[i]})
            wandb.log({"val_acc":acc_val[i]})
            wandb.log({"epoch": (i+1)})
        wandb.log({"acc":acc_val[-1]})
        wandb.run.name = name
        wandb.run.save()
        wandb.run.finish()
    sweep_id=wandb.sweep(config,project="DA6401_Assignment-1")
    wandb.agent(sweep_id,function=trainn,count=100)

'''def log_confusion_mat():
    wandb.init(project="DA6401_Assignment-1")
    _,_,train_x,train_y,val_x,val_y,x_test,y_test=data_preprocess()
    hidden_layers=6
    weights,biases,epoch_train_loss,epoch_val_loss,acc_train,acc_val=learning_params(hidden_layers=6,neuron=64,x_train=train_x,y_train=train_y,x_val=val_x,y_val=val_y,learning_algorithm="nadam",eta=0.001,epochs=10,batch_size=32,activation="ReLU",init_method="xavier",L2_lamb=0.0005,momentum=0.9 ,beta=0.9 ,beta1=0.9 ,beta2=0.99 ,epsilon=0.00001)
    L=hidden_layers+1
    full_output_test,_,_=FeedForwardNetwork(weights,biases,L,x_test,"ReLU")
    predicted_y=np.argmax(full_output_test,axis=0)
    predicted_y=np.array(predicted_y,dtype=object)
    acc_test=calc_accuracy(y_test,predicted_y)
    pred_y=predicted_y
    p_y=pred_y.tolist()
    y_t=y_test.tolist()
    conf= metrics.confusion_matrix(p_y,y_t)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf,display_labels=np.array(["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]))
    fig, ax = plt.subplots(figsize=(11,11))
    cm_display.plot(ax=ax)
    wandb.log({"confusion_matrix":plt})
    wandb.run.name = "Confusion Matrix"
    wandb.run.save()
    wandb.run.finish()
    return acc_test
'''
def log_confusion_mat():
    # Initialize Weights & Biases
    wandb.init(project="DA6401_Assignment-1")

    # Data Preprocessing
    _, _, train_x, train_y, val_x, val_y, x_test, y_test = data_preprocess()

    # Model Training Parameters
    params = {
        "hidden_layers": 6,
        "neuron": 64,
        "x_train": train_x,
        "y_train": train_y,
        "x_val": val_x,
        "y_val": val_y,
        "learning_algorithm": "nadam",
        "eta": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "activation": "ReLU",
        "init_method": "xavier",
        "L2_lamb": 0.0005,
        "momentum": 0.9,
        "beta": 0.9,
        "beta1": 0.9,
        "beta2": 0.99,
        "epsilon": 1e-5
    }

    # Train the model
    weights, biases, *_ = learning_params(**params)

    # Forward pass on test data
    L = params["hidden_layers"] + 1
    full_output_test, _, _ = FeedForwardNetwork(weights, biases, L, x_test, "ReLU")

    # Predictions
    predicted_y = np.argmax(full_output_test, axis=0).astype(object)
    acc_test = calc_accuracy(y_test, predicted_y)

    # Confusion Matrix
    conf_matrix = metrics.confusion_matrix(y_test.tolist(), predicted_y.tolist())
    labels = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"])

    fig, ax = plt.subplots(figsize=(11, 11))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    cm_display.plot(ax=ax)

    # Log Confusion Matrix to WandB
    wandb.log({"confusion_matrix": plt})
    wandb.run.name = "Confusion Matrix"
    wandb.run.save()
    wandb.run.finish()

    return acc_test

def wandb_run_configuration(project_name,entity,hidden_layers,neuron,x_train,y_train,x_val,y_val,x_test,y_test,learning_algorithm,eta,epochs,batch_size,activation,init_method,L2_lamb,momentum,beta,beta1,beta2,epsilon,loss):
    wandb.login()
    wandb.init(project=project_name)
    name='_h1_'+str(hidden_layers)+"_SL_"+str(neuron)+"_BS_"+str(batch_size)+"_OPT_"+str(learning_algorithm)+"_loss_"+str(loss)
    weights,biases,epoch_train_loss,epoch_val_loss,acc_train,acc_val=learning_params(hidden_layers,neuron,x_train,y_train,x_val,y_val,learning_algorithm,eta,epochs,batch_size,activation,init_method,L2_lamb,momentum,beta,beta1,beta2,epsilon,loss)
    for i in range(len(epoch_train_loss)):
        wandb.log({"loss":epoch_train_loss[i]})
        wandb.log({"val_loss":epoch_val_loss[i]})
        wandb.log({"accuracy":acc_train[i]})
        wandb.log({"val_acc":acc_val[i]})
        wandb.log({"epoch": (i+1)})
    wandb.log({"validation_accuracy":acc_val[-1]})
    L=hidden_layers+1
    full_output_test,_,_=FeedForwardNetwork(weights,biases,L,x_test,activation)
    predicted_y=np.argmax(full_output_test,axis=0)
    predicted_y=np.array(predicted_y,dtype=object)
    acc_test=calc_accuracy(y_test,predicted_y)
    wandb.log({"test_accuracy":acc_test})
    wandb.run.name = name
    wandb.run.save()
    wandb.run.finish()

def main():
    plot_images()
    _,_,train_x,train_y,val_x,val_y,_,_=data_preprocess()
    run_sweeps(train_x,train_y,val_x,val_y)
    acc_test=log_confusion_mat()
    print(acc_test)

if __name__=="__main__":
    main()

