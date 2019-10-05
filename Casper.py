# Implementation of a casper network
#
# program written by Daniel Hao, April 2019


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

# calculate balanced accuracy of the network given a confusion matrix
def balanced_accuracy(confusion, dimension):
    balanced_accuracy = 0
    for i in range(dimension):
        total = 0
        for j in range(dimension):
            total += confusion[i][j]
        balanced_accuracy += confusion[i][i] / total
    balanced_accuracy = balanced_accuracy / dimension
    return balanced_accuracy


data = pd.read_excel("SFEW_processed.xlsx")

 # remove rows with na valuesq
data = data.dropna()

# subtract 1 from all the label values, such that it starts from 0
data["label"] -= 1

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True) 


# randomly split data into training set (90%) and testing set (10%)
msk = np.random.rand(len(data)) < 0.9
train_data = data[msk]
test_data = data[~msk]

# split training data into input and target
# the first 9 columns are features, the last one is target
train_input = train_data.iloc[:, 1:]
train_target = train_data.iloc[:, 0]

# split testing data into input and target
# the first column is the target, the rest are features
test_input = test_data.iloc[:, 1:]
test_target = test_data.iloc[:, 0]

# create Tensors to hold inputs and outputs, and wrap them in Variables,
# as Torch only trains neural network on Variables
X = Variable(torch.Tensor(train_input.values).float())
Y = Variable(torch.Tensor(train_target.values).long())


# define the number of inputs, classes, training epochs, weight decay
# and learning rates for different regions of casper network
num_epochs = 1200
output_neurons = 7
input_neurons = 10
weight_decay = 0.998

L1 = 0.25
L2 = 0.01
L3 = 0.001

#  define a casper network
class CasperNetwork(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(CasperNetwork, self).__init__()
        
        # create input layer with a randomly initialized weight
        self.Initial = torch.nn.Linear(n_input, n_output) 
        self.Initial.weight.data = self.initialize_weights(n_input, n_output, 
                                                    -0.7, 0.7)
        
        # This list contains all the input connections to hidden neurons
        self.old_input_neurons = nn.ModuleList([]) 
        # This list contains all the ouput connections from previous neurons
        self.old_output_neurons = nn.ModuleList([]) 
        self.n_neurons = 0
        
        self.input_size = n_input
        self.output_size = n_output

        # initialize casper network with no hidden neurons
        self.L1 = None
        self.L2 = None

        self.output_layer = torch.nn.Linear(n_input, n_output) 
        self.output_layer.weight.data = self.initialize_weights(n_input, 
                                                                n_output, 
                                                                -0.7, 0.7)
    def forward(self, x):
        # calculate output from input layer
        out = x

        # if there are no hidden nerons, simply return the output
        if len(self.old_input_neurons) == 0:
            if self.L1 == None:
                
                # if no neurons has been inserted, simply pass input to ouput
                return self.Initial(x)
            
            # if there is a single hidden neuron, 
            # then add its output to the output layer
            else:
                
                temp = torch.tanh(self.L1(x))
                temp = torch.tanh(self.L2(temp))
                out = torch.cat((out, temp), 1)
                
        
        else:
            # if there is more than 1 hidden neuron, loop through the list of
            # casper neurons and add their output to the final output layer
            for index in range(0, len(self.old_input_neurons)):
                
                # calculate the inputs to the single casper neuron
                previous = self.old_input_neurons[index](x)
                
                # concatenate the output from the single casper neuron to the 
                # outputs of all previous neurons
                out = torch.cat((out, self.old_output_neurons[index]
                                (torch.tanh(previous))), 1) 

                # concatenate the single neuron to the input 
                x = torch.cat((x, previous), 1)
                
            # calculate the output from the most recent casper neuron 
            # add them to the final output layer
            new_neuron_input = torch.tanh(self.L1(x))
            new_neuron_output = torch.tanh(self.L2(new_neuron_input))
            out = torch.cat((out, new_neuron_output), 1)
          
        return self.output_layer(out)
    
     
    # adds new casper neuron to the network, which would be 2 linear layers
    # layer1: (input, 1) layer2: (1, output), 
    # the flow would be inputs -> 1 neuron -> output
    def add_layer(self):
        self.n_neurons += 1
        
        # concatenate all outputs from the hidden neurons and original input
        # to go into the output neurons
        previous_weights = self.output_layer.weight.data
        total_outputs = self.n_neurons + self.input_size
        self.output_layer = torch.nn.Linear(total_outputs , self.output_size) 
        
        # copy over the previsouly learnt weights, and initialize random
        # weight values for new neurons
        self.output_layer.weight.data = self.copy_initialize_weights(
                                                            previous_weights, 
                                                            total_outputs, 
                                                            self.output_size, 
                                                            -0.1, 0.1)
        
        # create the layers for the new casper neuron
        new_layer_in = torch.nn.Linear(self.input_size + self.n_neurons - 1, 1)
        # we pass it through another neuron in order to create an per neuron
        # learning rate for the final layer
        new_layer_out = torch.nn.Linear(1, 1)
        
        total_inputs = self.input_size + self.n_neurons - 1
        
        # initialize weights
        new_layer_in.weight.data = self.initialize_weights(total_inputs, 
                                                         1, -0.1, 0.1)
        new_layer_out.weight.data = self.initialize_weights(1, 1, -0.1, 0.1)
        
        # assign the layers to the network
        if self.L1 == None and self.L2 == None:
            self.L1 = new_layer_in
            self.L2 = new_layer_out
        
        else:
            self.old_input_neurons.append(self.L1)
            self.old_output_neurons.append(self.L2)
            self.L1 = new_layer_in
            self.L2 = new_layer_out
         
        
        
    # create a list of weights for initialization
    def initialize_weights(self, n_input, n_output, lower, upper):
        final = []
        for inputs in range(0, n_output):
            weights = []
            for value in range(0, n_input):
                weights.append(np.random.uniform(lower, upper))
            final.append(weights)
        return torch.Tensor(final)
    
    # Creates a list of weights for initialization, but also copies over the 
    # previous weights avoiding the need to relearn
    def copy_initialize_weights(self, previous_weight, n_input, n_output, 
                                lower, upper):
        final = []
        
        for row in range(0, n_output):
            weights = []
            
            for value in range(0, len(previous_weight[row])):
                weights.append(previous_weight[row][value])
                    
            for new_weight in range(0, n_input - len(previous_weight[row])):
                weights.append(np.random.uniform(lower, upper))

            final.append(weights)
        return torch.Tensor(final)
    
    
    def applyWeightDecay(self, decay):
        self.Initial.weight.data *= decay
        
        if self.L1 != None:
            self.L1.weight.data *= decay
            self.L2.weight.data *= decay
            
        if len(self.old_input_neurons) != 0:
            for layers in self.old_input_neurons:
                layers.weight.data *= decay
                
            for layers in self.old_output_neurons:
                layers.weight.data *= decay 
                
                
# define a neural network using the customised structure 
net = CasperNetwork(input_neurons, output_neurons)


# define loss function
loss_func = torch.nn.CrossEntropyLoss()



# define optimiser with per layer learning rates
# optimiser without any hidden neurons
optimiser = optim.Rprop([
                {'params': net.Initial.parameters(), 'lr' : L1},
                {'params': net.output_layer.parameters()},
                {'params': net.old_input_neurons.parameters()},
                {'params': net.old_output_neurons.parameters()}], 
                lr = L3, etas = (0.5, 1.2), step_sizes=(1e-06, 50))

# store all losses for visualisation
all_losses = []


previous_loss = None

# train a neural network
for epoch in range(num_epochs):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)

    # Compute loss
    loss = loss_func(Y_pred, Y)
    
    all_losses.append(loss.item())

    # print progress
    if epoch % 40 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(Y_pred, 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct)/total))
        
        # if the rate to which the loss value decreases slows beyond a certain
        # threshold, then add a casper neuron
        if (previous_loss != None and previous_loss > loss.item() and 
                                    previous_loss - loss.item() < 0.001) :
            
            net.add_layer()

            # adding custom learning rates to hidden neurons
            optimiser = optim.Rprop([
                {'params': net.Initial.parameters()},
                {'params': net.old_input_neurons.parameters()},
                {'params': net.old_output_neurons.parameters()},
                {'params': net.output_layer.parameters()},
                {'params': net.L1.parameters(), 'lr': L1},
                {'params': net.L2.parameters(), 'lr': L2}], 
                lr = L3, etas = (0.5, 1.2), step_sizes=(1e-06, 50))

        previous_loss = loss.item()
        
        
    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

    net.applyWeightDecay(weight_decay)



# plot the loss graph  
plt.figure()
plt.plot(all_losses)
plt.show()


"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""

# create Tensors to hold inputs and outputs, and wrap them in Variables,
# as Torch only trains neural network on Variables
X_test = Variable(torch.Tensor(test_input.values).float())
Y_test = Variable(torch.Tensor(test_target.values).long())

# test the neural network using testing data
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))



"""
Compute the confusion matrix and balanced accuracy for result analysis
"""

confusion_test = torch.zeros(output_neurons, output_neurons)


for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]
    
    confusion_test[actual_class][predicted_class] += 1


print('')
print('Balanced Accuracy: %.4f %%' % (100 * balanced_accuracy(confusion_test,
                                                             output_neurons)))

print('')
print('Confusion matrix for testing:')
print(confusion_test)

