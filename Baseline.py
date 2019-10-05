# Implementation of a standard feed forward network
#
# program written by Daniel Hao, April 2019


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as f
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
X = f.normalize(X, p = 1)
Y = Variable(torch.Tensor(train_target.values).long())


# define the number of inputs, classes, training epochs, weight decay
# and learning rates for different regions of casper network
num_epochs = 632
output_neurons = 7
input_neurons = 10
learning_rate = 0.2


# define a customised fully connected feed forward neural network structure
class MultiLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_output):
        super(MultiLayerNet, self).__init__()
        self.hidden = torch.nn.Linear(n_input, 9)
        self.hidden2 = torch.nn.Linear(9, 8)

        # define linear output layer output
        self.out = torch.nn.Linear(8, n_output)

    def forward(self, x):

        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = torch.tanh(h_input)

        h_input2 = self.hidden2(h_output)
        h_output2 = torch.tanh(h_input2)



        # get output layer output
        y_pred = self.out(h_output2)

        return y_pred

# define a neural network using the customised structure
net = MultiLayerNet(input_neurons, output_neurons)


# define loss function
loss_func = torch.nn.CrossEntropyLoss()


# define optimizer for standard network
optimiser = optim.Rprop(net.parameters(), lr =learning_rate, etas = (0.5, 1.2),
                        step_sizes=(1e-06, 50))

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


    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()


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
X_test = f.normalize(X_test, p = 1)
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
