# import libraries
import torch
import torch.nn as nn

# build a model
aModel = nn.Sequential(
    nn.Linear(10,14),  # input layer
    nn.Linear(14,19),  # hidden layer
    nn.Linear(19,8),   # output layer
      )

aModel

# print the sizes of the weights matrices in each layer
for i in range(len(aModel)):
  print( aModel[i].weight.shape )

M2 = nn.Sequential(
    nn.Linear(10,14),  # input layer
    nn.Linear(14,9),   # hidden layer
    nn.Linear(19,8),   # output layer
      )

for i in range(len(M2)):
  print( M2[i].weight.shape )

# generate the data
nsamples = 5
nfeatures = 10

fakedata = torch.randn(nsamples,nfeatures)

# test the first model

# does the size of the output make sense?
aModel(fakedata).shape

# test the second model

# does the size of the output make sense?
M2(fakedata).shape


