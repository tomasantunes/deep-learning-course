# import libraries
import numpy as np
import torch
import torch.nn as nn

# parameters
inChans  = 1 # RGB
imsize   = [256,256]
outChans = 1
krnSize  = 7 # should be an odd number
stride   = (1,3)
padding  = 1

# create the instance
c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

# create an image
img = torch.rand(1,inChans,imsize[0],imsize[1])

# run convolution and compute its shape
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size of the result according to the formula
expectSize = np.array([outChans,0,0],dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size of the output
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')

# parameters
inChans  = 
imsize   = 
outChans = 
krnSize  = 
stride   = 
padding  = 

# create the instance
c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

# create an image
img = torch.rand(1,inChans,imsize[0],imsize[1])

# run convolution and compute its shape
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size of the result according to the formula
expectSize = np.array([outChans,0,0],dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size of the output
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')

# parameters
inChans  = 
imsize   = 
outChans = 
krnSize  = 
stride   = 
padding  = 

# create the instance
c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

# create an image
img = torch.rand(1,inChans,imsize[0],imsize[1])

# run convolution and compute its shape
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size of the result according to the formula
expectSize = np.array([outChans,0,0],dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size of the output
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')

# note: these dimensions are the input -> first hidden layer of the famous LeNet-5

# parameters
inChans  = 
imsize   = 
outChans = 
krnSize  = 
stride   = 
padding  = 

# create the instance
c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

# create an image
img = torch.rand(1,inChans,imsize[0],imsize[1])

# run convolution and compute its shape
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size of the result according to the formula
expectSize = np.array([outChans,0,0],dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size of the output
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')

# note: these dimensions are the input -> first hidden layer of the famous AlexNet

# parameters
inChans  = 
imsize   = 
outChans = 
krnSize  = 
stride   = 
padding  = 

# create the instance
c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

# create an image
img = torch.rand(1,inChans,imsize[0],imsize[1])

# run convolution and compute its shape
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size of the result according to the formula
expectSize = np.array([outChans,0,0],dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size of the output
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')

# note: these dimensions are the input -> first hidden layer of the famous VGG-16

# parameters
inChans  = 
imsize   = 
outChans = 
krnSize  = 
stride   = 
padding  = 

# create the instance
c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

# create an image
img = torch.rand(1,inChans,imsize[0],imsize[1])

# run convolution and compute its shape
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size of the result according to the formula
expectSize = np.array([outChans,0,0],dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size of the output
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')



















# 1)
inChans  = 3
imsize   = [64,64]
outChans = 10
krnSize  = 9
stride   = (2,2)
padding  = 0

# 2)
inChans  = 3
imsize   = [196,96]
outChans = 5
krnSize  = 5
stride   = (3,2)
padding  = 3

# 3)
inChans  = 1
imsize   = [32,32]
outChans = 6
krnSize  = 5
stride   = (1,1)
padding  = 0

# 4)
inChans  = 3
imsize   = [227,227]
outChans = 96
krnSize  = 11
stride   = (4,4)
padding  = 1

# 5)
inChans  = 3
imsize   = [224,224]
outChans = 64
krnSize  = 3
stride   = (1,1)
padding  = 1
