#!/usr/bin/env python
# coding: utf-8

# # COURSE: A deep understanding of deep learning
# ## SECTION: Introduction to Python: functions
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202201

# # .
# # .
# # .

# # VIDEO: Inputs and outputs

# In[ ]:


sum([10,20,30])


# In[ ]:


alist = [2,3,4]

sum(alist)


# In[ ]:


# alist(0)
sum[alist]


# In[ ]:


len(alist)


# In[ ]:


# give outputs
listsum = sum(alist)
print('The sum is ' + str(listsum))


# ### Exercise

# In[ ]:


# compute the average
sum(alist) / len(alist)


# # .
# # .
# # .

# # VIDEO: Python modules (numpy, pandas)
# 

# In[ ]:


numbers = [1,2,3,4,5]
mean(numbers)
average(numbers)


# In[ ]:


import numpy as np

np.mean(numbers)


# In[ ]:


np.linspace(1,5,7)


# In[ ]:


# clear the workspace


# In[ ]:


# new type: numpy array
numbernp = np.array([1,2,3,4,5])

print(numbernp)
print(numbers)


# In[ ]:


print(type(numbernp))
print(type(numbers))


# In[ ]:


numberz.min()


# In[ ]:


n = [4,3,5,2,6,1,7]
print(n)
n.sort()
print(n)


# In[ ]:


n = [4,3,5,2,6,1,7]
print(n)
np.sort(n)
print(n)
n = np.sort(n) # notice different variable type; could set back
print(n)


# In[ ]:





# In[ ]:


import pandas as pd

# create some random data
var1 = np.random.randn(100)*5 + 20
var2 = np.random.randn(100)>0

# variable labels
labels = ['Temp. (C)','Ice cream']

# create as a dictionary
D = {labels[0]:var1, labels[1]:var2}

# import to pandas dataframe
df = pd.DataFrame(data=D)


# In[ ]:


df.head()


# In[ ]:


df.count()


# In[ ]:


df.mean()


# ### Exercise

# In[ ]:


# create a pandas dataframe with variables: 
# integers from 0 to 10, their square, and their log

nums = np.array(range(11))

D = {'v1':nums , 'v2':nums**2 , 'v3':np.log(nums)}

df = pd.DataFrame(D)
df


# In[ ]:





# # VIDEO: Getting help on functions

# In[ ]:


alist = [1,2,3]

sum(alist,10)


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'sum')


# In[ ]:


help(sum)


# In[ ]:


# advanced method

import inspect
import numpy as np
inspect.getsourcelines(np.linspace)


# In[ ]:


sum('test')


# # . 
# # . 
# # . 

# # VIDEO: Creating functions

# In[ ]:


def awesomeFunction():
  print(1+1)


# In[ ]:


awesomeFunction
awesomeFunction()


# In[ ]:


def SuperAwesomeFunction(input1,input2):
  return input1 + input2


# In[ ]:


SuperAwesomeFunction(5,8)


# In[ ]:


answer = SuperAwesomeFunction(5,8)


# In[ ]:


print(answer)


# In[ ]:


# with multiple outputs
def awesomeFunction(in1,in2):
  sumres = in1+in2
  prodres = in1*in2
  print('Their sum is ' + str(sumres))
  print('Their product is ' + str(prodres))
  return sumres,prodres


# In[ ]:


out1,out2 = awesomeFunction(4,5)

print(out1,out2)


# In[ ]:


# lambda functions

littlefun = lambda x : x**2 - 1

littlefun(4)


# ### Exercise

# In[ ]:


# create a function that computes a factorial, 
# then compare against the numpy factorial function
# hint: import math

def myfactorial(val):
  return np.prod(np.arange(1,val+1))


# In[ ]:


import math

x = 20 # inaccurate for >~21
print(np.arange(1,x+1))

print(myfactorial(x),math.factorial(x))
get_ipython().run_line_magic('pinfo2', 'math.factorial')


# In[ ]:





# # .
# # .
# # .

# # VIDEO: Global and local variable scopes
# 
# 
# 

# In[ ]:


# clear the workspace


# In[ ]:


def funfun():
  x = 7
  y = 10
  print(x)


# In[ ]:


funfun()


# In[ ]:


x


# In[ ]:


x = 3
print( funfun() )
print( x )
print( y )


# In[ ]:


def funfun():
  print(z)


# In[ ]:


funfun()


# In[ ]:


z = 3
funfun()


# In[ ]:


# rules: 
#  1) variables created inside a function are local (accessible only inside the function)
#  2) variables created outside a function are global (accessible in or out of the function)


# ### Exercise

# In[ ]:


# write a function that flips a coin N times and reports the average

import numpy as np

def coinflip(N):
  propCoinFlips = np.mean( np.random.randn(N)>0 )
  print( str(N) + ' coin flips had ' + str(propCoinFlips*100) + '% heads.' )


# In[ ]:


coinflip(2000)


# # .
# # .
# # .

# In[ ]:





# In[ ]:





# In[ ]:


# tangent on copy

a = [4,3]
b = a#[:]
b[0] = 5

print(a)
print(b)


# In[ ]:


print(id(a))
print(id(b))


# In[ ]:


import copy

a = {'q':1, 'w':2}
b = copy.deepcopy(a)

print(id(a))
print(id(b))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # VIDEO: Classes and object-oriented programming

# In[1]:


# class is like a blueprint for a set of attributes and methods
# instance is an example
# class is a cookie cutter, instance is an individual cookie

class model(object):

  # constructor method
  def __init__(self,numlayers,numunits,name):
    self.layers  = numlayers
    self.units   = numunits # these are attributes
    self.name    = name
    self.weights = 10
  
  # other methods
  def howManyUnits(self):
    totalUnits = self.layers * self.units
    print(f'There are {totalUnits} units in the model.')

  def trainTheModel(self,x):
    self.weights += x
    return self.weights

  def __str__(self):
    return f'This is a {self.name} architecture.'


# In[ ]:


# create an instance and check it
ex = model(2,3,'cnn')
# ex.howManyUnits()
str(ex)


# In[ ]:


ex.trainTheModel(3)


# In[ ]:


# Exercise: 
# create a new class so that 
# 1) the weights is a layers-by-units matrix, 
# 2) the weights are changed by multiplying by input x and summing input y


# In[ ]:


import numpy as np
class model2(object):

  # constructor method
  def __init__(self,numlayers,numunits,name):
    self.layers  = numlayers
    self.units   = numunits # these are attributes
    self.name    = name
    self.weights = np.random.randn(numlayers,self.units)
  
  # other methods
  def howManyUnits(self):
    totalUnits = self.layers * self.units
    print(f'There are {totalUnits} units in the model.')

  def trainTheModel(self,x,y):
    self.weights = self.weights*x + y

  def __str__(self):
    return f'This is a {self.name} architecture.'


# In[ ]:


ex1 = model2(2,3,'cnn')
print(ex1.weights)


# In[ ]:


ex1.trainTheModel(2,3)


# In[ ]:


ex1.weights

