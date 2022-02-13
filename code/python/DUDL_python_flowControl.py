#!/usr/bin/env python
# coding: utf-8

# # COURSE: A deep understanding of deep learning
# ## SECTION: Introduction to Python: flow control
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202201

# # .
# # .
# # .

# # VIDEO: If-else statements

# In[ ]:


if True:
    print('True is true')


# In[ ]:


if False:
    print('False is true')


# In[ ]:


if 4==4:
    print('hello')


# In[ ]:


import numpy as np

randnum = np.random.randn(1)

if randnum>0:
    print(randnum)


# In[ ]:


randnum = np.random.randn()

if randnum>0:
    print(str(randnum) + ' is positive')
else:
    print(str(randnum) + ' is negative')


# In[ ]:


# slightly better coding

randnum = np.random.randn()

if randnum>0:
    outcome = 'positive'
else:
    outcome = 'negative'

print(str(randnum) + ' is ' + outcome)


# In[ ]:


caketype = 'carrot'

if caketype=='carrot':
    print('Total yum!')
elif caketype=='chocolate':
    print('Yummier!')
elif caketype=='fruit':
    print('Healthy, I guess???')
else:
    print("OK, I'll try it.")


# In[ ]:


# one-liners
caketype = 'Carrot'

if caketype[0]=='c': print('Probably carrot cake.') # nothing different, just in one line

print('Probably carrot cake.') if caketype[0]=='c' else print('Or not. What do I know??')


# In[ ]:


# conjunctive conditionals
if 1<3 and 4==4:
    print('Both conditionals are true')
else:
    print('At least one condition is false')


# ### Exercise

# In[ ]:


# create a dot product function
# function checks that both inputs are numpy arrays (hint: isinstance())
# function checks for same length of two input vectors
# returns dot product if pass, useful error message otherwise


# In[ ]:


def dotproduct(a,b):
    # check that both vectors are numpy arrays
    if not (isinstance(a,np.ndarray) and isinstance(b,np.ndarray)):
        raise Exception('Must be numpy array')

    # check that they have the same length
    if not len(a)==len(b):
        raise Exception('Must be the same length')
    
    # return the dot product
    return sum(a*b)


# In[ ]:


v = np.array([1,2,3,4]) # error when list
w = np.array([3,4,5,1])

print(dotproduct(v,w))
print(np.dot(v,w))


# # .
# # .
# # .

# # VIDEO: For loops

# In[ ]:


for i in range(6):
    print(i)


# In[ ]:


for i in range(4,10):
    print(i,i**2)


# In[ ]:


numbers = np.linspace(3,17,22)

for i in range(len(numbers)):
    x = numbers[i]
    print('Iteration ' + str(i) + ' has a value of ' + str(x))


# In[ ]:


for n in range(18):
    if n%2==0:
        print(str(n) + ' is an even number.')
    else:
        print(str(n) + ' is an odd number.')


# In[ ]:


# soft- vs. hard-coding

iterations = 100

for i in range(iterations):
    pass

for i in range(100):
    pass


# ### Exercise

# In[ ]:


# create a function that reports the Fibonacci series for n+3 numbers

def fibseq(n):
    v = [0,1,1]
    for i in range(n):
        v.append(sum(v[-2:]))
    print(v)


# In[ ]:


fibseq(10)


# # .
# # .
# # .

# # VIDEO: While loop

# In[ ]:


toggle = True
i = 0

while toggle:
  print(i)
  i += 1

  if i==8:
    toggle = False


# In[ ]:


toggle = True
i = 0

while toggle:
  print(i)

  if i==8:
    toggle = False
  i += 1


# In[ ]:


toggle = True
i = 0

while toggle:
  print(i)

  if i==8:
    i += 1
    toggle = False


# In[ ]:


i = 0
while True:
  print(i)
  i += 1
  if i==8:
    break


# In[ ]:


i = 0
while i<8:
  print(i)
  i += 1


# In[ ]:


# prefer a for-loop when you know the number of iterations in advance
# prefer a while loop when you don't know the number of iterations in advance


# In[ ]:


# Exercise: Make a Poisson-like counter. repeat 100 times

# parameter lambda, positive number (10)
# start at 1, multiply by uniform-random number, count iterations until e(-l)


# In[ ]:


import numpy as np

def poissonCounter(lam):

  # initialize
  counter,currval = 0,1
  target = np.exp(-lam)

  # run algorithm (using while vs if)
  while currval>target:
    counter += 1
    currval *= np.random.rand()
  
  # return result
  return counter


# In[ ]:


poissonCounter(10)


# In[ ]:


poissonRand = np.zeros(100)

for i in range(100):
  poissonRand[i] = poissonCounter(10)

poissonRand


# In[ ]:


# same functionality but harder to read
def poissonCounter(l):
  k,p,t = 0,1,np.exp(-l)
  while p>t:
    k += 1
    p *= np.random.rand()
  return k

# appeal of compactness, maybe intentional
# problems:
#  1. harder to read
#  2. harder to find bugs
#  3. harder to modify
# don't pay by character


# # .
# # .
# # .

# # VIDEO: Initializing variables

# In[ ]:


for i in range(10):
  r[i] = i**2

print(r)


# In[ ]:


import numpy as np

r = np.zeros(10)
r = np.full(10,0)
for i in range(10):
  r[i] = i**2

print(r)


# In[ ]:


del r

r = []
for i in range(10):
  r.append(i**2)

print(r)


# In[ ]:


r = np.array([])
for i in range(10):
  r = np.append(r,i**2)

print(r)


# In[ ]:


r = np.zeros((4,5))
r[3,4] = 1
r[6,4] = 1

print(r)


# In[ ]:


r = 4
del r

del r

try:
  del r
except:
  pass


# In[ ]:


# Exercise: time all three methods! (append, np.append, initialize)
# import time
# time.perf_counter()

import time

t = [0]*3
N = 10000

tic = time.perf_counter()
r = []
for i in range(N):
  r.append(i**2)
t[0] = (time.perf_counter() - tic)*1000


tic = time.perf_counter()
r = np.zeros(N)
for i in range(N):
  r[i] = i**2
t[1] = (time.perf_counter() - tic)*1000


tic = time.perf_counter()
r = np.array([])
for i in range(N):
  r = np.append(r,i**2)
t[2] = (time.perf_counter() - tic)*1000


print(t) # in ms


# In[ ]:





# In[ ]:





# In[ ]:





# # VIDEO: Enumerate and zip

# In[ ]:


# enumerate
import numpy as np

somenumbers = np.linspace(-5,5,7)

for i in range(len(somenumbers)):
  print('Index ' + str(i) + ' has value ' + str(somenumbers[i]))

for i,n in enumerate(somenumbers):
  print('Index ' + str(i) + ' has value ' + str(n))


# In[ ]:


text = 'Hello my name is Mike'

vowels = np.zeros(len(text))

for i,l in enumerate(text):
  if l in 'aeiou':
    vowels[i] = 1

vowels


# In[ ]:


somelist = [3,5,4,2,4]
otherlist = ['d','e','r','s','x']

for i in range(len(somelist)):
  print(otherlist[i] + ' ' + str(somelist[i]))


# In[ ]:



for j,i in zip(somelist,otherlist):
  print(i,j)


# In[ ]:


z = zip(somelist,otherlist)
z # z[0]


# In[ ]:


print(z) # empty after running twice
list(z)


# In[ ]:


lz = list(z)
lz


# In[ ]:


lz[0][0]


# In[ ]:


somelist = [3,5,4,2,4]
otherlist = ['d','e']

for i,j in zip(somelist,otherlist):
  print(i,j)


# In[ ]:


# Exercise: given these two lists, create a dictionary using zip
names = ['alpha','beta','gamma']
values = [10,20,40]

D = dict(zip(names,values))
print(D)
D['gamma']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# now try/except
def fun(x,y):
  try:
    return x**y
  except:
    print("Couldn't complete the mission.")

fun(3,'asdf')


# In[ ]:


x = '4'
isinstance(x,str) and x.isnumeric()


# In[ ]:


# exercise: get this to work:
# funfunfun('4',3)


def funfunfun(x,y):
  
  # check if x is not a number
  if not isinstance(x,(int,float)):

    # check if x can be converted to a number
    if isinstance(x,str) and x.isnumeric():
      x = float(x)
      print('Note: converted x to a float.')
    
    # otherwise give error message
    else:
      raise Exception('Input x must be a number.')

  if not isinstance(y,(int,float)):
    raise Exception('Input y must be a number.')

  z = x*y
  return z


funfunfun('4',3)


# In[ ]:


# create a function that will input two numbers and output their product
def funfun(x,y):
  if not isinstance(x,(int,float)):
    raise Exception('Input x must be a number.')
  if not isinstance(y,(int,float)):
    raise Exception('Input y must be a number.')
  z = x*y
  return z

funfun(4,3)


# In[ ]:


# create a function that will input two numbers and output their product
def funfun(x,y):
  z = x*y
  return z

funfun(2,'hi')


# # .
# # .
# # .

# # VIDEO: Function error checking and handling

# # VIDEO: Continue

# In[ ]:


for i in range(10):
  if i%2==0:
    continue
  
  print(i)


# In[ ]:


text = 'HellO, my name is MikE.'

for i in text:
  if i in 'aeiou': # .lower()
    continue
  print(i)


# # .
# # .
# # .

# # .
# # .
# # .

# # VIDEO: Single-line loops

# In[ ]:


# reminder of single-line if-else

import random

if random.randint(1,10)>5: print('I like big numbers.')

print('I prefer smaller numbers.') if random.randint(1,10)>5 else print('I like big numbers.')


# In[ ]:


for i in range(10):
  print(i**2)


# In[ ]:


[print(i**2) for i in range(10)] # output None is output of print


# In[ ]:


n = print('hi')
print(n)


# In[ ]:


# list comprehension (a list from another list)
n = [i**2-i**(1/2) for i in range(10)]
print(n)


# In[ ]:


n = [i**2-i**(1/2) for i in range(10) if i>5]
print(n)


# In[ ]:


# Exercise (maybe in separate video?)
# convert the following loops into single-line loops


# In[ ]:


text = ['Promising','Yves','that','home','on','Nobb']

for word in text:
  print(word[0])

print(' ')

[print(word[0]) for word in text]; # note the #


# In[ ]:


# convert range to list

newlist = ['']*10

for i in range(10):
  if i%2==1:
    newlist[i] = 'Odd'
  else:
    newlist[i] = i
    
newlist


# In[ ]:


newlist = [ 'Odd' if i%2==1 else i for i in range(10) ] # (expression) for iterater in range
newlist


# In[ ]:


import numpy as np

# exercise 2: repeat using lists and single-line for loops
x1 = np.linspace(-3,3,11)
y1 = x1**2

x2 = [i/(5/3) for i in range(-5,6)]
y2 = [i**2 for i in x2]

print(x1)
print(x2)
print(' ')
print(y1)
print(y2)


# # .
# # .
# # .

# # VIDEO: Broadcasting in numpy

# In[ ]:


import numpy as np

X = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]   ])

w = np.array([10,20,30])

# goal: add w to each row of X
Y = X
for i in range(X.shape[0]):
  Y[i,:] = X[i,:] + w

print(Y)
print(' ')
print(X)


# In[ ]:


import copy

X = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]   ])

# goal: add w to each row of X
Y = copy.deepcopy(X)
for i in range(X.shape[0]):
  Y[i,:] = X[i,:] + w

print(Y)
print(' ')
print(X)


# In[ ]:


# using broadcasting
X = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]   ])

Y = X+w
print(Y)
print(' ')
print(X)


# In[ ]:


w = [-1,0,1]
v = np.array([-1,0,1,0],ndmin=2).T

print(X), print(' ')
print(X*v)


# In[ ]:


# exercise: 
# 1) create a list of integers from 0 to 8 (v)
# 2) reshape to a 3x3 matrix (M)
# 3) repeat that matrix to 9x3 (C)
# 4) broadcast-multiply with initial vector (B)

# 1
v = np.arange(9)
print(v), print(' ')

# 2
M = np.reshape(v,(3,3))
print(M), print(' ')

# 3
C = np.tile(M,(3,1))
print(C), print(' ')

# 4
B = C*np.reshape(v,(len(v),1))
print(B)

