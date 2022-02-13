#!/usr/bin/env python
# coding: utf-8

# # COURSE: A deep understanding of deep learning
# ## SECTION: Introduction to Python: indexing and slicing
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202201

# # .
# # .
# # .

# # VIDEO: Indexing

# In[ ]:


arange = range(5)

arange


# In[ ]:


alist = list(arange)
alist


# In[ ]:


alist[1]


# In[ ]:


alist = [5,4,1,-67,343,34]

alist[1]


# In[ ]:


print( alist[5] )

print( alist[-1] )
print( alist[-2] )


# In[ ]:


# can also use variables
idx = 3
alist[idx]


# In[ ]:


idxs = [4,2]
alist[idxs]


# In[ ]:


alist = [ 4,3,[4,3,5] ]

alist[2][1]


# ### Exercise

# In[ ]:


# Get the attribute of Penguin in the following list
listlist = [ 4,'hi',[5,4,3],'yo',{'Squirrel':'cute','Penguin':'Yummy'} ]


# In[ ]:



listlist[4]['Penguin']


# # .
# # .
# # .

# # VIDEO: Slicing, part 1

# In[ ]:


alist = list( range(5,11) )
alist


# In[ ]:


print( alist[0:2] )
print( alist[0:1] )


# In[ ]:


alist[2:5]


# In[ ]:


alist[2:]


# In[ ]:


alist[:4]


# In[ ]:


alist[::2]


# In[ ]:


# using variables
start = 1
stop  = 4
skip  = 2

alist[start:stop:skip]


# In[ ]:


# to reverse a list
alist[::-1]


# ### Exercise

# In[ ]:


# use slicing to write "Mike is a nice guy" from this text
text = '345dfyug ecin a si ekiM 93845d'

text[-8:4:-1]


# # .
# # .
# # .

# In[ ]:


# Broadcasting

