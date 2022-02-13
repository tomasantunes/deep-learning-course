#!/usr/bin/env python
# coding: utf-8

# # COURSE: A deep understanding of deep learning
# ## SECTION: Introduction to Python: text and plots
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202201

# # .
# # .
# # .

# # VIDEO: fprintf and f-strings
# 

# In[ ]:


# string formatting

num = 45.3
sng = 'cellar doors'

print( 'I would like to have ' + str(num) + ' ' + sng )


# In[ ]:


print( 'I would like to have %g %s.'%(num,sng) ) # i, g, s

print( 'I would like to have %10.3f %s.'%(num,sng) )


# In[ ]:


print( 'I would like to have %g %s.'%(num**2/18.5,sng) )


# In[ ]:


# f-strings
print(f'I would like to have {num} {sng}.')


# In[ ]:



print('I would like to have {n} {s}.'.format(n=num,s=sng))


# ### Exercise

# In[ ]:


# Exercise: print the letter and number of the alphabet. use correct ordinal indicators.

import string
letters = string.ascii_lowercase

for i in range(len(letters)):
  if i==0 or i==20:
    ordind = 'st'
  elif i==1 or i==21:
    ordind = 'nd'
  elif i==2 or i==22:
    ordind = 'rd'
  else:
    ordind = 'th'

  # print('%s is the %g%s letter of the alphabet.' %(letters[i],i+1,ordind))
  print(f'{letters[i]} is the {i+1}{ordind} letter of the alphabet.')


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(.001,3,99)

# fix,ax = plt.subplots()
# ax.plot(x,np.exp(x),color='b')
# ax.set_ylabel('y=exp(x)')

# axx = ax.twinx()
# axx.plot(x,np.log(x),color='r')
# axx.set_ylabel('y=log(x)')
# axx.set_xlabel('x')


# # .
# # .
# # .

# # VIDEO: Plotting dots and lines

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(3,4)#,'o')
plt.plot(2,4,'s')


# In[ ]:


plt.plot(1,2,'ro')
plt.plot(2,4,'bp')
plt.show()

plt.plot(3,2,'ro',label='red circle')
plt.plot(3,4,'bp',label='blue pentagon')
plt.legend()
plt.show()


# In[ ]:


# variables in plt
a = 3
b = 4
plt.plot(a,a*2,'s')
plt.plot(b,b**2,'o')


# In[ ]:


# plot lines

plt.plot([0,1],[0,2],'r')
plt.plot([0,0],[0,2],'g')


# In[ ]:


import numpy as np

x = np.linspace(0,3*np.pi,101)
y = np.sin(x)

plt.plot(x,y,'k.')


# In[ ]:


# exercise
# draw you favorite letter! If your favorite letter has curves in it, then pick a new favorite letter with only straight lines.

plt.plot([0,1],[0,1],'m',linewidth=14)
plt.plot([0,1],[1,0],'m')
plt.axis('square')
plt.show()


# # .
# # .
# # .

# # VIDEO: Subplot geometry

# In[ ]:


plt.subplot(1,2,1)
plt.plot(np.random.randn(10))

plt.subplot(1,2,2)
plt.plot(np.random.randn(10))

plt.show()


# In[ ]:


x = np.arange(10)

fig,ax = plt.subplots(1,3,figsize=(15,3)) # (w,h)

ax[0].plot(x,x**2,'b')
ax[1].plot(x,np.sqrt(x),'r')
ax[2].plot(x,x)

plt.show()


# In[ ]:


fig,ax = plt.subplots(2,2)

ax[0,0].plot(np.random.randn(4,4))
ax[0,1].plot(np.random.randn(4,4))
ax[1,0].plot(np.random.randn(4,4))
ax[1,1].plot(np.random.randn(4,4))

plt.tight_layout()
plt.show()


# ### Exercise

# In[ ]:


# Create a 3x3 subplot geometry and populate it inside a for loop
# hint: .flatten() method in numpy

M = np.random.randint(10,size=(4,4))

print(M)
print(' ')
print(M.flatten())


# In[ ]:



fig,ax = plt.subplots(3,3,figsize=(7,7))

for i in ax.flatten():
  i.plot(np.random.randn(4,5))

plt.show()


# # .
# # .
# # .
# 

# # VIDEO: Making the graphs look nicer

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


x = np.linspace(-3,3,101)

plt.plot(x,x,label='y=x')
plt.plot(x,x**2,label='y**2')
plt.plot(x,x**3,label='y**3')

plt.legend()
plt.xlabel('X')
plt.ylabel('y=f(x)')
plt.title('A really awesome plot')

plt.xlim([-3,3])
plt.ylim([-10,10])
# plt.axis('square') # square, equal
plt.gca().set_aspect(.3)#'auto') # or number
# plt.gca().set_aspect(1./plt.gca().get_data_ratio())

plt.show()


# In[ ]:


x = np.linspace(-3,3,101)

fix,ax = plt.subplots()

ax.plot(x,x,label='y=x')
ax.plot(x,x**2,label='y**2')
ax.plot(x,x**3,label='y**3',color=[.8,.1,.7])

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('y=f(x)')
ax.set_title('A really awesome plot')

ax.set_xlim([-3,3])
ax.set_ylim([-10,10])
ax.set_aspect('equal')
ax.set_aspect(1/ax.get_data_ratio())

ax.grid()

plt.show()


# # Exercise

# In[ ]:


fig,ax = plt.subplots()

x = np.linspace(0,10,100)

for i in np.linspace(0,1,50):
  ax.plot(x,x**i,color=[i/2,0,i])

plt.show()


# # .
# # .
# # .

# # VIDEO: Adding annotations

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-5,6)
y = x**2

fig, ax = plt.subplots()
ax.plot(x,y,'ko-',markerfacecolor='w',markersize=15)
ax.annotate('Hello',(x[3],y[3]),arrowprops=dict(),
            xycoords='data',xytext=(0,10),horizontalalignment='center',fontsize=17)

plt.show()


# In[ ]:


with plt.xkcd():
  plt.plot(x,y)
  plt.show()


# In[ ]:


# Exercise: annotate the minimum

y = x**2 + np.cos(x)*10

minpnt = np.argmin(y)

# note: argmin only returns the first instance of a minimum, even if that minimum value 
#   appears multiple times in the array. To find *all* minimum points, yuo can find all 
#   indices that equal the minimum:
minpnts = np.where(y==np.min(y))

txt = 'min: (%g,%.2f)'%(x[minpnt],y[minpnt])

fig, ax = plt.subplots()
ax.plot(x,y,'ko-',markerfacecolor='purple',markersize=15)
ax.annotate(txt,(x[minpnt],y[minpnt]),arrowprops=dict(color='purple',arrowstyle='wedge'),
            xycoords='data',
            xytext=(x[int(len(x)/2)],np.max(y)*.8),horizontalalignment='center',
            fontsize=14)

ax.set_xlabel('x')
ax.set_ylabel('f(x) = x^2 + 10cos(x)')

plt.show()


# # .
# # .
# # .

# # VIDEO: Seaborn
# 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

n = 200
D = np.zeros((n,2))
D[:,0] = np.linspace(0,10,n) + np.random.randn(n)
D[:,1] = D[:,0]**2 + np.random.randn(n)*10

sns.jointplot(D[:,0],D[:,1])
plt.show()


# In[ ]:



df = pd.DataFrame(data=D,columns=['var1','var2'])
sns.jointplot(df.columns[0],df.columns[1],data=df,kind='kde',color='purple')
plt.show()


# In[ ]:



x = np.linspace(-1,1,n)
y1 = x**2
y2 = np.sin(3*x)
y3 = np.exp(-10*x**2)

sns.scatterplot(x=y1,y=y2,hue=y3,legend=False,palette='jet')
plt.show()


# In[ ]:


# Exercise: make a regression plot in seaborn
sns.regplot(df.columns[0],df.columns[1],data=df,color='green')
plt.title(f'regression of {df.columns[0]} on {df.columns[1]}')

plt.show()


# # .
# # .
# # .

# # VIDEO: images

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

m = 3
n = 5

M = np.random.randint(10,size=(m,n))
print(M)

plt.imshow(M)

for i in range(m):
  for j in range(n):
    plt.text(j,i,str(M[i,j]),horizontalalignment='center',fontsize=20)

plt.colorbar()
plt.show()


# In[ ]:


from imageio import imread

img = imread('https://upload.wikimedia.org/wikipedia/en/8/86/Einstein_tongue.jpg')

plt.imshow(img)
plt.title('The smart guy')
plt.show()


# In[ ]:


print(type(img))
print(np.shape(img))

plt.hist(img.flatten(),bins=100)
plt.show()


# In[ ]:



plt.imshow(img,vmin=150,vmax=250,cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()


# ## Exercise: Visualize the Hilbert matrix using Seaborn
# 
# $$H_{i,j} = \frac{1}{i+j-1}$$
# 
# 

# In[ ]:


# Exercise: Hilbert matrix and show using seaborn
import seaborn as sns

n = 10

H = np.zeros((n,n))

for i in range(n):
  for j in range(n):
    H[i,j] = 1 / (i+j+1)

sns.heatmap(H,vmin=0,vmax=.4)
plt.show()


# In[ ]:





# In[ ]:





# # VIDEO: Export plots in low and high resolution

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

from IPython import display
display.set_matplotlib_formats('svg')

x  = np.linspace(.5,5,10)
y1 = np.log(x)
y2 = 2-np.sqrt(x)

plt.plot(x,y1,'bo-',label='log')
plt.plot(x,y2,'rs-',label='sqrt')

plt.legend()
plt.show()

plt.savefig('test.png')
plt.savefig('test.pdf')


# In[ ]:


# download to disk

from google.colab import files
files.download('test.pdf')
files.download('test.png')
files.download('test.svg')

