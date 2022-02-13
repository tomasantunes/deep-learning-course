# import libraries
import numpy as np
import matplotlib.pyplot as plt

# define a set of points to evaluate
x = np.linspace(.0001,1,200)

# compute their log
logx = np.log(x)



# plot!
fig = plt.figure(figsize=(10,4))

# increase font size. FYI
plt.rcParams.update({'font.size':15})

plt.plot(x,logx,'ks-',markerfacecolor='w')
plt.xlabel('x')
plt.ylabel('log(x)')
plt.show()

# demonstration that log and exp are inverses

# redefine with fewer points
x = np.linspace(.0001,1,20)

# log and exp
logx = np.log(x)
expx = np.exp(x)

# the plot
plt.plot(x,x,color=[.8,.8,.8])
plt.plot(x,np.exp(logx),'o',markersize=8)
plt.plot(x,np.log(expx),'x',markersize=8)
plt.xlabel('x')
plt.ylabel('f(g(x))')
plt.legend(['unity','exp(log(x))','log(exp(x))'])
plt.show()

