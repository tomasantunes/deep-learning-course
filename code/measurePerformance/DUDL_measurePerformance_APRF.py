# import libraries
import numpy as np
import matplotlib.pyplot as plt

# update default font size
import matplotlib
matplotlib.rcParams.update({'font.size':18})

## run experiment

# number of 'trials' in the experiment
N = 50 # actual trials is 2N

# number of experiment repetitions
numExps = 10000

# initialize
accuracy  = np.zeros(numExps)
precision = np.zeros(numExps)
recall    = np.zeros(numExps)
F1score   = np.zeros(numExps)


### run the experiment!
for expi in range(numExps):
    
  # generate data
  TP = np.random.randint(1,N)  # true positives,  aka hits
  FN = N-TP                    # false negatives, aka misses
  TN = np.random.randint(1,N)  # true negatives,  aka correct rejections
  FP = N-TN                    # false positives, aka false alarms
  

  ### the four performance measures discussed in lecture

  # accuracy
  accuracy[expi]  = (TP+TN) / (2*N)

  # precision
  precision[expi] = TP / (TP+FP)

  # recall
  recall[expi]    = TP / (TP+FN)

  # Fscore
  F1score[expi]   = TP / (TP+(FP+FN)/2)
  

## let's see how they relate to each other!

fig,ax = plt.subplots(1,2,figsize=(18,6))

ax[0].scatter(accuracy,F1score,s=5,c=precision)
ax[0].plot([0,1],[.5,.5],'k--',linewidth=.5)
ax[0].plot([.5,.5],[0,1],'k--',linewidth=.5)
ax[0].set_xlabel('Accuracy')
ax[0].set_ylabel('F1-score')
ax[0].set_title('F1-Accuracy by precision')


ax[1].scatter(accuracy,F1score,s=5,c=recall)
ax[1].plot([0,1],[.5,.5],'k--',linewidth=.5)
ax[1].plot([.5,.5],[0,1],'k--',linewidth=.5)
ax[1].set_xlabel('Accuracy')
ax[1].set_ylabel('F1-score')
ax[1].set_title('F1-Accuracy by recall')

plt.show()




