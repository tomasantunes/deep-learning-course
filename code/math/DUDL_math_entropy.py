# import libraries
import numpy as np
import matplotlib.pyplot as plt

# probability of an event happening
p = .25

# NOT the correct formula!
H = -( p*np.log(p) )
print('Wrong entropy: ' + str(H))





















# the correct way to compute entropy
x = [.25,.75]

H = 0
for p in x:
  H -= p*np.log(p)

print('Correct entropy: ' + str(H))

# also correct, written out for N=2 events
H = -( p*np.log(p) + (1-p)*np.log(1-p) )
print('Correct entropy: ' + str(H))

# note: all probs must sum to 1!
p = [   1,0   ] # sum=1
q = [ .25,.75 ] # sum=1

H = 0
for i in range(len(p)):
  H -= p[i]*np.log(q[i])

print('Cross entropy: ' + str(H))

# also correct, written out for N=2 events
H = -( p[0]*np.log(q[0]) + p[1]*np.log(q[1]) )
print('Correct entropy: ' + str(H))

# simplification
H = -np.log(q[0])
print('Manually simplified: ' + str(H))

# now using pytorch
import torch
import torch.nn.functional as F

# note: inputs must be Tensors
q_tensor = torch.Tensor(q)
p_tensor = torch.Tensor(p)

F.binary_cross_entropy(p,q)
