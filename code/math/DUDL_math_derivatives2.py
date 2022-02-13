# import libraries
import numpy as np
import sympy as sym

# make the equations look nicer
from IPython.display import display

# create symbolic variables in sympy
x = sym.symbols('x')

# create two functions
fx = 2*x**2
gx = 4*x**3 - 3*x**4

# compute their individual derivatives
df = sym.diff(fx)
dg = sym.diff(gx)

# apply the product rule "manually"
manual = df*gx + fx*dg
thewrongway = df*dg

# via sympy
viasympy = sym.diff( fx*gx )


# print everything
print('The functions:')
display(fx)
display(gx)
print(' ')

print('Their derivatives:')
display(df)
display(dg)
print(' ')

print('Manual product rule:')
display(manual)
print(' ')

print('Via sympy:')
display(viasympy)
print(' ')


print('The wrong way:')
display(thewrongway)

# repeat with chain rule
gx = x**2 + 4*x**3
fx = ( gx )**5

print('The function:')
display(fx)
print(' ')

print('Its derivative:')
display(sym.diff(fx))


