#!/usr/bin/env python
# coding: utf-8

# # COURSE: A deep understanding of deep learning
# ## SECTION: Introduction to Python: data types
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/dudl/?couponCode=202201

# # .
# # .
# # .

# # VIDEO: Variables

# In[ ]:


# This is a comment. Python will ignore this text.


# In[ ]:


a = 4
b = 3


# In[ ]:


a


# In[ ]:


a
b


# In[ ]:


print(a) # comments after a line
print(b)


# In[ ]:


c = 7
d = 7.0

print(c,d)


# In[ ]:


type(c)
type(d)


# In[ ]:


c = 'hello'
d = "Hey there!"


# In[ ]:


# define variables and parameters
aVariable    = 10
filterOrder  = 2048
user_name    = 'mike'
param4modelA = 42.42

not!allowed = 3
also#not = 3
^nope! = 3

# variable naming rule: no non-alphanumeric characters except _
# variables may contain numbers but not start with them


# In[ ]:


get_ipython().run_line_magic('whos', '')


# In[ ]:


# multiple assignment
varA,varB = 3,'test'

print(varA)
print(varB)


# ### Exercise:

# In[ ]:


# discover whether it's possible in Python to over-write variables:
#  1) within-type (e.g., numeric to a different number)
#  2) across-type (e.g., numeric to a string)


# # .
# # .
# # .

# # VIDEO: Math operators

# In[ ]:


# let's start by creating some variables
x = 7
y = 4.1
z = 0


# In[ ]:


# Addition and subtraction

print( x-7 )

c = y + x - 2


# In[ ]:


# multiplication and division
print( x*y ) # note: floating-point algebra is not the same thing as "real" algebra
4/3
4  +3


# In[ ]:


# powers
2**3


# In[ ]:


print( 9**1/2 )

# therefore, use parentheses to overwrite order of operations
print( 9**(1/2) )


# In[ ]:


# can we "add" strings??
firstName = 'Mike'
lastName  = 'Cohen'

print( firstName + lastName )
print( firstName + ' ' + lastName )


# In[ ]:


# just curious...
firstName - lastName


# In[ ]:


firstName * 3


# In[ ]:


# printing mixed variable types
print( '7 times 4.1 equals 28.7' )

print( x ' times ' y ' equals ' x*y )
print( x + ' times ' + y + ' equals ' + x*y )


# In[ ]:


print( str(x) + ' times ' + str(y) + ' equals ' + str(x*y) )


# In[ ]:


# inputting data from the user
ans = input('Feed me a stray cat ')
print(ans)
print(type(ans))


# In[ ]:


ans*10


# In[ ]:


numans = input( 'Input a number: ' )
numans = float(numans)

type(numans)


# ### Exercise

# In[ ]:


# The goal is to apply the Pythagorean theorem.
#  Input from the user two lengths, and you return the third length
side1 = int( input('Length of side a: ') )
side2 = int( input('Length of side b: ') )

thirdside = (side1**2 + side2**2)**(1/2)

print(' ')
print( 'The length of side c is ' + str( thirdside ) )


# # .
# # .
# # .

# # VIDEO: Lists

# In[ ]:


aList = [0,1,2,3,4,5]

aList


# In[ ]:


strList = [ 'hi','my','name','is','Mike' ]
strList


# In[ ]:


mixList = [ 3,'three',4,'four' ]
mixList


# In[ ]:


listList = [ 3,['3','4','5'],5,[4,5,6] ]
listList


# In[ ]:


listList = [ 
            3,              # a number
            ['3','h','5'],  # a list of strings
            5,              # another number
            [4,5,6]         # a list of numbers
           ]

listList


# In[ ]:


# test whether an item is in a list
4 in aList


# In[ ]:


print( 4 in aList )

print( 14 in aList )
print( 14 not in aList )


# In[ ]:


aList + strList


# In[ ]:


aList*3


# In[ ]:


newlist = [4,3,4,5,6,7,7,7,7]

set(newlist)


# In[ ]:


## some list methods 

# show the original
print(aList)

# add a new element
aList.append(-100)
print(aList)

# sort
aList.sort()
print(aList)


# ### Exercise

# In[ ]:


# Use a list method to find the number of 7's in newlist
newlist.count(7)


# # .
# # .
# # .

# # VIDEO: Lists part 2

# In[ ]:


# append, insert, remove, del, sort

lst = [1,2,3,4,5]

print(lst)
lst.append(7)
print(lst)


# In[ ]:


lst.insert(2,20)
print(lst)


# In[ ]:


lst.append(3)
print(lst)
lst.remove(3)
print(lst)


# In[ ]:


print(lst)

lst.sort()
print(lst)

lst.sort(reverse=True)
print(lst)


# In[ ]:


lst.append('hi')
lst.sort()


# # VIDEO: Tuples

# In[ ]:


atuple = ( 3,'4',3 )

atuple*3


# In[ ]:


atuple.count(3)


# In[ ]:


alist = list(atuple)

print(atuple)
print(alist)


# In[ ]:


# tuples are immutable
atuple[0] = 5


# In[ ]:


# lists are mutable
alist[0] = 5
alist


# # .
# # .
# # .

# # VIDEO: Booleans

# In[ ]:


boolTrue = True
booltrue = 'true'

get_ipython().run_line_magic('whos', '')


# In[ ]:


# comparisons
4 == 4


# In[ ]:


4>5


# In[ ]:


4>4


# In[ ]:


4>=4


# In[ ]:


4 != 5


# In[ ]:


a = 8
b = 4

outcome = a == b*2
print(outcome)


# In[ ]:


# conjunctive comparisons
14>5 and 7<10


# In[ ]:


14>5 or 7>10


# In[ ]:


# converting to boolean
print( bool(0) )
print( bool(10) )

print(' ')

print( bool('asdf') )
print( bool(' ') )
print( bool('') )


# ### Exercise

# In[ ]:


# Ask users to input Pythagorean triplet, check whether it's real


# In[ ]:


print("Let's test your knowledge of Pythagorean triples!")

sidea = float(input('Input length of side a: '))
sideb = float(input('Input length of side b: '))
sidec = float(input('Input length of side c: '))

# check
isPythTriplet = sidea**2 + sideb**2 == sidec**2

print('Your answer is ' + str(isPythTriplet) '!!')


# # .
# # .
# # .

# # VIDEO: Dictionaries

# In[ ]:


D = dict()
D


# In[ ]:


D['name'] = 'Mike' # key/value pairs
D['AgeRange'] = [20,50]

D


# In[ ]:


D = {'name':'Mike' , 'AgeRange':[20,50]}

D


# In[ ]:


# retrieve one item
D['name']


# In[ ]:


# list all properties
D.keys()


# In[ ]:


# test whether a key is in the dictionary
'name' in D


# In[ ]:


'Mike' in D


# In[ ]:


'Mike' not in D


# In[ ]:


D.values()


# In[ ]:


D.items()


# ### Exercise

# In[ ]:


# Input two numbers from the user. 
# Create a dictionary with keys 'firstNum', 'secondNum', and 'product'


# In[ ]:


del D
D = dict()

num1 = float(input('Give me a number '))
D['FirstNum'] = num1

num2 = float(input('Give me another number '))
D['SecondNum'] = num2

D['Product'] = num1*num2

D.items()


# In[ ]:




