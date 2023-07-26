# Databricks notebook source
!pip install micrograd

# COMMAND ----------

from micrograd.engine import Value

# COMMAND ----------

a = Value(-4.0)
b = Value(2.0)

# COMMAND ----------

c = a + b
# c contains points to nodes a and b
d = a * b + b**3
c += c+1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 /f 

# COMMAND ----------

print(f'{g.data}:.4f') # prints outcome of the forward pass

# COMMAND ----------

# Initialize back propogation at node g
# Goes backwards through expression graph and apply chain rule from calculus. Evaluate derivative of g with respect to all the internal nodes, along with the inputs (a and b)
# Derivative tells us how a and b are affecting g through this mathematical expression. If we sligthly nudge a, then the slope of the growth in a is 138. Slope of growth of b will be 645
g.backward()

# COMMAND ----------

print(f'{a.grad:.4f}') # prints 138, numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645, numerical value of dg/db

# COMMAND ----------

# Neural networks are just mathematical expression just like the expression above which was meaningless. 
# Neural networks take inputs, data and weights. Then, the output is your predictions or loss function. Neural networks are just a class of mathematical expressions. Backpropogation is much more general, it doesn't care about neural networks, only tells us about arbitrary mathematical expressions and we happen to use it to train neural networks

# COMMAND ----------

# Micrograd is a scalar value autograd engine. Works on the level of individual scalars like -4 and 2 from a and b. Breaking neural networks all the way down to the atoms of neural networks, where we do multiplies, adds, etc... Allows us to not deal with n-dimensional tensors yet. Tensors are arrays of these scalars for efficiency/ parallelism

# COMMAND ----------

# Micrograd is what you need to train neural networks and everything else is for efficiency

# COMMAND ----------

# Broken down into engine.py and nn.py. Engine knows nothing about neural nets

# Backpropogation/ autograd engine that gives the power of neural nets is 100 lines of code in engine.py
# nn.py. Define a neuron, layer of neuron, and multi-level perception(which is a layer of neurons). 50 ish lines of code

# COMMAND ----------

# What is a derivative
import math
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

def f(x):
  return 3 * x**2 - 4*x + 5

# COMMAND ----------

f(3.0)

# COMMAND ----------

xs = np.arange(-5, 5, 0.25)
print(xs)
ys = f(xs)
print(ys)

# COMMAND ----------

plt.plot(xs, ys)

# COMMAND ----------

# what is the derivative of this function at any point x?
# take mathematical expression and derive it
# what is derivative telling you about the function?
#if you sligthly increase funcion by number h, how does the function respond? what is the slope? does the function go up or down and by how much?
h = 0.00001
x = 2/3
(f(x+h) - f(x)) / h
# If you add too many zeros then at some point it will converge because of floating point aritmetic

# COMMAND ----------

# Let's get more complicated
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)
# Function of 3 scalar inputs and single output d

# COMMAND ----------

# Want to look at derivatives of d with respsect to a, b, and c

h = 0.0001

#inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a +=h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2-d1)/ h)

# Will d2 be slightly greater or slightly lower?
# The sign of the derivative will tell us
# a is slighly more positive, but b is negative. Because b is -3 will be adding less to d

# COMMAND ----------

d1 = a*b + c
b +=h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2-d1)/ h)

# If we bump b, because b is positive then goes up. 

d1 = a*b + c
c +=h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2-d1)/ h)

# If we bump c, then it is a bit higher by exactly what we added to c. Slope is 1. d will increase as a constant, 1

# COMMAND ----------

# Need data structures to maintain expressions for neural networks. Will build out Value object in micrograd

class Value:
  def __init__(self, data, _children=()):
    self.data = data
    self._prev = set(_children)
    # Need to keep these expression graphs so need to keep pointers about which values produce which other values. So, introduce _children to __init__, then store it as _prev

  def __repr__(self):
      return f"Value(data={self.data})"
  
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other))
    return out
  
  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other))
    return out
    
a = Value(2.0)
b = Value(-3.0)
print(a + b)
c = Value(10.0)
print(a * b)
print((a.__mul__(b)).__add__(c)) 
d = a*b + c
print(d._prev)

# COMMAND ----------

# Need to keep these expression graphs so need to keep pointers about which values produce which other values. So, introduce _children to __init__, then store it as _prev
# When creating a value for addition or multiplication. Will feed in children of value
