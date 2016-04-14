#!/usr/bin/env python
# functioning as an abstract to both tensorflow and theano

#import theano_backend_lite as KTH
#import tensorflow_backend_lite as KTF
import os, sys
from common import _FLOATX, _EPSILON

PLF = None # platform

def get_platform():
    return PLF

def set_platform(platform):
  assert platform in ('theano', 'tensorflow'), 'The platform is neither tensorflow nor theano!'
  KT = None
  if platform == 'theano':
    import theano_backend_lite as KT
  else:
    import tensorflow_backend_lite as KT
    #sess = KT.tf.Session()
    #KT.set_session(sess)
  global PLF
  PLF = KT

# Variable manipulation

def variable(value, dtype=_FLOATX, name=None):
  if PLF:
    return PLF.variable(value, dtype=dtype, name=name)
  else:
    raise TypeError("Please set the platform first, choose theano or tensorflow")


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
  if PLF:
    return PLF.placeholder(shape=shape, ndim=ndim, dtype=dtype, name=name)
  else:
    raise TypeError("Please set the platform first, choose theano or tensorflow")


def shape(x):
  return x.shape

def ndim(x):
  return x.ndim

def eval(x):
  return x.eval()

# Value manipulation


def get_value(x):
  if not hasattr(x, 'get_value'):
    raise Exception("get_value() can only be called on a variable.")
  return x.get_value()

def set_value(x, value):
  x.set_value(np.asarray(value, dtype=x.dtype))


# Function manipulation

def function(inputs, outputs, updates=[], **kwargs):
  if PLF:
    return PLF.function(inputs, outputs, updates=updates, **kwargs)
  else:
    raise TypeError("Please set the platform first, choose theano or tensorflow")


# Control flow

def rnn(step_function, inputs, initial_states,go_backwards=False, mask=None, constants=None):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.rnn(step_function, inputs, initial_states,go_backwards=backwards, mask=mask, constants=constants)

def switch(condition, then_expression, else_expression):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.switch(condition, then_expression, else_expression)


# NN operations

def relu(x, alpha=0., max_value=None):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.relu(x, alpha=alpha, max_value=max_value)

def softmax(x):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.softmax(x)

def softplus(x):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.softplus(x)

def sigmoid(x):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.sigmoid(x)

def tanh(x):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.tanh(x)

def l2_normalize(x, axis):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.l2_normalize(x, axis)

# additioanl functions

def log(x):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.log(x)

def exp(x):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.exp(x)

def dot(x, y):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.dot(x,y)

def grad(loss, variables):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.gradients(loss, variables)

def mean(x, axis=None, keepdims=False):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.mean(x, axis=axis, keepdims=keepdims)

def sum(x, axis=None, keepdims=False):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.sum(x, axis=axis, keepdims=keepdims)

def argmax(x, axis=-1):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.argmax(x, axis=axis)

def argmin(x, axis=-1):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.argmin(x, axis=axis)

def minimum(x, y):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.minimum(x, y)

def maximum(x, y):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.maximum(x, y)


#TODO: implement min, max, prod, std

def cast(x, dtype):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.cast(x, dtype)


def eq(x,y):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.equal(x, y)

def neq(x,y):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.not_equal(x, y)


def arange(start=0, limit=None, step=1):
  if not PLF:
    raise TypeError("Please set the platform first, choose theano or tensorflow")
  return PLF.arange(start=start, limit=limit, step=step)









