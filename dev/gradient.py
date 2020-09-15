
import numpy as np
import matplotlib.pyplot as plt

#create simple gradient

x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

z = np.outer(x,y)

#now, create an arbitrary rotation of this gradient

def grad(theta, x,y):
    x = np.cos(theta)*x+np.sin(theta)*y
    y = np.cos(theta)*y-np.sin(theta)*x
    zz = x*y
    zz = (zz-zz.min())
    zz = zz/zz.max()
    return zz

def rotate_g(theta):
    zz = np.outer(x*np.cos(theta)+y*np.sin(theta),y*np.cos(theta)-y*np.sin(theta))
    return zz

def rotate(theta, x_vec):
    r = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    y = np.matmul(r,x_vec)
    return y



