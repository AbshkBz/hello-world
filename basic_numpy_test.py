'''
Tests elementary numpy operations along with matplotlib plots.
All numpy methods are preceded by a comment explaining the method and succeeded by an empty line for greater coherence
'''
## import dependencies
import numpy as np
import matplotlib.pyplot as plt
from math import pow

### ------------------------------------------------------------------------- ###

# initialize a 1-D numpy array
# a = np.array([1,2,3,4,5,6])

# initialize a 2-D numpy array
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])

# create a numpy array with al elements zero
# a = np.zeros(4)

# create a numpy array with all elements equal to 1
# a = np.ones(4)

# create an empty array (initial content is random and depends on the state of the memory -- faster than other array initialization methods)
# a = np.empty(4)

# numpy.arange(n) generates a ndarray with elements 0 till n-1
# a = np.arange(6)

# numpy.linspace(start,stop,num=interval) creates an array with values between start and stop spaced linearly in a specified interval. Unlike numpy.arrange, the array runs from start till stop and not till stop-1
# a = np.linspace(0,10,num=10)

# default data type is floating point however one can explicitly specify data type
# a = np.empty(4, dtype=np.int64)

# sorting an array in numpy with numpy.sorting
# a = np.sort(a)

# concatenate two numpy array's
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# c = np.concatenate((a,b))

# concatenating along a specified axis
# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# c = np.concatenate((a,b),axis=1) # by default axis 0 is row and axis 1 column

# to find the number of dimensions in a numpy array (returns number of rows)
# print(c.ndim)

# to find the number of elements in an array
# print(c.size)

# to find the shape of an array
# print(c.shape)

# reshape an array (total no. of elements should be same)
# c = c.reshape(8,1)

# convert a 1-D array into a 2-D array .. this snippet specifically converts (n,) into (1,n)
# a = np.array([1,2,3,4,5])
# a = a[np.newaxis, :]
# print(a.shape)

# expand_dims to add an axis at a certain index position axis=0 converts to (1,n) whereas axis=1 converts to (n,1)
# a = np.array([1,2,3,4,5,6])
# a = np.expand_dims(a, axis=0)
# print(a,type(a),a.shape)

# slicing a numpy array to reveal contents
# a = np.array([1,2,3,4,5])
# print(a[-1:])

# geneate array elements satisfying specific conditions
# a = np.array([1,2,3,4,5,6,7,8,9,10])
# print(a[a<5])
# c = a[(a>2)&(a<8)]
# print(c)
# print(a[a%2==0]) # array elements divisible by 2 or even numbers
# c = a[(a>2)|(a<4)]
# print(c)

# numpy vstack / hstack to stack arrays together
# a = np.array([1,2,3,4])
# b = np.array([5,6,7,8])
# c = np.vstack((a,b))
# d = np.hstack((a,b))
# print(c,c.shape,d,d.shape)

# numpy hsplit to split array into several simialr arrays
# a = np.arange(1,25).reshape(2,12)
# b = np.hsplit(a,(3,5)) # splits at specific columns
# print(b)

# addition / subtraction of numpy arrays
# a = np.array([1,2,3])
# b = np.arange(1,4)
# c = a-b
# print(a,b,c)

# numpy array sum of all elements
a = np.array([[1,1],[2,2]])
print(a,a.sum(axis=1))

# numpy broadcasting is the scalar multiplication of each array element with a scalar individually
# a = np.array([1,5])
# b = a * 1.6
# print(b)

# aggregation functions on numpy array
# a = np.array([1,2,5,4,7,8,9])
# print(a.max(),a.min(),a.mean(),a.std(),a.prod())

# aggregation functions on numpy matrices
# A = np.ones((3,2))
# print(A.max(),A.min())

# generate random matrices using numpy
# rnd = np.random.default_rng()
# l = rnd.random((3,2))
# print(l,type(l))

# generate numpy array where each element is a random integer between a specified interval
# rnd = np.random.default_rng(0)
# A = rnd.integers(7, size=(3,4))
# print(A)

# get unique values in a numpy array
# a = np.array([1,2,2,4,8,7,7,6,5])
# print(np.unique(a))
# unique_values, indices = np.unique(a, return_index=True) # get indices of unique values in a numpy array
# print(indices)

# transpose numpy matrix
# A = np.array([[1,2,3],[4,5,6]])
# print(A,'\n',A.shape,'\n',A.transpose(),'\n',A.transpose().shape)

# reverse contents of an array
# a = np.arange(8).reshape(2,4)
# print(a,'\n',np.flip(a, axis=1))

# flatten a numpy array
# rnd = np.random.default_rng(0)
# A = rnd.random((2,2))
# print(A,'\n',A.flatten()) # flatten does not change the original array

# save numpy variables to a specific file
# a = np.array([1,4,5,7,8,9,6,3,5,4,2,4,7])
# np.save('Abhishek',a) # saved as .npy file
# load the saved file
# b = np.load('Abhishek.npy')
# print(b)

# to convert adrray into a python list, use tolist() method
# a = a.tolist()

# access elements inside numpy array
# print(a,'\n',type(a))     # for 1-D array
# print(a[0][2])  # for 2-D array

# plot some numpy arrays using matplotlib
# a = np.array([1,2,3,4,5,6])
# b = np.arange(6)
# for i in range(0,len(a)):
    # b[i] = pow(a[i],3)

# plt.figure()
# plt.plot(a,b)
# plt.title('Random stuff')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.grid(True)
# plt.show()


# get a nice 3-D plot using matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-5,5, 0.15)
# Y = np.arange(-5,5, 0.15)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap='viridis')
# plt.show()

### -------------------------------------------------------------------------- ###
