

# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools # @lru_cache(maxsize=32)

from numbers import Number
import numpy as np
import search
def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)

some_3d_underground_1 = np.array([[[ 0.455,  0.579, -0.54 , -0.995, -0.771],
                                   ],
                                  [[ 0.801,  0.072, -2.183,  0.858, -1.504],
                                   ],
                                  [[ -0.857,  0.309, -1.623,  0.364,  0.097],
                                   ]])

some_2d_underground_1 = np.array([
       [-0.814,  0.637, 1.824, -0.563],
       [ 0.559, -0.234, -0.366,  0.07 ],
       [ 0.175, -0.284,  0.026, -0.316],
       [ 0.212,  0.088,  0.304,  0.604],
       [-1.231, 1.558, -0.467, -0.371]])

state = np.array([[1],[1],[1]])
state2 = np.array([1,1,3,1])
rp1 = np.array(np.arange(len(state)))

# state_3d = np.array([[0,0],[0,1],[1,0],[2,0]])
pay = 0

x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])

print(np.where(np.broadcast_to(np.arange(1, some_3d_underground_1.shape[2] + 1, 1) <= state[:, :, np.newaxis], some_3d_underground_1.shape), some_3d_underground_1, 0))



# print(np.any(state[0:]))
print(np.repeat(rp1[:],state2[:]))
# [1,1,2,2,2,3]
# print(some_2d_underground_1[rows,[0,1,0,1,2,0]])

# print(some_3d_underground_1[2,0])
# print(np.sum(some_2d_underground_1[:],axis=1))

# a = [([1,2],(3,4),(5,6)),
# ((2,2),(3,4),(5,6))]

# b= np.sum(a[0],axis=0)
# print(b)
# print(a[0][0])

# print(some_3d_underground_1.shape)
# len_x = len(some_3d_underground_1)
# len_y = len(some_3d_underground_1[0])
# len_z = len(some_3d_underground_1[0][0])
# state = some_3d_underground_1
# abs_neighbours = np.abs(some_3d_underground_1)
# print(abs_neighbours)
# print(some_3d_underground_1[0][0])
# L = [-2.5]
# B = [-1.3,-0.2,-0.3]
# same = np.full(len(B),L)
# val = []
# val.append(abs(same[:]-B[:]))
# arr = np.array(val)
# print(arr)
# cond = (arr[:]<1)
# print(cond)
# print(np.any(cond))

# threed = [[0, 0], [1, 0], [2, 0]]    
# twod =[0, 1, 2, 3, 4]
# state = [1]
# b = np.array(twod)
# print(np.broadcast_to(state,shape = b.shape))
# print(np.broadcast_to(state,shape = ab.shape))
# # print(ab.ndim)