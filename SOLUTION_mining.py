#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools  # @lru_cache(maxsize=32)

from numbers import Number

import search

    
def convert_to_tuple(a):
    """
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.

    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    """
    if isinstance(a, Number):
        return a
    if len(a) == 0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    
    
def convert_to_list(a):
    """
    Convert the array-like parameter 'a' into a nested list of the same
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    """
    if isinstance(a, Number):
        return a
    if len(a) == 0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]    


class Mine(search.Problem):
    """

    Mine represent an open mine problem defined by a grid of cells
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.

    The z direction is pointing down, the x and y directions are surface
    directions.

    An instance of a Mine is characterized by
    - self._underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between
                           adjacent columns

    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the
                                         mine

    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.

    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.

    States must be tuple-based.

    """
    
    def __init__(self, underground, dig_tolerance=1):
        """
        Constructor

        Initialize the attributes
        self._underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial

        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        """
        # super().__init__() # call to parent class constructor not needed
        
        self._underground = underground
        # self._underground  should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance
        assert underground.ndim in (2, 3)
        
        if self._underground.ndim == 2:
            self.len_x, self.len_z = self._underground.shape
            self.cumsum_mine = np.cumsum(self._underground, dtype=float, axis=1)
            self.initial = np.zeros(self.len_x, dtype=int)
        elif self._underground.ndim == 3:
            self.len_x, self.len_y, self.len_z = self._underground.shape
            self.cumsum_mine = np.cumsum(self._underground, dtype=float, axis=2)
            self.initial = np.zeros((self.len_x, self.len_y), dtype=int)

    def surface_neigbhours(self, loc):
        """
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        """
        L = []
        assert len(loc) in (1, 2)
        if len(loc) == 1:
            if loc[0]-1 >= 0:
                L.append((loc[0]-1,))
            if loc[0]+1 < self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx, dy in ((-1, -1), (-1, 0), (-1, +1),
                           (0, -1), (0, +1),
                           (+1, -1), (+1, 0), (+1, +1)):
                if (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L
    
    def actions(self, state):
        """
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state :
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        """
        state = np.array(state)

        actions_list = []

        if self._underground.ndim == 2:
            for index in range(state.shape[0]):
                if not state[index] >= self.len_z:
                    state_copy = np.copy(state)
                    state_copy[index] += 1
                    if not self.is_dangerous(state_copy):
                        actions_list.append((index,))

            return actions_list

        else:
            for index_1 in range(state.shape[0]):
                for index_2 in range(state.shape[1]):
                    if not state[index_1][index_2] >= self.len_z:
                        state_copy = np.copy(state)
                        state_copy[index_1][index_2] += 1
                        if not self.is_dangerous(state):
                            actions_list.append((index_1, index_2))

            return actions_list

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        action = tuple(action)
        new_state = np.array(state)  # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)

    def console_display(self):
        """
        Display the mine on the console

        Returns
        -------
        None.

        """
        print('Mine of depth {}'.format(self.len_z))
        if self._underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())
        
    def __str__(self):
        if self._underground.ndim == 2:
            # 2D mine
            return str(self._underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                             + str(self._underground[..., z]) for z in range(self.len_z))

    @staticmethod   
    def plot_state(state):
        if state.ndim == 1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]),
                   state)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim == 2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x)  # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3, 3))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        """
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.

        No loops needed in the implementation!
        """
        # convert to np.array in order to use tuple addressing
        state = np.array(state)
        # state[loc]   where loc is a tuple
        if state.ndim == 1:
            return np.sum(np.where(np.broadcast_to(np.arange(1, self._underground.shape[1] + 1, 1)
                                                   <= state[:, np.newaxis],
                                                   self._underground.shape), self._underground, 0), dtype=float)
        else:
            return np.sum(np.where(np.broadcast_to(np.arange(1, self._underground.shape[2] + 1, 1)
                                                   <= state[:, :, np.newaxis],
                                                   self._underground.shape), self._underground, 0))

    def is_dangerous(self, state):
        """
        Return True iff the given state breaches the dig_tolerance constraints.

        No loops needed in the implementation!
        """
        # convert to np.array in order to use numpy operators
        state = np.array(state)

        if self._underground.ndim == 2:
            return np.any(np.where(np.abs(state[1:] - state[:-1]) > self.dig_tolerance,
                                   np.abs(state[1:] - state[:-1]), 0))

        if self._underground.ndim == 3:
            return np.any(np.where(np.abs(state[:-1, :-1] - state[1:, 1:]) > self.dig_tolerance, True, False)) \
                   or np.any(np.where(np.abs(np.rot90(state)[:-1, :-1] - np.rot90(state)[1:, 1:]) > self.dig_tolerance,
                                      True, False)) or np.any(np.where(np.abs(np.diff(state)) > self.dig_tolerance,
                                                                       True, False)) \
                   or np.any(np.where(np.abs(np.diff(state, axis=0)) > self.dig_tolerance, True, False))

    # ========================  Class Mine  ==================================


def search_dp_dig_plan(mine):
    """
    Search using Dynamic Programming the most profitable sequence of
    digging actions from the initial state of the mine.

    Return the sequence of actions, the final state and the payoff


    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    """
    state_payoff_dict = {}
    initial = convert_to_tuple(mine.initial)
    state_payoff_dict.update({initial: mine.payoff(initial)})

    """
    computes recursively a best solution starting from the state s

    Parameters
    ----------
    s : the state from which the search begins

    Raises
    ------
    AssertionError
        when the given state is of the incorrect dimension
    """
    @functools.lru_cache(maxsize=None)
    def search_rec(s):
        assert np.array(s).ndim == 2 or 3
        # print(s)
        actions = mine.actions(s)
        if not len(actions) == 0:
            for action in actions:
                child_state = mine.result(s, action)
                if child_state not in state_payoff_dict:
                    state_payoff_dict[tuple(child_state)] = mine.payoff(child_state)
                    search_rec(tuple(child_state))

    search_rec(convert_to_tuple(initial))

    best_payoff = max(state_payoff_dict.values())
    best_final_state = dict(map(reversed, state_payoff_dict.items()))[best_payoff]
    best_action_list = find_action_sequence(mine.initial, best_final_state)

    return best_payoff, best_action_list, best_final_state


def search_bb_dig_plan(mine):
    """
    Compute, using Branch and Bound, the most profitable sequence of
    digging actions from the initial state of the mine.


    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    """
    
    assert NotImplementedError


def find_action_sequence(s0, s1):
    """
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.

    Preconditions:
        s0 and s1 are legal states, s0<=s1 and

    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    """
    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]

    s0 = np.array(s0)
    s1 = np.array(s1)
    actions_list = []

    if s0.ndim == 1:
        while np.any(np.where(s0 < s1, True, False)):
            s0_min_index = np.argmin(np.where(s0 < s1, s0, np.max(s1)))
            actions_list.append((s0_min_index, ))
            s0[s0_min_index] += 1

        return actions_list
    elif s0.ndim == 2:
        while np.any(np.where(s0 < s1, True, False)):
            s0_min_index = np.unravel_index((np.argmin(np.where(s0 < s1, s0, np.max(s1) + 1))), s0.shape)
            actions_list.append(s0_min_index)
            s0[s0_min_index] += 1

        return actions_list


def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    """
    return [(10404074, 'Dimithri', 'Young'), (10240977, 'Jun', 'Chen')]
