#!/usr/bin/env python3
# File: chaotic.py
# Name: D.Saravanan
# Date: 09/07/2021

""" Script to demonstrate chaos in a simple dynamical system """

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 10          # Acceleration due to gravity, m.s-2
R = 1           # Circle radius, m
dt = 0.001      # Time step, s

def solve(u0):
    """ Solve the equation of motion for a ball bouncing in a cirle.
    u0 = [x0, vx0, y0, vy0] are the initial conditions (position and velocity. """

    # initial time, final time, s
    t0, tf = 0, 10

    def fun(t, u):
        """ Return the derivaties of the dynamics variables acked into u. """
        x, xdot, y, ydot = u
        xdot = 0
        ydot = -g
        return xdot, xddot, ydot, yddot

        
    



