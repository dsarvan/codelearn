#!/usr/bin/env python3
#coding=utf8
from pygnuplot import gnuplot

g = gnuplot.Gnuplot(terminal = 'pngcairo font "arial,10" fontscale 1.0 size\
        600, 400', output = '"graphplot.png"')

g.plot('[-10:10] sin(x)', 'atan(x)', 'cos(atan(x))', 
        key = 'fixed left top vertical Right noreverse enhanced autotitle\
        box lt black linewidth', 
        style = 'increment default',
        samples = '50, 50', 
        title = '"Simple Plots" font",2-" norotate')
