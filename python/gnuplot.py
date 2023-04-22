#!/usr/bin/env python
# File: gnuplot.py
# Name: D.Saravanan
# Date: 22/04/2023

""" Script that implement python interface to gnuplot """

import os


class gnuplot:
    def __init__(self):
        print("opening new gnuplot session...")
        self.session = os.popen("gnuplot", "w")

    def __del__(self):
        print("closing gnuplot session...")
        self.session.close()

    def send(self, cmd):
        self.session.write(cmd + "\n")
        self.session.flush()


if __name__ == "__main__":

    print("Single-window output:")
    g = gnuplot()
    g.send("plot sin(x)")
    g.send("replot cos(x)")
    raw_input("press ENTER to continue")
    del g

    print("Multiple window output:")
    g1 = gnuplot()
    g2 = gnuplot()
    g1.send("plot sin(x)")
    g2.send("plot cos(x)")
    raw_input("press ENTER to continue")
    del g1
    del g2
