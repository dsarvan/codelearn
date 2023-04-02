#!/usr/bin/env python
# File: threadpoolclass.py
# Name: D.Saravanan
# Date: 02/04/2023

""" Script demonstrates a parallel for-loop with the ThreadPool class """

from multiprocessing.pool import ThreadPool


def task(value):
    """This approach is very effective for
    executing tasks that invole calling the
    same function many times with different
    arguments. The ThreadPool class provides
    many variations of the map() function, such
    as lazy versions and a version that allows
    multiple arguments to the task function."""

    return value


if __name__ == "__main__":
    # create the pool with the default number of workers
    with ThreadPool() as pool:
        # issue one task for each call to the function
        for result in pool.map(task, range(100)):
            # handle the result
            print(f">got {result}")

    # report that all tasks are completed
    print("Done")
