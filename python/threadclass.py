#!/usr/bin/env python
# File: threadclass.py
# Name: D.Saravanan
# Date: 02/03/2023

""" Script demonstrates a parallel for-loop using the Thread class """

from threading import Thread


# execute a task
def task(value):
    """This approach is effective for a small number
    of tasks that all need to be run at once. It is
    less effective if we have many more tasks than we
    can support concurrently because all of the tasks
    will run at the same time and could slow each other
    down. It also does not allow results from tasks to
    be returned easily."""

    print(f" .done {value}")


if __name__ == "__main__":
    # create all tasks
    threads = [Thread(target=task, args=(i,)) for i in range(20)]

    # start all threads
    for thread in threads:
        thread.start()

    # wait for all threads to complete
    for thread in threads:
        thread.join()

    # report that all tasks are completed
    print("Done")
