#!/usr/bin/env python

import os

os.chdir("/home/saran/codelearn/python/")
file = open("zenpython.txt", 'r')
text = file.read()
print(text, end="")
file.close()
