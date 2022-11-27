#!/usr/bin/env python
# File: matplotlibfont.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Script to find the matplotlib font family """

import os
from matplotlib.font_manager import findfont, FontProperties

for family in ["serif", "sans", "monospace"]:
    font = findfont(FontProperties(family=family))
    print(family, ":" , os.path.basename(font))
