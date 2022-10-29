#!/usr/bin/env python3
# File: imginfo.py
# Name: D.Saravanan
# Date: 24/05/2021

""" Script to read the image """

from PIL import Image

image = Image.open("/home/saran/Pictures/uluntemp.jpg")
print(image.width, image.height)
print(image.format)
