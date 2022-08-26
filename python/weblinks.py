#!/usr/bin/env python3
# File: weblinks.py
# Name: D.Saravanan
# Date: 17/01/2021

""" Script to get the list of links from a website """

import argparse
import re
import requests

parser = argparse.ArgumentParser(description="Get list of links from a website")

parser.add_argument("url", nargs="?", help="URL", default=None)
arguments = parser.parse_args()
USE_ARGUMENTS = True if arguments.url is not None else False

if USE_ARGUMENTS:
    url = arguments.url
else:
    while True:
        url = input("Enter URL: ")

        if url == "":
            print("Invalid URL")
            continue
        break

html = requests.get(url).text
links = re.findall('"(https?://.*?)"', html)

for link in links:
    print(link)
