#!/usr/bin/env python3
# File: scraper.py
# Name: D.Saravanan
# Date: 19/01/2021
# Script to scrape a webpage

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

page = requests.get("https://locations.familydollar.com/id/")
page.encoding = 'ISO-885901'
soup = BeautifulSoup(page.text, 'html.parser')
#print(soup.prettify())

addrList = soup.find_all(class_='itemlist')
for n in addrList[:2]:
    print(n)

print(type(addrList))
print(len(addrList))
