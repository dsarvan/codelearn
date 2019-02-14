#!/usr/bin/env python3
# File: subsites.py
# Name: D.Saravanan
# Date: 09/06/2021

""" Script to get all subsites of a website """

import json

import requests
from bs4 import BeautifulSoup as bs

sites = ["https://www.opensource.com"]


def base_site(site):
    ind = site.find("/", 7)
    if ind == -1:
        return site + "/"
    return site[:ind]


def enqueue(queue, hrefs):
    queue += hrefs


def get_hrefs(url, site):
    res = requests.get(url)
    html = bs(res.content, "html.parser")
    atags = html.find_all("a", href=True)
    hrefs = [tag["href"] for tag in atags]

    lshrefs = []

    for href in hrefs:

        if "#" in href:
            continue

        if site in href:
            lshrefs.append(href)

        elif not "http://" in href and not "https://" in href:

            if href[0] == "/":
                lshrefs.append(base_site(site) + href)

            elif url[-1] == "/":
                lshrefs.append(url + href)

            else:
                lshrefs.append(url + "/" + href)

        return lshrefs


if __name__ == "__main__":

    json_sites = {}

    for site in sites:
        visited = []
        queue = []
        queue.append(site)

        while True:

            if not queue:
                break

            url = queue.pop(0)

            try:
                if not url in visited:
                    urls = get_hrefs(url, site)
                    enqueue(queue, urls)
                    visited.append(url)
                    print(url)

            except:
                pass

        json_sites[site] = visited

    with open("subsites.json", "w") as outfile:
        json.dump(json_sites, outfile)

    print("-----completed-----")
