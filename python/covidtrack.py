#!/usr/bin/env python
# File: covidtrack.py
# Name: D.Saravanan
# Date: 18/05/2024

""" Script to parse html from worldometer covid-19 page """

import datetime
from urllib import request

import bs4


def covidtrack(url: str) -> None:

    page = request.urlopen(url).read().decode("utf8")
    soup = bs4.BeautifulSoup(page, "html.parser")
    data = soup.find_all("div", id="maincounter-wrap")

    for info in data:
        print(f"{info.h1.string} {info.span.string}")


def main() -> None:

    print(datetime.datetime.now().strftime("%d %B %Y @ %H:%M"))

    url = "https://www.worldometers.info/coronavirus/"
    covidtrack(url)


if __name__ == "__main__":
    main()
