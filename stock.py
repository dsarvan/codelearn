#!/usr/bin/env python3
# File: stock.py
# Date: 06/10/2020

from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

def sprice(p, d, m, s):
    
    data = []
    for d in range(d):
        prob = stats.norm.rvs(loc=m, scale=s)
        price = (p*prob)
        data.append(price)
        p = price

    return data

if __name__ == '__main__':

    stk_price, days, mu, sigma = 20, 200, 1.001, 0.005

    x = 0
    while x < 100:
        data = sprice(stk_price, days, mu, sigma)
        x += 1

        plt.plot(data)

    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xlim(0, 200)
    plt.title('Stock closing price')
    plt.savefig('stockprize.png'); plt.show()
