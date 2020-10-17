import math
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
import pandas as pd
from itertools import chain
from math import e, log
# Construct the investor category and return its forecast of the closing price of the day


def inv_1(pclose,score, v):
    # Institutional Investor Value Forecast Noise Error
    e2 = 0.5
    e2 = random.normal(loc=0.0, scale=e2)
    # Institutional investors' forecast of the speed of price recovery
    d1 = random.uniform(0.2, 0.8)
    # The degree to which institutional investors follow the trend of forecasted prices
    d2 = random.uniform(0.2, 0.8)
    # Calculate the average of N days
    N = random.randint(2, 30)
    ma = np.sum(pclose[-(N+1): -1])/N
    vmean = 0
    for i in range(N):
        a = i+1
        vmean = (pclose[-(a+1)] - ma)**2 + vmean
    mean = vmean/N
    mean = mean**0.5
    e1 = random.normal(loc=0,scale=mean)
    # Discount factor, the closer the day is, the greater the impact on score
    sigma = 0.9
    # Last period return
    r = math.log10(pclose[-1]) - math.log10(pclose[-2])
    sign = 0
    if r > 0:
        sign = 1
    elif r <= 0:
        sign = -1
    score = sign * (pclose[-2] - ma) * r + sigma * score
    f = math.log10(v[-2]) - math.log10(pclose[-2])
    g = 0
    if score > abs(f):
        g = 1
    elif score <= abs(f):
        g = 0
    pred = pclose[-2] + g*(d2*(pclose[-2] - ma)+e1) + (1 - g)*(d1*(v[-2] - pclose[-2]) + e2)
    if pred > 0:
        pass
    else:
        pred = 0

    return [pred, score]


def inv_2(pclose):
    # Trend traders' prediction of the extent to which prices will continue or reverse.
    d3 = random.uniform(-1.5, 1.5)
    # Calculate the average of N days
    if len(pclose) >= 30:
        N = random.randint(2, 31)
    else:
        N = random.randint(1, len(pclose) + 1)
    ma = np.sum(pclose[-N:]) / N
    vmean = 0
    for i in range(N):
        a = i + 1
        vmean = (pclose[-a] - ma) ** 2 + vmean
    mean = vmean / N
    mean = mean**0.5
    e4 = random.normal(loc=0, scale=mean)
    pred = pclose[-1] + d3*(pclose[-1] - ma) + e4
    if pred > 0:
        pass
    else:
        pred = 0
    return pred


def inv_3(pclose):
    # the noise standard predict deviation of noise trader
    e5 = 0.5
    e5 = random.normal(loc=0, scale=e5)
    pred = pclose[-1] + e5
    if pred > 0:
        pass
    else:
        pred = 0
    return pred


# The quotations formed by the three types of traders based on the real-time transaction price.
def giveprice_1(pclose, pred, price):
    callprice = 0
    # Trading signal, 1 is the bid price, -1 is the sell price
    tradesign = 0
    if pred > pclose:
        tradesign = 1
        if pred > price:
            callprice = random.uniform(price,pred)
        else:
            callprice = random.uniform(pclose,pred)
    elif pred <= pclose:
        tradesign = -1
        if pred > price:
            callprice = random.uniform(pred, pclose)
        else:
            callprice = random.uniform(pred, price)

    if callprice > 0:
        pass
    else:
        giveprice_1(pclose, pred, price)

    return [callprice, tradesign]


# Generate initial investor information
def generate_origin_info():
    # Number of three types of investors n_1, n_2, n_3
    n_1 = 100
    n_2 = 100
    n_3 = 100
    # Investor classification list
    list_1 = []
    # List of investor funds
    list_2 = []
    # List of stocks held by investors
    list_3 = []
    # List of wealth owned by investors
    list_4 = []
    # List of method scores used by investors
    list_5 = []
    for i in range(n_1):
        list_1.append(1)
    for i in range(n_2):
        list_1.append(2)
    for i in range(n_3):
        list_1.append(3)

    n = n_2 + n_1 + n_3
    for i in range(n):
        list_2.append(10000)
        list_3.append(100)
        list_4.append(20000)
        list_5.append(0)

    dictionary = {'type': list_1, 'money': list_2, 'volume': list_3, 'fortune': list_4, 'score': list_5}
    primeinfo = pd.DataFrame(dictionary)
    return primeinfo


# Execute one day's trading process
def buy_sold_a_day(investorinfo, pclose, v):
    buy = [[len(investorinfo) + 1, 0, 0]]
    sold = [[len(investorinfo) + 1, 10000, 0]]
    price = []
    price.append(pclose[-1])
    # Let investors enter the market in random order
    rd = []
    for i in range(len(investorinfo)):
        rd.append(i)
    random.shuffle(rd)
    for i in range(len(rd)):
        candidate = investorinfo.iloc[rd[i]]
        # the first type of trader
        if candidate[0] == 1:
            score = candidate[4]
            # Form the quotation and score of the first category of traders.
            respond = inv_1(pclose, score, v)
            # update score
            investorinfo.iloc[rd[i], 4] = respond[1]
            preclose = pclose[-1]
            price_now = price[-1]
            pred_1 = respond[0]
            callprice_1 = giveprice_1(preclose, pred_1, price_now)

            # Determine the bid price and the sell price, 1 is buying, -1 is selling
            if callprice_1[1] == 1:
                # quotation of buying
                money_now = candidate[1]
                if money_now > 0:
                    afa = random.uniform(0.25, 0.85)
                    quant = int(afa * (money_now / callprice_1[0]))
                    q_origin = quant
                    # Final volumn
                    q = 0
                    sold.sort(key=lambda x: x[1])
                    j = 0
                    # Write the buy price to the buy list and update the trading volume after the transaction is completed
                    buy.append([rd[i], callprice_1[0], quant])
                    # Find all seller orders whose selling price is lower than the buying price, and update the open interest every time one is completed.
                    while sold[j][1] <= callprice_1[0]:
                        if sold[j][2] >= quant and quant> 0:
                            sold[j][2] = sold[j][2] - quant
                            q = q + quant
                            if quant > 0:
                                price.append(sold[j][1])
                                investorinfo.iloc[sold[j][0], 1] = investorinfo.copy().iloc[sold[j][0], 1] + quant*sold[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] - quant * sold[j][1]
                                investorinfo.iloc[sold[j][0], 2] = investorinfo.copy().iloc[sold[j][0], 2] - quant
                                quant = 0
                            else:
                                pass
                        elif sold[j][2] < quant and sold[j][2] > 0:
                            quant = quant - sold[j][2]
                            q = q + sold[j][2]
                            if sold[j][2] > 0:
                                price.append(sold[j][1])
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] - sold[j][2] * sold[j][1]
                                investorinfo.iloc[sold[j][0], 1] = investorinfo.copy().iloc[sold[j][0], 1] + sold[j][2] * sold[j][1]
                                investorinfo.iloc[sold[j][0], 2] = investorinfo.copy().iloc[sold[j][0], 2] - sold[j][2]
                            sold[j][2] = 0
                        if j + 1 >= len(sold):
                            break
                        else:
                            j = j + 1
                    buy[-1][2] = q_origin - q
                    investorinfo.iloc[rd[i], 2] = investorinfo.copy().iloc[rd[i], 2] + q
                else:
                    pass

            elif callprice_1[1] == -1:
                if candidate[2] > 0:
                    afa = random.uniform(0.25, 0.85)
                    quant = int(afa * candidate[2])
                    q_origin = quant
                    # Final volumn
                    q = 0
                    buy.sort(key=lambda x: x[1], reverse=True)
                    j = 0
                    # Write the buy price to the buy list and update the trading volume after the transaction is completed
                    sold.append([rd[i], callprice_1[0], quant])
                    # Find all seller orders whose selling price is lower than the buying price, and update the open interest every time one is completed.
                    while buy[j][1] >= callprice_1[0]:
                        if buy[j][2] >= quant and quant > 0:
                            buy[j][2] = buy[j][2] - quant
                            q = q + quant
                            if quant > 0:
                                price.append(buy[j][1])
                                investorinfo.iloc[buy[j][0], 1] = investorinfo.copy().iloc[buy[j][0], 1] - quant * buy[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] + quant * buy[j][1]
                                investorinfo.iloc[buy[j][0], 2] = investorinfo.copy().iloc[buy[j][0], 2] + quant
                                quant = 0
                            else:
                                pass
                        elif buy[j][2] < quant and buy[j][2] > 0:
                            quant = quant - buy[j][2]
                            q = q + buy[j][2]
                            if buy[j][2] > 0:
                                price.append(buy[j][1])
                                investorinfo.iloc[buy[j][0], 1] = investorinfo.copy().iloc[buy[j][0], 1] - buy[j][2] * buy[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] + buy[j][2] * buy[j][1]
                                investorinfo.iloc[buy[j][0], 2] = investorinfo.copy().iloc[buy[j][0], 2] + buy[j][2]
                            else:
                                pass
                            buy[j][2] = 0
                        if j + 1 >= len(buy):
                            break
                        else:
                            j = j + 1
                    sold[-1][2] = q_origin - q
                    investorinfo.iloc[rd[i], 2] = investorinfo.copy().iloc[rd[i], 2] - q
                else:
                    pass

        # the second type of trader
        elif candidate[0] == 2:
            respond = inv_2(pclose)
            # update score
            preclose = pclose[-1]
            price_now = price[-1]
            pred_1 = respond
            callprice_1 = giveprice_1(preclose, pred_1, price_now)

            # Determine the bid price and the sell price, 1 is buying, -1 is selling
            if callprice_1[1] == 1:
                # quotation of buying
                money_now = candidate[1]
                if money_now > 0:
                    afa = random.uniform(0.25, 0.85)
                    quant = int(afa * money_now / callprice_1[0])
                    q_origin = quant
                    # Final volumn
                    q = 0
                    sold.sort(key=lambda x: x[1])
                    j = 0
                    # Write the buy price to the buy list and update the trading volume after the transaction is completed
                    buy.append([rd[i], callprice_1[0], quant])
                    # Find all seller orders whose selling price is lower than the buying price, and update the open interest every time one is completed.
                    while sold[j][1] <= callprice_1[0]:
                        if sold[j][2] >= quant and quant > 0:
                            sold[j][2] = sold[j][2] - quant
                            q = q + quant
                            if quant > 0:
                                price.append(sold[j][1])
                                investorinfo.iloc[sold[j][0], 1] = investorinfo.copy().iloc[sold[j][0], 1] + quant*sold[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] - quant * sold[j][1]
                                investorinfo.iloc[sold[j][0], 2] = investorinfo.copy().iloc[sold[j][0], 2] - quant
                                quant = 0
                            else:
                                pass
                        elif sold[j][2] < quant and sold[j][2] > 0:
                            quant = quant - sold[j][2]
                            q = q + sold[j][2]
                            if sold[j][2] > 0:
                                price.append(sold[j][1])
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] - sold[j][2] * sold[j][1]
                                investorinfo.iloc[sold[j][0], 1] = investorinfo.copy().iloc[sold[j][0], 1] + sold[j][2] * sold[j][1]
                                investorinfo.iloc[sold[j][0], 2] = investorinfo.copy().iloc[sold[j][0], 2] - sold[j][2]
                            sold[j][2] = 0
                        if j + 1 >= len(sold):
                            break
                        else:
                            j = j + 1
                    buy[-1][2] = q_origin - q
                    investorinfo.iloc[rd[i], 2] = investorinfo.copy().iloc[rd[i], 2] + q
                else:
                    pass

            elif callprice_1[1] == -1:
                if candidate[2] > 0:
                    afa = random.uniform(0.25, 0.85)
                    quant = int(afa * candidate[2])
                    q_origin = quant
                    # Final volumn
                    q = 0
                    buy.sort(key=lambda x: x[1], reverse=True)
                    j = 0
                    # Write the selling price into the selling price order and update the trading volume after the transaction is completed
                    sold.append([rd[i], callprice_1[0], quant])
                    # Find all buyer orders whose bid price is higher than the sell price, and update the open interest every time one is completed
                    while buy[j][1] >= callprice_1[0]:
                        if buy[j][2] >= quant and quant > 0:
                            buy[j][2] = buy[j][2] - quant
                            q = q + quant
                            if quant > 0:
                                price.append(buy[j][1])
                                investorinfo.iloc[buy[j][0], 1] = investorinfo.copy().iloc[buy[j][0], 1] - quant * buy[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] + quant * buy[j][1]
                                investorinfo.iloc[buy[j][0], 2] = investorinfo.copy().iloc[buy[j][0], 2] + quant
                                quant = 0
                            else:
                                pass
                        elif buy[j][2] < quant and buy[j][2] > 0:
                            quant = quant - buy[j][2]
                            q = q + buy[j][2]
                            if buy[j][2] > 0:
                                price.append(buy[j][1])
                                investorinfo.iloc[buy[j][0], 1] = investorinfo.copy().iloc[buy[j][0], 1] - buy[j][2] * buy[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] + buy[j][2] * buy[j][1]
                                investorinfo.iloc[buy[j][0], 2] = investorinfo.copy().iloc[buy[j][0], 2] + buy[j][2]
                            else:
                                pass
                            buy[j][2] = 0
                        if j + 1 >= len(buy):
                            break
                        else:
                            j = j + 1
                    sold[-1][2] = q_origin - q
                    investorinfo.iloc[rd[i], 2] = investorinfo.copy().iloc[rd[i], 2] - q
                else:
                    pass

        # the third type of trader
        elif candidate[0] == 3:
            respond = inv_3(pclose)
            preclose = pclose[-1]
            price_now = price[-1]
            pred_1 = respond
            callprice_1 = giveprice_1(preclose, pred_1, price_now)

            # Determine the bid price and the sell price, 1 is buying, -1 is selling
            if callprice_1[1] == 1:
                # quotation of buying
                money_now = candidate[1]
                if money_now > 0:
                    afa = random.uniform(0.25, 0.85)
                    quant = int(afa * money_now / callprice_1[0])
                    q_origin = quant
                    # final volumn
                    q = 0
                    sold.sort(key=lambda x: x[1])
                    j = 0
                    # Write the buy price into the buy order and update the trading volume after the transaction is completed
                    buy.append([rd[i], callprice_1[0], quant])
                    # Find all seller orders whose selling quotation is lower than the buying quotation, and update the open interest for each completed
                    while sold[j][1] <= callprice_1[0]:
                        if sold[j][2] >= quant and quant > 0:
                            sold[j][2] = sold[j][2] - quant
                            q = q + quant
                            if quant > 0:
                                price.append(sold[j][1])
                                investorinfo.iloc[sold[j][0], 1] = investorinfo.copy().iloc[sold[j][0], 1] + quant*sold[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] - quant * sold[j][1]
                                investorinfo.iloc[sold[j][0], 2] = investorinfo.copy().iloc[sold[j][0], 2] - quant
                                quant = 0
                            else:
                                pass
                        elif sold[j][2] < quant and sold[j][2] > 0:
                            quant = quant - sold[j][2]
                            q = q + sold[j][2]
                            if sold[j][2] > 0:
                                price.append(sold[j][1])
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] - sold[j][2] * sold[j][1]
                                investorinfo.iloc[sold[j][0], 1] = investorinfo.copy().iloc[sold[j][0], 1] + sold[j][2] * sold[j][1]
                                investorinfo.iloc[sold[j][0], 2] = investorinfo.copy().iloc[sold[j][0], 2] - sold[j][2]
                            sold[j][2] = 0
                        if j + 1 >= len(sold):
                            break
                        else:
                            j = j + 1
                    buy[-1][2] = q_origin - q
                    investorinfo.iloc[rd[i], 2] = investorinfo.copy().iloc[rd[i], 2] + q
                else:
                    pass

            elif callprice_1[1] == -1:
                if candidate[2] > 0:
                    afa = random.uniform(0.25, 0.85)
                    quant = int(afa * candidate[2])
                    q_origin = quant
                    # Final volume
                    q = 0
                    buy.sort(key=lambda x: x[1], reverse=True)
                    j = 0
                    # Write the selling price into the selling price order and update the trading volume after the transaction is completed
                    sold.append([rd[i], callprice_1[0], quant])
                    # Find all buyer orders whose bid price is higher than the sell price, and update the open interest every time one is completed
                    while buy[j][1] >= callprice_1[0]:
                        if buy[j][2] >= quant and quant > 0:
                            buy[j][2] = buy[j][2] - quant
                            q = q + quant
                            if quant > 0:
                                price.append(buy[j][1])
                                investorinfo.iloc[buy[j][0], 1] = investorinfo.copy().iloc[buy[j][0], 1] - quant * buy[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] + quant * buy[j][1]
                                investorinfo.iloc[buy[j][0], 2] = investorinfo.copy().iloc[buy[j][0], 2] + quant
                                quant = 0
                            else:
                                pass
                        elif buy[j][2] < quant and buy[j][2] > 0:
                            quant = quant - buy[j][2]
                            q = q + buy[j][2]
                            if buy[j][2] > 0:
                                price.append(buy[j][1])
                                investorinfo.iloc[buy[j][0], 1] = investorinfo.copy().iloc[buy[j][0], 1] - buy[j][2] * buy[j][1]
                                investorinfo.iloc[rd[i], 1] = investorinfo.copy().iloc[rd[i], 1] + buy[j][2] * buy[j][1]
                                investorinfo.iloc[buy[j][0], 2] = investorinfo.copy().iloc[buy[j][0], 2] + buy[j][2]
                            else:
                                pass
                            buy[j][2] = 0
                        if j + 1 >= len(buy):
                            break
                        else:
                            j = j + 1
                    sold[-1][2] = q_origin - q
                    investorinfo.iloc[rd[i], 2] = investorinfo.copy().iloc[rd[i], 2] - q
                else:
                    pass




    # Update final wealth and closing price
    rate = price[-1]/pclose[-1]
    pclose = price[-1]
    investorinfo['fortune'] = investorinfo.apply(lambda x: x['money'] + price[-1] * x['volume'], axis=1)

    return investorinfo, pclose, price[1:], rate


# The closing price of the first 100 periods is 100
pclose = [100 for i in range(100)]
# Random walk distribution of value
v = [100]
# Value random walk standard deviation
e3 = 0.05
price = []
# rate of return
rate = []
investorinfo = generate_origin_info()
# Set the number of periods to run:
num = 100
for i in range(num):
    value = v[-1]
    # v is the value of the stock, which conforms to the random walk process and is a random variable
    n = random.normal(loc=0.0, scale=e3)
    value = e**(log(value)+n)
    v.append(value)
    back = buy_sold_a_day(investorinfo, pclose, v)
    investorinfo = back[0]
    pclose.append(back[1])
    price.append(back[2])
    rate.append(back[3])
    print('\rCurrent progress：{:.2f}%'.format((i+1) * 100 / num), end='')
print(investorinfo)
print(pclose)
print(rate)


price = list(chain.from_iterable(price))
print(price)
draw_pclose = pclose[99:]
x = np.linspace(0, num, 1)
plt.xlabel('periods')
plt.ylabel('price')
plt.plot(price)
plt.show()