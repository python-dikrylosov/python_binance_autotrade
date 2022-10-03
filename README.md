---

### :fire: My Stats :
[![GitHub Streak](http://github-readme-streak-stats.herokuapp.com?user=python-dikrylosov&theme=dark&background=000000)](https://git.io/streak-stats)
# python_binance_autotrade
# AXS-BTC / BTC-RUB / ETHBTC 


Регистрация на бинансе : https://accounts.binance.com/ru/register?ref=125688411

Присылаете мне ваш id и если вы правильно зарегистрировались и вписали номер пригласившего 125688411 ,

то я вас проконсультирую по коду что написал
_______________________________________
Страница в вк : 
https://vk.com/python_dikrylosov

Канал создание криптоботов:
https://t.me/python_binance_autotrade

Группа для обсуждения:
https://t.me/+hut1ldc_dF04ODVi
_______________________________________
Список источников:

1 - видео с ютуба https://www.youtube.com/watch?v=QIUxPv5PJOY 

2 - Докунтация по api binance  https://python-binance.readthedocs.io/en/latest/

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/java/java-original-wordmark.svg" title="Java" alt="Java" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/react/react-original-wordmark.svg" title="React" alt="React" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/spring/spring-original-wordmark.svg" title="Spring" alt="Spring" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/materialui/materialui-original.svg" title="Material UI" alt="Material UI" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/flutter/flutter-original.svg" title="Flutter" alt="Flutter" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/redux/redux-original.svg" title="Redux" alt="Redux " width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/css3/css3-plain-wordmark.svg"  title="CSS3" alt="CSS" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/html5/html5-original.svg" title="HTML5" alt="HTML" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/javascript/javascript-original.svg" title="JavaScript" alt="JavaScript" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/firebase/firebase-plain-wordmark.svg" title="Firebase" alt="Firebase" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/gatsby/gatsby-original.svg" title="Gatsby"  alt="Gatsby" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/mysql/mysql-original-wordmark.svg" title="MySQL"  alt="MySQL" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/nodejs/nodejs-original-wordmark.svg" title="NodeJS" alt="NodeJS" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/amazonwebservices/amazonwebservices-plain-wordmark.svg" title="AWS" alt="AWS" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/git/git-original-wordmark.svg" title="Git" **alt="Git" width="40" height="40"/>
</div>

        start_asynhio_BTC_RUB.py
        работа файла
        import time
        import asyncio
        import matplotlib.pyplot as plt
        import pandas as pd
        import cv2
        import numpy
        import password
        api_key = password.binance_api_key
        secret_key = password.binance_secret_key
        from binance.client import Client
        client = Client(api_key, secret_key)

        info = client.get_all_tickers()
        for i in range(0):
                symbol_symbol = info[i]
                symbol_symbol_symbol = symbol_symbol["symbol"]
                symbol_symbol_price = symbol_symbol["price"]
                data = [i,symbol_symbol_symbol, symbol_symbol_price]
                print(data)
                if symbol_symbol_symbol == "AXSUSDT":
                    print([i,symbol_symbol_symbol, symbol_symbol_price])
                    exit()
        #666 BTCRUB, 688 USDTRUB, 673 'BUSDRUB', 615 'BUSDUSDT',
        #11 'BTCUSDT',614, 'BTCBUSD',
        #1137, 'AXSBTC',1139, 'AXSUSDT'


        time_sec = str(time.strftime("%S"))
        time_min = str(time.strftime("%M"))
        time_hour = str(time.strftime("%H"))


        while True:
            time.sleep(0)

            if time_hour != str(time.strftime("%S")):
                time_hour = str(time.strftime("%H"))
                print(["Час прошел, запускаю программу", time_hour, str(time.strftime("%H"))])
                #Курсы валют
                symbol_BTCUSDT = info[11]
                symbol_BTCUSDT_symbol = symbol_BTCUSDT["symbol"]
                symbol_BTCUSDT_price = symbol_BTCUSDT["price"]

                symbol_ETHUSDT = info[12]
                symbol_ETHUSDT_symbol = symbol_ETHUSDT["symbol"]
                symbol_ETHUSDT_price = symbol_ETHUSDT["price"]

                symbol_ETHBTC = info[0]
                symbol_ETHBTC_symbol = symbol_ETHBTC["symbol"]
                symbol_ETHBTC_price = symbol_ETHBTC["price"]

                symbol_BTCRUB = info[666]
                symbol_BTCRUB_symbol = symbol_BTCRUB["symbol"]
                symbol_BTCRUB_price = symbol_BTCRUB["price"]

                symbol_USDTRUB = info[688]
                symbol_USDTRUB_symbol = symbol_USDTRUB["symbol"]
                symbol_USDTRUB_price = symbol_USDTRUB["price"]

                symbol_AXSBTC = info[1137]
                symbol_AXSBTC_symbol = symbol_AXSBTC["symbol"]
                symbol_AXSBTC_price = symbol_AXSBTC["price"]

                symbol_AXSUSDT = info[1139]
                symbol_AXSUSDT_symbol = symbol_AXSUSDT["symbol"]
                symbol_AXSUSDT_price = symbol_AXSUSDT["price"]

                #Балансы валют
                balance_btc = client.get_asset_balance(asset='BTC')
                # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                balance_btc_asset,balance_btc_free,balance_btc_locked = balance_btc["asset"],float(balance_btc["free"]),float(balance_btc["locked"])
                balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
                balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

                balance_ETH = client.get_asset_balance(asset='ETH')
                balance_ETH_asset, balance_ETH_free, balance_ETH_locked = balance_ETH["asset"], float(balance_ETH["free"]), float(balance_ETH["locked"])
                balance_usd_ETH_usd_free_locked_sum = float(balance_ETH["free"]) + float(balance_ETH["locked"])
                balance_usd_ETH_usd_present = balance_usd_ETH_usd_free_locked_sum * float(symbol_ETHUSDT["price"])

                balance_RUB = client.get_asset_balance(asset='RUB')
                balance_RUB_asset, balance_RUB_free, balance_RUB_locked = balance_RUB["asset"], float(balance_RUB["free"]), float(balance_RUB["locked"])
                balance_usd_RUB_usd_free_locked_sum = float(balance_RUB["free"]) + float(balance_RUB["locked"])
                balance_usd_RUB_usd_present = balance_usd_RUB_usd_free_locked_sum * float(symbol_USDTRUB["price"])

                balance_AXS = client.get_asset_balance(asset='AXS')
                balance_AXS_asset, balance_AXS_free, balance_AXS_locked = balance_AXS["asset"], float(balance_AXS["free"]),float(balance_AXS["locked"])
                balance_usd_AXS_usd_free_locked_sum = float(balance_AXS["free"]) + float(balance_AXS["locked"])
                balance_usd_AXS_usd_present = balance_usd_AXS_usd_free_locked_sum * float(symbol_AXSUSDT["price"])

                async def ETHBTCorder():
                    #Сохранение и отрытие фильтрации ордеров в картинки
                    data_read_order_ETHBTC = pd.read_csv("ETHBTCorder.csv")
                    data_read_order_ETHBTC_filter_price_ETHBTC = data_read_order_ETHBTC.filter(["price_ETHBTC"])
                    data_read_order_ETHBTC_filter_price_order = data_read_order_ETHBTC.filter(["price_order"])
                    data_read_order_ETHBTC_filter_origQty = data_read_order_ETHBTC.filter(["origQty"])

                    data_read_order_BTCRUB = pd.read_csv("BTCRUBorder.csv")
                    data_read_order_BTCRUB_filter_price_BTCRUB = data_read_order_BTCRUB.filter(["price_ETHBTC"])
                    data_read_order_BTCRUB_filter_price_order = data_read_order_BTCRUB.filter(["price_order"])
                    data_read_order_BTCRUB_filter_origQty = data_read_order_BTCRUB.filter(["origQty"])


                    figure, ETHBTC = plt.subplots(2,2)

                    ETHBTC[0, 0].plot(data_read_order_ETHBTC_filter_price_ETHBTC,"green")
                    ETHBTC[0, 0].plot(data_read_order_ETHBTC_filter_price_order,"red")
                    ETHBTC[0, 1].plot(data_read_order_BTCRUB_filter_price_BTCRUB, "green")
                    ETHBTC[0, 1].plot(data_read_order_BTCRUB_filter_price_order, "red")
                    ETHBTC[1, 0].bar(data_read_order_ETHBTC_filter_origQty)

                    plt.savefig("ETHBTC_Filter_order_price.png")

                    #img_ETHBTC_Filter_order_price_v = cv2.imread("ETHBTC_Filter_order_price.png")
                    #cv2.imshow("ETHBTC_Filter_order_price", img_ETHBTC_Filter_order_price_v)
                    print(["Открытие картинки order_ETHBTC"])
                    # await asyncio.sleep(1)

                async def BTCRUBorder():
                    #Сохранение и отрытие фильтрации ордеров в картинки
                    data_read_order_BTCRUB = pd.read_csv("BTCRUBorder.csv")
                    data_read_order_BTCRUB_filter_price_BTCRUB = data_read_order_BTCRUB.filter(["price_BTCRUB"])
                    data_read_order_BTCRUB_filter_price_order = data_read_order_BTCRUB.filter(["price_order"])
                    data_read_order_BTCRUB_filter_origQty = data_read_order_BTCRUB.filter(["origQty"])

                    figure, BTCRUB = plt.subplots(2,2)

                    BTCRUB[0, 0].plot(data_read_order_BTCRUB_filter_price_BTCRUB,"green")
                    BTCRUB[0, 0].plot(data_read_order_BTCRUB_filter_price_order,"red")
                    BTCRUB[1, 0].bar(data_read_order_BTCRUB_filter_origQty)

                    plt.savefig("BTCRUB_Filter_order_price.png")

                    #img_BTCRUB_Filter_order_price_v = cv2.imread("BTCRUB_Filter_order_price.png")
                    #cv2.imshow("BTCRUB_Filter_order_price", img_BTCRUB_Filter_order_price_v)
                    print(["Открытие картинки order_ETHBTC"])
                    # await asyncio.sleep(1)

                async def AXSBTCorder():
                    #Сохранение и отрытие фильтрации ордеров в картинки
                    data_read_order_AXSBTC = pd.read_csv("AXSBTCorder.csv")
                    data_read_order_AXSBTC_filter_price_AXSBTC = data_read_order_AXSBTC.filter(["price_AXSBTC"])
                    data_read_order_AXSBTC_filter_price_order = data_read_order_AXSBTC.filter(["price_order"])
                    data_read_order_AXSBTC_filter_origQty = data_read_order_AXSBTC.filter(["origQty"])

                    figure, BTCRUB = plt.subplots(2,2)

                    BTCRUB[0, 0].plot(data_read_order_AXSBTC_filter_price_AXSBTC,"green")
                    BTCRUB[0, 0].plot(data_read_order_AXSBTC_filter_price_order,"red")
                    BTCRUB[1, 0].bar(data_read_order_AXSBTC_filter_origQty)

                    plt.savefig("AXSBTC_Filter_order_price.png")

                    #img_AXSBTC_Filter_order_price_v = cv2.imread("AXSBTC_Filter_order_price.png")
                    #cv2.imshow("AXSBTC_Filter_order_price", img_AXSBTC_Filter_order_price_v)
                    print(["Открытие картинки order_ETHBTC"])
                    # await asyncio.sleep(1)

                async def control_balance_RUB():
                    #Баланс биткоина
                    RUB_v_balance = [balance_RUB_asset,balance_RUB_free,balance_RUB_locked,balance_usd_RUB_usd_present]

                    RUB_v_balance_safe = open("RUB_v_balance.csv","a")
                    RUB_v_balance_safe.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                    RUB_v_balance_safe.write(",")
                    RUB_v_balance_safe.write(str(balance_RUB_free))
                    RUB_v_balance_safe.write(",")
                    RUB_v_balance_safe.write(str(balance_RUB_locked))
                    RUB_v_balance_safe.write(",")
                    RUB_v_balance_safe.write(str(balance_usd_RUB_usd_present))
                    RUB_v_balance_safe.write(",")
                    RUB_v_balance_safe.write(str(symbol_BTCRUB_price))
                    RUB_v_balance_safe.write("\n")
                    RUB_v_balance_safe.close()

                    #print(BTC_v_balance)
                    #await asyncio.sleep(1)
                    #Фильтр баланса биткоина
                    RUB_read = pd.read_csv("RUB_v_balance.csv")
                    RUB_filter_balance_free = RUB_read.filter(["balance_RUB_free"])
                    RUB_filter_balance_locked = RUB_read.filter(["balance_RUB_locked"])
                    RUB_filter_balance_BTCRUB = RUB_read.filter(["balance_usd_RUB_usd_present"])
                    RUB_filter_balance_BTCRUB_price = RUB_read.filter(["symbol_BTCRUB_price"])



                    #print(BTC_read)
                    #Открытие баланса картинки биткоина
                    for i in range(1):
                        fig, axs = plt.subplots(2,2)

                        axs[0, 0].plot(RUB_filter_balance_free)
                        axs[0, 1].plot(RUB_filter_balance_locked)
                        axs[1, 0].plot(RUB_filter_balance_BTCRUB)
                        axs[1, 1].plot(RUB_filter_balance_BTCRUB_price)

                        plt.savefig("RUB_Filter.png")

                    #img_RUB = cv2.imread("RUB_Filter.png")
                    #("RUB", img_RUB)

                    print(["Открытие картинки RUB"])
                    await asyncio.sleep(0)


                async def control_balance_BTC():
                    #Баланс биткоина
                    BTC_v_balance = [balance_btc_asset,balance_btc_free,balance_btc_locked,balance_usd_btc_usd_present]

                    BTC_v_balance_safe = open("BTC_v_balance.csv","a")
                    BTC_v_balance_safe.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                    BTC_v_balance_safe.write(",")
                    BTC_v_balance_safe.write(str(balance_btc_free))
                    BTC_v_balance_safe.write(",")
                    BTC_v_balance_safe.write(str(balance_btc_locked))
                    BTC_v_balance_safe.write(",")
                    BTC_v_balance_safe.write(str(balance_usd_btc_usd_present))
                    BTC_v_balance_safe.write(",")
                    BTC_v_balance_safe.write(str(symbol_BTCUSDT_price))
                    BTC_v_balance_safe.write("\n")
                    BTC_v_balance_safe.close()

                    #print(BTC_v_balance)
                    #await asyncio.sleep(1)
                    #Фильтр баланса биткоина
                    BTC_read = pd.read_csv("BTC_v_balance.csv")
                    BTC_filter_balance_free = BTC_read.filter(["BTC_free"])
                    BTC_filter_balance_locked = BTC_read.filter(["BTC_locked"])
                    BTC_filter_balance_BTCUSDT = BTC_read.filter(["BTC_USDT"])
                    BTC_filter_balance_BTCUSDT_price = BTC_read.filter(["BTCUSDT_price"])



                    #print(BTC_read)
                    #Открытие баланса картинки биткоина
                    for i in range(1):
                        fig, axs = plt.subplots(2,2)

                        axs[0, 0].plot(BTC_filter_balance_free)
                        axs[0, 1].plot(BTC_filter_balance_locked)
                        axs[1, 0].plot(BTC_filter_balance_BTCUSDT)
                        axs[1, 1].plot(BTC_filter_balance_BTCUSDT_price)

                        plt.savefig("BTC_Filter.png")

                    #img_BTC = cv2.imread("BTC_Filter.png")
                    #cv2.imshow("BTC", img_BTC)

                    print(["Открытие картинки BTC"])
                    await asyncio.sleep(0)


                async def control_balance_ETH():
                    #Баланс етериума
                    ETH_v_balance = [balance_ETH_asset, balance_ETH_free, balance_ETH_locked, balance_usd_ETH_usd_present]

                    ETH_v_balance_safe = open("ETHBTC.csv", "a")
                    ETH_v_balance_safe.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                    ETH_v_balance_safe.write(",")
                    ETH_v_balance_safe.write(str(symbol_ETHBTC_price))
                    ETH_v_balance_safe.write(",")
                    ETH_v_balance_safe.write(str(balance_ETH_locked))
                    ETH_v_balance_safe.write(",")
                    ETH_v_balance_safe.write(str(symbol_ETHUSDT_price))
                    ETH_v_balance_safe.write(",")
                    ETH_v_balance_safe.write(str(balance_usd_ETH_usd_present))
                    ETH_v_balance_safe.write(",")
                    ETH_v_balance_safe.write(str(balance_ETH_free))
                    ETH_v_balance_safe.write("\n")
                    ETH_v_balance_safe.close()

                    #print(ETH_v_balance)
                    #await asyncio.sleep(1)
                    #Фильтрация етериума
                    ETH_read = pd.read_csv("ETHBTC.csv")
                    ETH_filter_balance_free = ETH_read.filter(["ETH_free"])
                    ETH_filter_balance_locked = ETH_read.filter(["ETH_locked"])
                    ETH_filter_balance_ETHUSDT = ETH_read.filter(["ETH_USDT"])
                    ETH_filter_balance_price_ETHBTC = ETH_read.filter(["price_ETHBTC"])
                    ETH_filter_balance_ETHUSDT_price = ETH_read.filter(["price_ETHUSDT"])

                    #print(ETH_read)
                    #Открытие картинки етериума
                    for i in range(1):
                        fig, axs = plt.subplots(2, 3)

                        axs[0, 0].plot(ETH_filter_balance_free)
                        axs[0, 1].plot(ETH_filter_balance_locked)
                        axs[1, 0].plot(ETH_filter_balance_ETHUSDT)
                        axs[1, 1].plot(ETH_filter_balance_price_ETHBTC)
                        axs[1, 2].plot(ETH_filter_balance_ETHUSDT_price)

                        plt.savefig("ETH_Filter.png")

                    #img_ETH = cv2.imread("ETH_Filter.png")
                    #cv2.imshow("ETH", img_ETH)
                    print(["Открытие картинки ETH"])
                    await asyncio.sleep(0)

                async def control_balance_AXS():
                    info = client.get_all_tickers()
                    symbol_AXSBTC = info[1137]
                    symbol_AXSBTC_symbol = symbol_AXSBTC["symbol"]
                    symbol_AXSBTC_price = symbol_AXSBTC["price"]

                    symbol_AXSUSDT = info[1139]
                    symbol_AXSUSDT_symbol = symbol_AXSUSDT["symbol"]
                    symbol_AXSUSDT_price = symbol_AXSUSDT["price"]

                    balance_AXS = client.get_asset_balance(asset='AXS')
                    balance_AXS_asset, balance_AXS_free, balance_AXS_locked = balance_AXS["asset"], float(balance_AXS["free"]), float(balance_AXS["locked"])
                    balance_usd_AXS_usd_free_locked_sum = float(balance_AXS["free"]) + float(balance_AXS["locked"])
                    balance_usd_AXS_usd_present = balance_usd_AXS_usd_free_locked_sum * float(symbol_AXSUSDT["price"])
                    #Баланс етериума
                    AXS_v_balance = [balance_AXS_asset, balance_AXS_free, balance_AXS_locked, balance_usd_AXS_usd_present]

                    AXS_v_balance_safe = open("AXSBTC.csv", "a")
                    AXS_v_balance_safe.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                    AXS_v_balance_safe.write(",")
                    AXS_v_balance_safe.write(str(symbol_AXSBTC_price))
                    AXS_v_balance_safe.write(",")
                    AXS_v_balance_safe.write(str(balance_AXS_locked))
                    AXS_v_balance_safe.write(",")
                    AXS_v_balance_safe.write(str(symbol_AXSUSDT_price))
                    AXS_v_balance_safe.write(",")
                    AXS_v_balance_safe.write(str(balance_usd_AXS_usd_present))
                    AXS_v_balance_safe.write(",")
                    AXS_v_balance_safe.write(str(balance_AXS_free))
                    AXS_v_balance_safe.write("\n")
                    AXS_v_balance_safe.close()

                    #print(ETH_v_balance)
                    #await asyncio.sleep(1)
                    #Фильтрация етериума
                    AXS_read = pd.read_csv("AXSBTC.csv")
                    AXS_filter_balance_free = AXS_read.filter(["AXS_free"])
                    AXS_filter_balance_locked = AXS_read.filter(["AXS_locked"])
                    AXS_filter_balance_AXSUSDT = AXS_read.filter(["AXS_USDT"])
                    AXS_filter_balance_price_AXSBTC = AXS_read.filter(["price_AXSBTC"])
                    AXS_filter_balance_AXSUSDT_price = AXS_read.filter(["price_AXSUSDT"])

                    #print(ETH_read)
                    #Открытие картинки етериума
                    for i in range(1):
                        fig, AXSBTC = plt.subplots(2, 3)

                        AXSBTC[0, 0].plot(AXS_filter_balance_free)
                        AXSBTC[0, 1].plot(AXS_filter_balance_locked)
                        AXSBTC[1, 0].plot(AXS_filter_balance_AXSUSDT)
                        AXSBTC[1, 1].plot(AXS_filter_balance_price_AXSBTC)
                        AXSBTC[1, 2].plot(AXS_filter_balance_AXSUSDT_price)

                        plt.savefig("AXSBTC_Filter.png")

                    #img_AXSBTC = cv2.imread("AXSBTC_Filter.png")
                    #cv2.imshow("AXSBTC", img_AXSBTC)
                    print(["Открытие картинки AXSBTC"])
                    await asyncio.sleep(0)

                async def start_ETHBTC():
                    #Курс валюты ETHBTC
                    info = client.get_all_tickers()
                    symbol_ETHBTC = info[0]
                    symbol_ETHBTC_symbol = symbol_ETHBTC["symbol"]
                    symbol_ETHBTC_price = symbol_ETHBTC["price"]

                    balance_btc = client.get_asset_balance(asset='BTC')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
                    balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

                    balance_ETH = client.get_asset_balance(asset='ETH')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_ETH_usd_free_locked_sum = float(balance_ETH["free"]) + float(balance_ETH["locked"])
                    balance_usd_ETH_usd_present = balance_usd_ETH_usd_free_locked_sum * float(symbol_ETHUSDT["price"])

                    #Запуск нейронки
                    import os
                    import math
                    import numpy as np
                    from sklearn.preprocessing import MinMaxScaler
                    from tensorflow.keras.models import load_model
                    from tensorflow.keras.models import save_model
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, LSTM
                    import matplotlib.pyplot as plt
                    start_time = time.time()



                    data_read_pandas_ETHBTC = pd.read_csv("ETHBTC.csv")
                    # data_read_pandas_ETHBTC = data_read_pandas_ETHBTC.tail(500)
                    data_read_pandas_ETHBTC_shape_row, data_read_pandas_ETHBTC_shape_col = data_read_pandas_ETHBTC.shape[0], \
                                                                                           data_read_pandas_ETHBTC.shape[1]
                    print(data_read_pandas_ETHBTC.shape)
                    print([data_read_pandas_ETHBTC_shape_row, data_read_pandas_ETHBTC_shape_col])

                    filter_ETHBTC_price = data_read_pandas_ETHBTC.filter(["price_ETHBTC"])

                    print(filter_ETHBTC_price)

                    # create dATEFRAME CLOSE
                    data = data_read_pandas_ETHBTC.filter(["price_ETHBTC"])

                    # data_df_pandas_filter = data_df_pandas.filter(["Well"])
                    print(data)

                    # convert dataframe
                    dataset = data.values

                    # dataset  = data_df_pandas_filter.values
                    print(dataset)

                    # get the number rows to train the model
                    training_data_len = math.ceil(len(dataset) * .8)
                    print(training_data_len)
                    # scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(dataset)
                    print(scaled_data)
                    plt.plot(scaled_data)
                    plt.savefig("scaled_data_ETHBTC.png")

                    # create the training dataset
                    train_data = scaled_data[0:training_data_len, :]
                    # split the data into x_train and y_train data sets
                    x_train = []
                    y_train = []
                    for rar in range(60, len(train_data)):
                        x_train.append(train_data[rar - 60:rar, 0])
                        y_train.append(train_data[rar, 0])
                        if rar <= 61:
                            print(x_train)
                            print(y_train)
                            print()
                    # conver the x_train and y_train to numpy arrays
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    # reshape the data
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    print(x_train.shape)
                    import tensorflow as tf

                    # biuld to LST model

                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(101, return_sequences=False))
                    model.add(Dense(50))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    # cmopale th emodel
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    # train_the_model
                    model.summary()
                    print("Fit model on training data")

                    # Evaluate the model on the test data using `evaluate`
                    print("Evaluate on test data")
                    results = model.evaluate(x_train, y_train, batch_size=1)
                    print("test loss, test acc:", results)

                    model = tf.keras.models.load_model(os.path.join("./dnn/", "ETHBTC_model.h5"))
                    model.fit(x_train, y_train, batch_size=1, epochs=1)

                    model.save(os.path.join("./dnn/", "ETHBTC_model.h5"))
                    # reconstructed_model = tf.keras.models.load_model(os.path.join("./dnn/", "BTC-RUB_model.h5"))

                    # np.testing.assert_allclose(model.predict(x_train), reconstructed_model.predict(x_train))
                    # reconstructed_model.fit(x_train, y_train)

                    # create the testing data set
                    # create a new array containing scaled values from index 1713 to 2216
                    test_data = scaled_data[training_data_len - 60:, :]
                    # create the fata sets x_test and y_test
                    x_test = []
                    y_test = dataset[training_data_len:, :]
                    for resr in range(60, len(test_data)):
                        x_test.append(test_data[resr - 60:resr, 0])

                    # conert the data to numpy array
                    x_test = np.array(x_test)

                    # reshape the data
                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                    # get the model predicted price values
                    predictions = model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)

                    # get the root squared error (RMSE)
                    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
                    print(rmse)

                    # get the quate

                    new_df = data_read_pandas_ETHBTC.filter(["price_ETHBTC"])

                    # get teh last 60 days closing price values and convert the dataframe to an array
                    last_60_days = new_df[-60:].values
                    # scale the data to be values beatwet 0 and 1

                    last_60_days_scaled = scaler.transform(last_60_days)

                    # creAte an enemy list
                    X_test = []
                    # Append past 60 days
                    X_test.append(last_60_days_scaled)

                    # convert the x tesst dataset to numpy
                    X_test = np.array(X_test)

                    # Reshape the dataframe
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    # get predict scaled

                    pred_price = model.predict(X_test)
                    # undo the scaling
                    pred_price = scaler.inverse_transform(pred_price)
                    print(pred_price)

                    pred_price_a = pred_price[0]
                    pred_price_aa = pred_price_a[0]
                    preset_pred_price = round(pred_price_aa, 6)
                    print(pred_price)
                    print(preset_pred_price)
                    old_time = time.time() - start_time
                    print("Время на расчеты :" + str(old_time))
                    min_lovume_btc = 0.00011

                    time.sleep(5)



                    # pred_price = float(symbol_BTCRUB_price) + float(random.randint(-1000,1000))
                    #Запуск ордера
                    if preset_pred_price <= float(symbol_ETHBTC_price):
                        info = client.get_all_tickers()
                        symbol_ETHBTC = info[0]
                        a = float(1)
                        b = float(balance_btc["free"])
                        ab_sum = a * b
                        data_coin = float(ab_sum) - min_lovume_btc
                        print(data_coin)
                        quantity = float(min_lovume_btc / float(symbol_ETHBTC_price))
                        print(quantity)
                        if data_coin <= 0:
                            print([data_coin, a, b])
                            print(ab_sum)
                            quantity = float(min_lovume_btc / float(symbol_ETHBTC_price))
                            print(quantity)
                            print("Недостаточно  btc")
                        elif data_coin >= 0:
                            print([data_coin, a, b])
                            print("\n" + "SELL Покупать btc  " + str(preset_pred_price))
                            print(a)
                            quantity = float(min_lovume_btc / float(symbol_ETHBTC_price))
                            quantity_start = round(quantity, 4)
                            print(quantity_start)
                            order = client.order_limit_buy(symbol='ETHBTC', quantity=quantity_start, price=preset_pred_price)
                            print(order)
                            data_safe_file_ETHBTC = open("ETHBTCorder.csv", "a")
                            data_safe_file_ETHBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(symbol_ETHBTC_price))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['symbol']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['orderId']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['transactTime']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['price']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['origQty']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['side']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write("\n")
                            data_safe_file_ETHBTC.close()



                    elif preset_pred_price >= float(symbol_ETHBTC_price):
                        info = client.get_all_tickers()
                        symbol_ETHBTC = info[0]
                        a = float(symbol_ETHBTC_price)
                        b = float(balance_ETH["free"])
                        ab_sum = a * b
                        data_coin = float(ab_sum) - min_lovume_btc
                        print(data_coin)
                        quantity = float(min_lovume_btc / float(symbol_ETHBTC_price))
                        print(quantity)
                        if data_coin <= 0:
                            print([data_coin, a, b])
                            print(ab_sum)
                            quantity = float(min_lovume_btc / float(symbol_ETHBTC_price))
                            print(quantity)
                            print("Недостаточно ETH для продажи")
                        elif data_coin >= 0:
                            print([data_coin, a, b])
                            print("\n" + "BUY Покупать за ETH " + str(preset_pred_price))
                            print(a)
                            quantity = float(min_lovume_btc / float(symbol_ETHBTC_price))
                            quantity_start = round(quantity, 4)
                            print(quantity_start)
                            order = client.order_limit_sell(symbol='ETHBTC', quantity=quantity_start, price=preset_pred_price)
                            print(order)
                            data_safe_file_ETHBTC = open("ETHBTCorder.csv", "a")
                            data_safe_file_ETHBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(symbol_ETHBTC_price))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['symbol']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['orderId']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['transactTime']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['price']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['origQty']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write(str(order['side']))
                            data_safe_file_ETHBTC.write(",")
                            data_safe_file_ETHBTC.write("\n")
                            data_safe_file_ETHBTC.close()


                    #await asyncio.sleep(1)

                    #await asyncio.sleep(1)
                #Запуск обновления окошка картинок

                async def start_BTCRUB():
                    #Курс валюты ETHBTC
                    info = client.get_all_tickers()
                    symbol_BTCRUB = info[666]
                    symbol_BTCRUB_symbol = symbol_BTCRUB["symbol"]
                    symbol_BTCRUB_price = symbol_BTCRUB["price"]

                    symbol_USDTRUB = info[688]
                    symbol_USDTRUB_symbol = symbol_USDTRUB["symbol"]
                    symbol_USDTRUB_price = symbol_USDTRUB["price"]

                    balance_btc = client.get_asset_balance(asset='BTC')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
                    balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

                    balance_ETH = client.get_asset_balance(asset='ETH')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_ETH_usd_free_locked_sum = float(balance_ETH["free"]) + float(balance_ETH["locked"])
                    balance_usd_ETH_usd_present = balance_usd_ETH_usd_free_locked_sum * float(symbol_ETHUSDT["price"])

                    balance_RUB = client.get_asset_balance(asset='RUB')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_RUB_usd_free_locked_sum = float(balance_RUB["free"]) + float(balance_RUB["locked"])
                    balance_usd_RUB_usd_present = balance_usd_RUB_usd_free_locked_sum * float(symbol_USDTRUB["price"])

                    #Запуск нейронки
                    import os
                    import math
                    import numpy as np
                    from sklearn.preprocessing import MinMaxScaler
                    from tensorflow.keras.models import load_model
                    from tensorflow.keras.models import save_model
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, LSTM
                    import matplotlib.pyplot as plt
                    start_time = time.time()



                    data_read_pandas_BTCRUB = pd.read_csv("RUB_v_balance.csv")
                    # data_read_pandas_ETHBTC = data_read_pandas_ETHBTC.tail(500)
                    data_read_pandas_BTCRUB_shape_row, data_read_pandas_BTCRUB_shape_col = data_read_pandas_BTCRUB.shape[0], \
                                                                                           data_read_pandas_BTCRUB.shape[1]
                    print(data_read_pandas_BTCRUB.shape)
                    print([data_read_pandas_BTCRUB_shape_row, data_read_pandas_BTCRUB_shape_col])

                    filter_BTCRUB_price = data_read_pandas_BTCRUB.filter(["symbol_BTCRUB_price"])

                    print(filter_BTCRUB_price)

                    # create dATEFRAME CLOSE
                    data = data_read_pandas_BTCRUB.filter(["symbol_BTCRUB_price"])

                    # data_df_pandas_filter = data_df_pandas.filter(["Well"])
                    print(data)

                    # convert dataframe
                    dataset = data.values

                    # dataset  = data_df_pandas_filter.values
                    print(dataset)

                    # get the number rows to train the model
                    training_data_len = math.ceil(len(dataset) * .8)
                    print(training_data_len)
                    # scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(dataset)
                    print(scaled_data)
                    plt.plot(scaled_data)
                    plt.savefig("scaled_data_BTCRUB.png")

                    # create the training dataset
                    train_data = scaled_data[0:training_data_len, :]
                    # split the data into x_train and y_train data sets
                    x_train = []
                    y_train = []
                    for rar in range(60, len(train_data)):
                        x_train.append(train_data[rar - 60:rar, 0])
                        y_train.append(train_data[rar, 0])
                        if rar <= 61:
                            print(x_train)
                            print(y_train)
                            print()
                    # conver the x_train and y_train to numpy arrays
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    # reshape the data
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    print(x_train.shape)
                    import tensorflow as tf

                    # biuld to LST model

                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(101, return_sequences=False))
                    model.add(Dense(50))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    # cmopale th emodel
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    # train_the_model
                    model.summary()
                    print("Fit model on training data")

                    # Evaluate the model on the test data using `evaluate`
                    print("Evaluate on test data")
                    results = model.evaluate(x_train, y_train, batch_size=1)
                    print("test loss, test acc:", results)

                    model = tf.keras.models.load_model(os.path.join("./dnn/", "BTCRUB_model.h5"))
                    model.fit(x_train, y_train, batch_size=1, epochs=1)

                    model.save(os.path.join("./dnn/", "BTCRUB_model.h5"))
                    # reconstructed_model = tf.keras.models.load_model(os.path.join("./dnn/", "BTC-RUB_model.h5"))

                    # np.testing.assert_allclose(model.predict(x_train), reconstructed_model.predict(x_train))
                    # reconstructed_model.fit(x_train, y_train)

                    # create the testing data set
                    # create a new array containing scaled values from index 1713 to 2216
                    test_data = scaled_data[training_data_len - 60:, :]
                    # create the fata sets x_test and y_test
                    x_test = []
                    y_test = dataset[training_data_len:, :]
                    for resr in range(60, len(test_data)):
                        x_test.append(test_data[resr - 60:resr, 0])

                    # conert the data to numpy array
                    x_test = np.array(x_test)

                    # reshape the data
                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                    # get the model predicted price values
                    predictions = model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)

                    # get the root squared error (RMSE)
                    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
                    print(rmse)

                    # get the quate

                    new_df = data_read_pandas_BTCRUB.filter(["symbol_BTCRUB_price"])

                    # get teh last 60 days closing price values and convert the dataframe to an array
                    last_60_days = new_df[-60:].values
                    # scale the data to be values beatwet 0 and 1

                    last_60_days_scaled = scaler.transform(last_60_days)

                    # creAte an enemy list
                    X_test = []
                    # Append past 60 days
                    X_test.append(last_60_days_scaled)

                    # convert the x tesst dataset to numpy
                    X_test = np.array(X_test)

                    # Reshape the dataframe
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    # get predict scaled

                    pred_price = model.predict(X_test)
                    # undo the scaling
                    pred_price = scaler.inverse_transform(pred_price)
                    print(pred_price)

                    pred_price_a = pred_price[0]
                    pred_price_aa = pred_price_a[0]
                    preset_pred_price = round(pred_price_aa, 6)
                    print(pred_price)
                    print(preset_pred_price)
                    old_time = time.time() - start_time
                    print("Время на расчеты :" + str(old_time))
                    min_lovume_btc = 0.00011

                    time.sleep(5)



                    # pred_price = float(symbol_BTCRUB_price) + float(random.randint(-1000,1000))
                    #Запуск ордера
                    if preset_pred_price <= float(symbol_BTCRUB_price):
                        info = client.get_all_tickers()
                        symbol_BTCRUB = info[666]
                        a = float(1)
                        b = float(balance_btc["free"])
                        ab_sum = a * b
                        data_coin = float(ab_sum) - min_lovume_btc
                        print(data_coin)
                        quantity = float(min_lovume_btc / float(symbol_BTCRUB_price))
                        print(quantity)
                        if data_coin <= 0:
                            print([data_coin, a, b])
                            print(ab_sum)
                            quantity = float(min_lovume_btc / float(symbol_BTCRUB_price))
                            print(quantity)
                            print("Недостаточно  btc")
                        elif data_coin >= 0:
                            print([data_coin, a, b])
                            print("\n" + "SELL Покупать btc  " + str(preset_pred_price))
                            print(a)
                            quantity = float(min_lovume_btc / float(symbol_BTCRUB_price))
                            quantity_start = round(quantity, 5)
                            print(quantity_start)
                            order = client.order_limit_buy(symbol='BTCRUB', quantity=quantity_start, price=preset_pred_price)
                            print(order)
                            data_safe_file_RUBBTC = open("BTCRUBorder.csv", "a")
                            data_safe_file_RUBBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(symbol_BTCRUB_price))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['symbol']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['orderId']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['transactTime']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['price']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['origQty']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['side']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write("\n")
                            data_safe_file_RUBBTC.close()



                    elif preset_pred_price >= float(symbol_BTCRUB_price):
                        info = client.get_all_tickers()
                        symbol_BTCRUB = info[666]
                        a = float(symbol_BTCRUB_price)
                        b = float(balance_RUB["free"])
                        ab_sum = a * b
                        data_coin = float(ab_sum) - min_lovume_btc
                        print(data_coin)
                        quantity = float(min_lovume_btc / float(symbol_BTCRUB_price))
                        print(quantity)
                        if data_coin <= 0:
                            print([data_coin, a, b])
                            print(ab_sum)
                            quantity = float(min_lovume_btc / float(symbol_BTCRUB_price))
                            print(quantity)
                            print("Недостаточно RUB для продажи")
                        elif data_coin >= 0:
                            print([data_coin, a, b])
                            print("\n" + "BUY Покупать за RUB " + str(preset_pred_price))
                            print(a)
                            quantity = float(min_lovume_btc / float(symbol_BTCRUB_price))
                            quantity_start = round(quantity, 5)
                            print(quantity_start)
                            order = client.order_limit_sell(symbol='BTCRUB', quantity=quantity_start, price=preset_pred_price)
                            print(order)
                            data_safe_file_RUBBTC = open("BTCRUBorder.csv", "a")
                            data_safe_file_RUBBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(symbol_BTCRUB_price))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['symbol']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['orderId']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['transactTime']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['price']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['origQty']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write(str(order['side']))
                            data_safe_file_RUBBTC.write(",")
                            data_safe_file_RUBBTC.write("\n")
                            data_safe_file_RUBBTC.close()


                    #await asyncio.sleep(1)

                    #await asyncio.sleep(1)
                #Запуск обновления окошка картинок

                async def start_AXSBTC():
                    #Курс валюты ETHBTC
                    info = client.get_all_tickers()
                    symbol_AXSBTC = info[1137]
                    symbol_AXSBTC_symbol = symbol_AXSBTC["symbol"]
                    symbol_AXSBTC_price = symbol_AXSBTC["price"]

                    symbol_AXSUSDT = info[1139]
                    symbol_AXSUSDT_symbol = symbol_AXSUSDT["symbol"]
                    symbol_AXSUSDT_price = symbol_AXSUSDT["price"]

                    balance_btc = client.get_asset_balance(asset='BTC')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
                    balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

                    balance_ETH = client.get_asset_balance(asset='ETH')
                    # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                    balance_usd_ETH_usd_free_locked_sum = float(balance_ETH["free"]) + float(balance_ETH["locked"])
                    balance_usd_ETH_usd_present = balance_usd_ETH_usd_free_locked_sum * float(symbol_ETHUSDT["price"])

                    balance_AXS = client.get_asset_balance(asset='AXS')
                    balance_AXS_asset, balance_AXS_free, balance_AXS_locked = balance_AXS["asset"], float(balance_AXS["free"]), float(balance_AXS["locked"])
                    balance_usd_AXS_usd_free_locked_sum = float(balance_AXS["free"]) + float(balance_AXS["locked"])
                    balance_usd_AXS_usd_present = balance_usd_AXS_usd_free_locked_sum * float(symbol_AXSUSDT["price"])

                    #Запуск нейронки
                    import os
                    import math
                    import numpy as np
                    from sklearn.preprocessing import MinMaxScaler
                    from tensorflow.keras.models import load_model
                    from tensorflow.keras.models import save_model
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, LSTM
                    import matplotlib.pyplot as plt
                    start_time = time.time()


                    data_read_pandas_AXSBTC = pd.read_csv("AXSBTC.csv")
                    # data_read_pandas_ETHBTC = data_read_pandas_ETHBTC.tail(500)
                    data_read_pandas_AXSBTC_shape_row, data_read_pandas_AXSBTC_shape_col = data_read_pandas_AXSBTC.shape[0], \
                                                                                           data_read_pandas_AXSBTC.shape[1]
                    print(data_read_pandas_AXSBTC.shape)
                    print([data_read_pandas_AXSBTC_shape_row, data_read_pandas_AXSBTC_shape_col])

                    filter_AXSBTC_price = data_read_pandas_AXSBTC.filter(["price_AXSBTC"])

                    print(filter_AXSBTC_price)

                    # create dATEFRAME CLOSE
                    data = data_read_pandas_AXSBTC.filter(["price_AXSBTC"])

                    # data_df_pandas_filter = data_df_pandas.filter(["Well"])
                    print(data)

                    # convert dataframe
                    dataset = data.values

                    # dataset  = data_df_pandas_filter.values
                    print(dataset)

                    # get the number rows to train the model
                    training_data_len = math.ceil(len(dataset) * .8)
                    print(training_data_len)
                    # scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(dataset)
                    print(scaled_data)
                    plt.plot(scaled_data)
                    plt.savefig("scaled_data_AXSBTC.png")

                    # create the training dataset
                    train_data = scaled_data[0:training_data_len, :]
                    # split the data into x_train and y_train data sets
                    x_train = []
                    y_train = []
                    for rar in range(60, len(train_data)):
                        x_train.append(train_data[rar - 60:rar, 0])
                        y_train.append(train_data[rar, 0])
                        if rar <= 61:
                            print(x_train)
                            print(y_train)
                            print()
                    # conver the x_train and y_train to numpy arrays
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    # reshape the data
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    print(x_train.shape)
                    import tensorflow as tf

                    # biuld to LST model

                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(101, return_sequences=False))
                    model.add(Dense(50))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    # cmopale th emodel
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    # train_the_model
                    model.summary()
                    print("Fit model on training data")

                    # Evaluate the model on the test data using `evaluate`
                    print("Evaluate on test data")
                    results = model.evaluate(x_train, y_train, batch_size=1)
                    print("test loss, test acc:", results)

                    model = tf.keras.models.load_model(os.path.join("./dnn/", "AXSBTC_model.h5"))
                    model.fit(x_train, y_train, batch_size=1, epochs=1)

                    model.save(os.path.join("./dnn/", "AXSBTC_model.h5"))
                    # reconstructed_model = tf.keras.models.load_model(os.path.join("./dnn/", "BTC-RUB_model.h5"))

                    # np.testing.assert_allclose(model.predict(x_train), reconstructed_model.predict(x_train))
                    # reconstructed_model.fit(x_train, y_train)

                    # create the testing data set
                    # create a new array containing scaled values from index 1713 to 2216
                    test_data = scaled_data[training_data_len - 60:, :]
                    # create the fata sets x_test and y_test
                    x_test = []
                    y_test = dataset[training_data_len:, :]
                    for resr in range(60, len(test_data)):
                        x_test.append(test_data[resr - 60:resr, 0])

                    # conert the data to numpy array
                    x_test = np.array(x_test)

                    # reshape the data
                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                    # get the model predicted price values
                    predictions = model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)

                    # get the root squared error (RMSE)
                    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
                    print(rmse)

                    # get the quate

                    new_df = data_read_pandas_AXSBTC.filter(["price_AXSBTC"])

                    # get teh last 60 days closing price values and convert the dataframe to an array
                    last_60_days = new_df[-60:].values
                    # scale the data to be values beatwet 0 and 1

                    last_60_days_scaled = scaler.transform(last_60_days)

                    # creAte an enemy list
                    X_test = []
                    # Append past 60 days
                    X_test.append(last_60_days_scaled)

                    # convert the x tesst dataset to numpy
                    X_test = np.array(X_test)

                    # Reshape the dataframe
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    # get predict scaled

                    pred_price = model.predict(X_test)
                    # undo the scaling
                    pred_price = scaler.inverse_transform(pred_price)
                    print(pred_price)

                    pred_price_a = pred_price[0]
                    pred_price_aa = pred_price_a[0]
                    preset_pred_price = round(pred_price_aa, 6)
                    print(pred_price)
                    print(preset_pred_price)
                    old_time = time.time() - start_time
                    print("Время на расчеты :" + str(old_time))
                    min_lovume_btc = 0.00011

                    time.sleep(5)



                    # pred_price = float(symbol_BTCRUB_price) + float(random.randint(-1000,1000))
                    #Запуск ордера
                    if preset_pred_price <= float(symbol_AXSBTC_price):
                        info = client.get_all_tickers()
                        symbol_AXSBTC = info[1137]
                        symbol_AXSBTC_price = symbol_AXSBTC["price"]
                        a = float(1)
                        b = float(balance_btc["free"])
                        ab_sum = a * b
                        data_coin = float(ab_sum) - min_lovume_btc
                        print(data_coin)
                        quantity = float(min_lovume_btc / float(symbol_AXSBTC_price))
                        print(quantity)
                        if data_coin <= 0:
                            print([data_coin, a, b])
                            print(ab_sum)
                            quantity = float(min_lovume_btc / float(symbol_AXSBTC_price))
                            print(quantity)
                            print("Недостаточно  btc")
                        elif data_coin >= 0:
                            print([data_coin, a, b])
                            print("\n" + "SELL Покупать btc  " + str(preset_pred_price))
                            print(a)
                            quantity = float(min_lovume_btc / float(symbol_AXSBTC_price))
                            quantity_start = round(quantity, 2)
                            print(quantity_start)
                            order = client.order_limit_buy(symbol='AXSBTC', quantity=quantity_start, price=preset_pred_price)
                            print(order)
                            data_safe_file_AXSBTC = open("AXSBTCorder.csv", "a")
                            data_safe_file_AXSBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(symbol_AXSBTC_price))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['symbol']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['orderId']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['transactTime']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['price']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['origQty']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['side']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write("\n")
                            data_safe_file_AXSBTC.close()



                    elif preset_pred_price >= float(symbol_AXSBTC_price):
                        info = client.get_all_tickers()
                        symbol_AXSBTC = info[1137]
                        symbol_AXSBTC_price = symbol_AXSBTC["price"]
                        a = float(symbol_AXSBTC_price)
                        b = float(balance_AXS["free"])
                        ab_sum = a * b
                        data_coin = float(ab_sum) - min_lovume_btc
                        print(data_coin)
                        quantity = float(min_lovume_btc / float(symbol_AXSBTC_price))
                        print(quantity)
                        if data_coin <= 0:
                            print([data_coin, a, b])
                            print(ab_sum)
                            quantity = float(min_lovume_btc / float(symbol_AXSBTC_price))
                            print(quantity)
                            print("Недостаточно AXS для продажи")
                        elif data_coin >= 0:
                            print([data_coin, a, b])
                            print("\n" + "BUY Покупать за AXS " + str(preset_pred_price))
                            print(a)
                            quantity = float(min_lovume_btc / float(symbol_AXSBTC_price))
                            quantity_start = round(quantity, 2)
                            print(quantity_start)
                            order = client.order_limit_sell(symbol='AXSBTC', quantity=quantity_start, price=preset_pred_price)
                            print(order)
                            data_safe_file_AXSBTC = open("AXSBTCorder.csv", "a")
                            data_safe_file_AXSBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(symbol_AXSBTC_price))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['symbol']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['orderId']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['transactTime']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['price']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['origQty']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write(str(order['side']))
                            data_safe_file_AXSBTC.write(",")
                            data_safe_file_AXSBTC.write("\n")
                            data_safe_file_AXSBTC.close()


                    #await asyncio.sleep(1)

                    #await asyncio.sleep(1)
                #Запуск обновления окошка картинок


                #Запуск асинхронной программы
                ioloop = asyncio.get_event_loop()
                tasks = [ioloop.create_task(control_balance_AXS()),ioloop.create_task(start_AXSBTC()),ioloop.create_task(AXSBTCorder()),
                         ioloop.create_task(control_balance_BTC()),ioloop.create_task(start_BTCRUB()),ioloop.create_task(BTCRUBorder()),
                         ioloop.create_task(control_balance_ETH()),ioloop.create_task(start_ETHBTC()),ioloop.create_task(ETHBTCorder()),
                         ioloop.create_task(control_balance_RUB())] #
                wait_tasks = asyncio.wait(tasks)
                ioloop.run_until_complete(wait_tasks)
                ioloop.close()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

            elif time_hour == str(time.strftime("%H")):
                for i in range(5):
                    time.sleep(1)
                    print(["pause", i])
                print(["равно h", time_hour, str(time.strftime("%H"))])
                #закрытие асинхронной программы
        

<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>
<div id="badges">
  <a href="your-linkedin-URL">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="your-youtube-URL">
    <img src="https://img.shields.io/badge/YouTube-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Youtube Badge"/>
  </a>
  <a href="your-twitter-URL">
    <img src="https://img.shields.io/badge/Twitter-blue?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter Badge"/>
  </a>
</div>
<img src="https://komarev.com/ghpvc/?username=your-github-username&style=flat-square&color=blue" alt=""/>
