import password


import time
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy

from binance.client import Client
api_key = password.binance_api_key
secret_key = password.binance_secret_key
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
    start_time = time.time()

    if time_hour != str(time.strftime("%H")):
        time_hour = str(time.strftime("%H"))
        print(["Час прошел, запускаю программу", time_hour, str(time.strftime("%H"))])
        min_lovume_btc = 0.00011
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

        symbol_ETHRUB = info[667]
        symbol_ETHRUB_symbol = symbol_ETHRUB["symbol"]
        symbol_ETHRUB_price = symbol_ETHRUB["price"]

        symbol_LTCBTC = info[1]
        symbol_LTCBTC_symbol = symbol_LTCBTC["symbol"]
        symbol_LTCBTC_price = symbol_LTCBTC["price"]

        symbol_LTCRUB = info[1205]
        symbol_LTCRUB_symbol = symbol_LTCRUB["symbol"]
        symbol_LTCRUB_price = symbol_LTCRUB["price"]

        symbol_BNBBTC = info[2]
        symbol_BNBBTC_symbol = symbol_BNBBTC["symbol"]
        symbol_BNBBTC_price = symbol_BNBBTC["price"]

        symbol_BNBRUB = info[669]
        symbol_BNBRUB_symbol = symbol_BNBRUB["symbol"]
        symbol_BNBRUB_price = symbol_BNBRUB["price"]


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
        balance_usd_RUB_usd_present = balance_usd_RUB_usd_free_locked_sum / float(symbol_USDTRUB["price"])

        balance_AXS = client.get_asset_balance(asset='AXS')
        balance_AXS_asset, balance_AXS_free, balance_AXS_locked = balance_AXS["asset"], float(balance_AXS["free"]),float(balance_AXS["locked"])
        balance_usd_AXS_usd_free_locked_sum = float(balance_AXS["free"]) + float(balance_AXS["locked"])
        balance_usd_AXS_usd_present = balance_usd_AXS_usd_free_locked_sum * float(symbol_AXSUSDT["price"])



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

            figure, AXSBTCs = plt.subplots(2,2)

            AXSBTCs[0, 0].plot(data_read_order_AXSBTC_filter_price_AXSBTC,"green")
            AXSBTCs[0, 0].plot(data_read_order_AXSBTC_filter_price_order,"red")
            AXSBTCs[1, 0].bar(data_read_order_AXSBTC_filter_origQty)

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
            RUB_v_balance_safe.write(str(symbol_BTCRUB_price))
            RUB_v_balance_safe.write(",")
            RUB_v_balance_safe.write(str(balance_RUB_free))
            RUB_v_balance_safe.write(",")
            RUB_v_balance_safe.write(str(balance_RUB_locked))
            RUB_v_balance_safe.write(",")
            RUB_v_balance_safe.write(str(balance_usd_RUB_usd_present))
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
            import matplotlib.pyplot as plt
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

            # Сохранение и отрытие фильтрации ордеров в картинки
            data_read_order_ETHBTC = pd.read_csv("ETHBTCorder.csv")
            data_read_order_ETHBTC_filter_price_ETHBTC = data_read_order_ETHBTC.filter(["price_ETHBTC"])
            data_read_order_ETHBTC_filter_price_order = data_read_order_ETHBTC.filter(["price_order"])
            data_read_order_ETHBTC_filter_origQty = data_read_order_ETHBTC.filter(["origQty"])

            data_read_order_BTCRUB = pd.read_csv("BTCRUBorder.csv")
            data_read_order_BTCRUB_filter_price_BTCRUB = data_read_order_BTCRUB.filter(["price_BTCRUB"])
            data_read_order_BTCRUB_filter_price_order = data_read_order_BTCRUB.filter(["price_order"])
            data_read_order_BTCRUB_filter_origQty = data_read_order_BTCRUB.filter(["origQty"])

            figure, ETHBTC = plt.subplots(2, 2)

            ETHBTC[0, 0].plot(data_read_order_ETHBTC_filter_price_ETHBTC, "green")
            ETHBTC[0, 0].plot(data_read_order_ETHBTC_filter_price_order, "red")
            ETHBTC[0, 1].plot(data_read_order_BTCRUB_filter_price_BTCRUB, "green")
            ETHBTC[0, 1].plot(data_read_order_BTCRUB_filter_price_order, "red")
            ETHBTC[1, 0].bar(data_read_order_ETHBTC_filter_origQty)

            plt.savefig("ETHBTC_Filter_order_price.png")

            # img_ETHBTC_Filter_order_price_v = cv2.imread("ETHBTC_Filter_order_price.png")
            # cv2.imshow("ETHBTC_Filter_order_price", img_ETHBTC_Filter_order_price_v)
            print(["Открытие картинки order_ETHBTC"])
            # await asyncio.sleep(1)
            # Баланс етериума
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

            # print(ETH_v_balance)
            # await asyncio.sleep(1)
            # Фильтрация етериума
            ETH_read = pd.read_csv("ETHBTC.csv")
            ETH_filter_balance_free = ETH_read.filter(["ETH_free"])
            ETH_filter_balance_locked = ETH_read.filter(["ETH_locked"])
            ETH_filter_balance_ETHUSDT = ETH_read.filter(["ETH_USDT"])
            ETH_filter_balance_price_ETHBTC = ETH_read.filter(["price_ETHBTC"])
            ETH_filter_balance_ETHUSDT_price = ETH_read.filter(["price_ETHUSDT"])

            # print(ETH_read)
            # Открытие картинки етериума
            for i in range(1):
                fig, axs = plt.subplots(2, 3)

                axs[0, 0].plot(ETH_filter_balance_free)
                axs[0, 1].plot(ETH_filter_balance_locked)
                axs[1, 0].plot(ETH_filter_balance_ETHUSDT)
                axs[1, 1].plot(ETH_filter_balance_price_ETHBTC)
                axs[1, 2].plot(ETH_filter_balance_ETHUSDT_price)

                plt.savefig("ETH_Filter.png")

            # img_ETH = cv2.imread("ETH_Filter.png")
            # cv2.imshow("ETH", img_ETH)
            print(["Открытие картинки ETH"])
            # await asyncio.sleep(0)


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
            data_read_pandas_ETHBTC = data_read_pandas_ETHBTC.tail(3000)
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
            #plt.plot(scaled_data)
            #plt.savefig("scaled_data_ETHBTC.png")

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
            balance_usd_RUB_usd_present = balance_usd_RUB_usd_free_locked_sum / float(symbol_USDTRUB["price"])

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
            data_read_pandas_BTCRUB = data_read_pandas_BTCRUB.tail(3000)
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
            #plt.plot(scaled_data)
            #plt.savefig("scaled_data_BTCRUB.png")

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
            preset_pred_price = round(pred_price_aa,0)
            print(pred_price)
            print(preset_pred_price)
            old_time = time.time() - start_time
            print("Время на расчеты :" + str(old_time))


            time.sleep(5)



            # pred_price = float(symbol_BTCRUB_price) + float(random.randint(-1000,1000))
            #Запуск ордера
            if preset_pred_price >= float(symbol_BTCRUB_price):
                info = client.get_all_tickers()
                symbol_BTCRUB = info[666]
                symbol_BTCRUB_price = symbol_BTCRUB["price"]
                a = float(1)
                b = float(balance_btc["free"])
                ab_sum = a * b
                data_coin = float(ab_sum) - min_lovume_btc
                print(data_coin)
                quantity = float(min_lovume_btc)
                print(quantity)
                if data_coin <= 0:
                    print([data_coin, a, b])
                    print(ab_sum)
                    quantity = float(min_lovume_btc)
                    print(quantity)
                    print("Недостаточно  btc")
                elif data_coin >= 0:
                    print([data_coin, a, b])
                    print("\n" + "SELL Продавать btc  " + str(preset_pred_price))
                    print(a)
                    quantity = float(min_lovume_btc)
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



            elif preset_pred_price <= float(symbol_BTCRUB_price):
                info = client.get_all_tickers()
                symbol_BTCRUB = info[666]
                min_lovume_rub = min_lovume_btc * float(symbol_BTCRUB_price)
                print([min_lovume_btc, min_lovume_rub])
                a = float(1)
                b = float(balance_RUB["free"])
                ab_sum = a * b
                data_coin = float(ab_sum) - min_lovume_rub
                print(data_coin)
                quantity = float(min_lovume_rub / float(symbol_BTCRUB_price))
                print(quantity)
                if data_coin <= 0:
                    print([data_coin, a, b])
                    print(ab_sum)
                    quantity = float(min_lovume_rub / float(symbol_BTCRUB_price))
                    print(quantity)
                    print("Недостаточно RUB для продажи")
                elif data_coin >= 0:
                    print([data_coin, a, b])
                    print("\n" + "BUY Покупать за RUB " + str(preset_pred_price))
                    print(a)
                    quantity = float(min_lovume_rub / float(symbol_BTCRUB_price))
                    print(quantity)
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
            data_read_pandas_AXSBTC = data_read_pandas_AXSBTC.tail(3000)
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
            #plt.plot(scaled_data)
            #plt.savefig("scaled_data_AXSBTC.png")

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


                min_lovume_axs = min_lovume_btc / float(symbol_AXSBTC_price)
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
        async def start_LTCBTC():
            import matplotlib.pyplot as plt
            info = client.get_all_tickers()
            # Курсы валют
            symbol_BTCUSDT = info[11]
            symbol_BTCUSDT_symbol = symbol_BTCUSDT["symbol"]
            symbol_BTCUSDT_price = symbol_BTCUSDT["price"]

            symbol_LTCBTC = info[1]
            symbol_LTCBTC_symbol = symbol_LTCBTC["symbol"]
            symbol_LTCBTC_price = symbol_LTCBTC["price"]

            symbol_LTCUSDT = info[190]
            symbol_LTCUSDT_symbol = symbol_LTCUSDT["symbol"]
            symbol_LTCUSDT_price = symbol_LTCUSDT["price"]

            # Балансы валют
            balance_btc = client.get_asset_balance(asset='BTC')
            # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
            balance_btc_asset, balance_btc_free, balance_btc_locked = balance_btc["asset"], float(
                balance_btc["free"]), float(balance_btc["locked"])
            balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
            balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

            balance_LTC = client.get_asset_balance(asset='LTC')
            balance_LTC_asset, balance_LTC_free, balance_LTC_locked = balance_LTC["asset"], float(
                balance_LTC["free"]), float(balance_LTC["locked"])
            balance_usd_LTC_usd_free_locked_sum = float(balance_LTC["free"]) + float(balance_LTC["locked"])
            balance_usd_LTC_usd_present = balance_usd_LTC_usd_free_locked_sum * float(symbol_LTCUSDT["price"])

            data_read_order_LTCBTC = pd.read_csv("LTCBTCorder.csv")
            data_read_order_LTCBTC_filter_price_LTCBTC = data_read_order_LTCBTC.filter(["price_LTCBTC"])
            data_read_order_LTCBTC_filter_price_order = data_read_order_LTCBTC.filter(["price_order"])
            data_read_order_LTCBTC_filter_origQty = data_read_order_LTCBTC.filter(["origQty"])

            figure, LTCBTC = plt.subplots(2, 2)
            LTCBTC[0, 0].plot(data_read_order_LTCBTC_filter_price_LTCBTC, "green")
            LTCBTC[0, 0].plot(data_read_order_LTCBTC_filter_price_order, "red")
            LTCBTC[1, 0].plot(data_read_order_LTCBTC_filter_origQty)
            plt.savefig("LTCBTC_Filter_order_price.png")

            # img_ETHBTC_Filter_order_price_v = cv2.imread("ETHBTC_Filter_order_price.png")
            # cv2.imshow("ETHBTC_Filter_order_price", img_ETHBTC_Filter_order_price_v)
            print(["Открытие картинки order_LTCBTC"])

            # Баланс биткоина
            LTC_v_balance = [balance_LTC_asset, balance_LTC_free, balance_LTC_locked, balance_usd_LTC_usd_present]

            LTC_v_balance_safe = open("LTCBTC.csv", "a")
            LTC_v_balance_safe.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
            LTC_v_balance_safe.write(",")
            LTC_v_balance_safe.write(str(symbol_LTCBTC_price))
            LTC_v_balance_safe.write(",")
            LTC_v_balance_safe.write(str(balance_LTC_free))
            LTC_v_balance_safe.write(",")
            LTC_v_balance_safe.write(str(balance_LTC_locked))
            LTC_v_balance_safe.write(",")
            LTC_v_balance_safe.write(str(balance_usd_LTC_usd_present))
            LTC_v_balance_safe.write("\n")
            LTC_v_balance_safe.close()

            # print(BTC_v_balance)
            # await asyncio.sleep(1)
            # Фильтр баланса биткоина
            LTC_read = pd.read_csv("LTCBTC.csv")
            LTC_filter_balance_free = LTC_read.filter(["balance_LTC_free"])
            LTC_filter_balance_locked = LTC_read.filter(["balance_LTC_locked"])
            LTC_filter_balance_LTCBTC = LTC_read.filter(["balance_usd_LTC_usd_present"])
            LTC_filter_balance_LTCBTC_price = LTC_read.filter(["symbol_LTCBTC_price"])

            # print(BTC_read)
            # Открытие баланса картинки биткоина
            for i in range(1):
                fig, LTC = plt.subplots(2, 2)
                LTC[0, 0].plot(LTC_filter_balance_free)
                LTC[0, 1].plot(LTC_filter_balance_locked)
                LTC[1, 0].plot(LTC_filter_balance_LTCBTC)
                LTC[1, 1].plot(LTC_filter_balance_LTCBTC_price)
                plt.savefig("LTC_Filter.png")

            # img_RUB = cv2.imread("RUB_Filter.png")
            # ("RUB", img_RUB)

            print(["Открытие картинки LTC"])

            # Баланс биткоина
            BTC_v_balance = [balance_btc_asset, balance_btc_free, balance_btc_locked, balance_usd_btc_usd_present]

            BTC_v_balance_safe = open("BTC_v_balance.csv", "a")
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

            # print(BTC_v_balance)
            # await asyncio.sleep(1)
            # Фильтр баланса биткоина
            BTC_read = pd.read_csv("BTC_v_balance.csv")
            BTC_filter_balance_free = BTC_read.filter(["BTC_free"])
            BTC_filter_balance_locked = BTC_read.filter(["BTC_locked"])
            BTC_filter_balance_BTCUSDT = BTC_read.filter(["BTC_USDT"])
            BTC_filter_balance_BTCUSDT_price = BTC_read.filter(["BTCUSDT_price"])

            # print(BTC_read)

            # Запуск нейронки
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

            data_read_pandas_LTCBTC = pd.read_csv("LTCBTC.csv")
            data_read_pandas_LTCBTC = data_read_pandas_LTCBTC.tail(3000)
            data_read_pandas_LTCBTC_shape_row, data_read_pandas_LTCBTC_shape_col = data_read_pandas_LTCBTC.shape[0], \
                                                                                   data_read_pandas_LTCBTC.shape[1]
            print(data_read_pandas_LTCBTC.shape)
            print([data_read_pandas_LTCBTC_shape_row, data_read_pandas_LTCBTC_shape_col])

            filter_LTCBTC_price = data_read_pandas_LTCBTC.filter(["symbol_LTCBTC_price"])

            print(filter_LTCBTC_price)

            # create dATEFRAME CLOSE
            data = data_read_pandas_LTCBTC.filter(["symbol_LTCBTC_price"])

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
            #plt.plot(scaled_data)
            plt.savefig("scaled_data_LTCBTC.png")

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

            model = tf.keras.models.load_model(os.path.join("./dnn/", "LTCBTC_model.h5"))
            model.fit(x_train, y_train, batch_size=1, epochs=1)

            model.save(os.path.join("./dnn/", "LTCBTC_model.h5"))
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

            new_df = data_read_pandas_LTCBTC.filter(["symbol_LTCBTC_price"])

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

            time.sleep(5)

            # pred_price = float(symbol_BTCRUB_price) + float(random.randint(-1000,1000))
            # Запуск ордера
            if preset_pred_price <= float(symbol_LTCBTC_price):
                info = client.get_all_tickers()
                symbol_LTCBTC = info[1]
                symbol_LTCBTC_price = symbol_LTCBTC["price"]
                a = float(1)
                b = float(balance_btc["free"])
                ab_sum = a * b
                data_coin = float(ab_sum) - min_lovume_btc
                print(data_coin)
                quantity = float(min_lovume_btc / float(symbol_LTCBTC_price))
                print(quantity)
                if data_coin <= 0:
                    print([data_coin, a, b])
                    print(ab_sum)
                    quantity = float(min_lovume_btc / float(symbol_LTCBTC_price))
                    print(quantity)
                    print("Недостаточно  btc")
                elif data_coin >= 0:
                    print([data_coin, a, b])
                    print("\n" + "BUY Покупать btc  " + str(preset_pred_price))
                    print(a)
                    quantity = float(min_lovume_btc / float(symbol_LTCBTC_price))
                    quantity_start = round(quantity, 3)
                    print(quantity_start)
                    order = client.order_limit_buy(symbol='LTCBTC', quantity=quantity_start,
                                                   price=preset_pred_price)
                    print(order)
                    data_safe_file_LTCBTC = open("LTCBTCorder.csv", "a")
                    data_safe_file_LTCBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(symbol_LTCBTC_price))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['symbol']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['orderId']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['transactTime']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['price']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['origQty']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['side']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write("\n")
                    data_safe_file_LTCBTC.close()



            elif preset_pred_price >= float(symbol_LTCBTC_price):
                info = client.get_all_tickers()
                symbol_LTCBTC = info[1]
                symbol_LTCBTC_price = symbol_LTCBTC["price"]
                a = float(symbol_LTCBTC_price)
                b = float(balance_LTC["free"])
                ab_sum = a * b
                data_coin = float(ab_sum) - min_lovume_btc
                print(data_coin)
                quantity = float(min_lovume_btc / float(symbol_LTCBTC_price))
                print(quantity)
                if data_coin <= 0:
                    print([data_coin, a, b])
                    print(ab_sum)
                    quantity = float(min_lovume_btc / float(symbol_LTCBTC_price))
                    print(quantity)
                    print("Недостаточно LTC для продажи")
                elif data_coin >= 0:
                    print([data_coin, a, b])
                    print("\n" + "BUY Покупать за LTC " + str(preset_pred_price))
                    print(a)
                    quantity = float(min_lovume_btc / float(symbol_LTCBTC_price))
                    quantity_start = round(quantity, 3)
                    print(quantity_start)
                    order = client.order_limit_sell(symbol='LTCBTC', quantity=quantity_start,
                                                    price=preset_pred_price)
                    print(order)
                    data_safe_file_LTCBTC = open("LTCBTCorder.csv", "a")
                    data_safe_file_LTCBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(symbol_LTCBTC_price))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['symbol']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['orderId']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['transactTime']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['price']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['origQty']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write(str(order['side']))
                    data_safe_file_LTCBTC.write(",")
                    data_safe_file_LTCBTC.write("\n")
                    data_safe_file_LTCBTC.close()



        async def start_BNBBTC():
                import matplotlib.pyplot as plt

                info = client.get_all_tickers()
                # Курсы валют
                symbol_BTCUSDT = info[11]
                symbol_BTCUSDT_symbol = symbol_BTCUSDT["symbol"]
                symbol_BTCUSDT_price = symbol_BTCUSDT["price"]

                symbol_BNBBTC = info[2]
                symbol_BNBBTC_symbol = symbol_BNBBTC["symbol"]
                symbol_BNBBTC_price = symbol_BNBBTC["price"]

                symbol_BNBUSDT = info[98]
                symbol_BNBUSDT_symbol = symbol_BNBUSDT["symbol"]
                symbol_BNBUSDT_price = symbol_BNBUSDT["price"]

                # Балансы валют
                balance_btc = client.get_asset_balance(asset='BTC')
                # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
                balance_btc_asset, balance_btc_free, balance_btc_locked = balance_btc["asset"], float(
                    balance_btc["free"]), float(balance_btc["locked"])
                balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
                balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

                balance_BNB = client.get_asset_balance(asset='BNB')
                balance_BNB_asset, balance_BNB_free, balance_BNB_locked = balance_BNB["asset"], float(
                    balance_BNB["free"]), float(balance_BNB["locked"])
                balance_usd_BNB_usd_free_locked_sum = float(balance_BNB["free"]) + float(balance_BNB["locked"])
                balance_usd_BNB_usd_present = balance_usd_BNB_usd_free_locked_sum * float(symbol_BNBUSDT["price"])

                data_read_order_BNBBTC = pd.read_csv("BNBBTCorder.csv")
                data_read_order_BNBBTC_filter_price_BNBBTC = data_read_order_BNBBTC.filter(["price_BNBBTC"])
                data_read_order_BNBBTC_filter_price_order = data_read_order_BNBBTC.filter(["price_order"])
                data_read_order_BNBBTC_filter_origQty = data_read_order_BNBBTC.filter(["origQty"])

                figure, BNBBTC = plt.subplots(2, 2)
                BNBBTC[0, 0].plot(data_read_order_BNBBTC_filter_price_BNBBTC, "green")
                BNBBTC[0, 0].plot(data_read_order_BNBBTC_filter_price_order, "red")
                BNBBTC[1, 0].plot(data_read_order_BNBBTC_filter_origQty)
                plt.savefig("BNBBTC_Filter_order_price.png")

                # img_ETHBTC_Filter_order_price_v = cv2.imread("ETHBTC_Filter_order_price.png")
                # cv2.imshow("ETHBTC_Filter_order_price", img_ETHBTC_Filter_order_price_v)
                print(["Открытие картинки order_BNBBTC"])

                # Баланс биткоина
                BNB_v_balance = [balance_BNB_asset, balance_BNB_free, balance_BNB_locked, balance_usd_BNB_usd_present]

                BNB_v_balance_safe = open("BNBBTC.csv", "a")
                BNB_v_balance_safe.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                BNB_v_balance_safe.write(",")
                BNB_v_balance_safe.write(str(symbol_BNBBTC_price))
                BNB_v_balance_safe.write(",")
                BNB_v_balance_safe.write(str(balance_BNB_free))
                BNB_v_balance_safe.write(",")
                BNB_v_balance_safe.write(str(balance_BNB_locked))
                BNB_v_balance_safe.write(",")
                BNB_v_balance_safe.write(str(balance_usd_BNB_usd_present))
                BNB_v_balance_safe.write("\n")
                BNB_v_balance_safe.close()

                # print(BTC_v_balance)
                # await asyncio.sleep(1)
                # Фильтр баланса биткоина
                BNB_read = pd.read_csv("BNBBTC.csv")
                BNB_filter_balance_free = BNB_read.filter(["balance_BNB_free"])
                BNB_filter_balance_locked = BNB_read.filter(["balance_BNB_locked"])
                BNB_filter_balance_BNBBTC = BNB_read.filter(["balance_usd_BNB_usd_present"])
                BNB_filter_balance_BNBBTC_price = BNB_read.filter(["symbol_BNBBTC_price"])

                # print(BTC_read)
                # Открытие баланса картинки биткоина
                for i in range(1):
                    fig, BNB = plt.subplots(2, 2)
                    BNB[0, 0].plot(BNB_filter_balance_free)
                    BNB[0, 1].plot(BNB_filter_balance_locked)
                    BNB[1, 0].plot(BNB_filter_balance_BNBBTC)
                    BNB[1, 1].plot(BNB_filter_balance_BNBBTC_price)
                    plt.savefig("BNB_Filter.png")

                # img_RUB = cv2.imread("RUB_Filter.png")
                # ("RUB", img_RUB)

                print(["Открытие картинки BNB"])

                # Баланс биткоина
                BTC_v_balance = [balance_btc_asset, balance_btc_free, balance_btc_locked, balance_usd_btc_usd_present]

                BTC_v_balance_safe = open("BTC_v_balance.csv", "a")
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

                # print(BTC_v_balance)
                # await asyncio.sleep(1)
                # Фильтр баланса биткоина
                BTC_read = pd.read_csv("BTC_v_balance.csv")
                BTC_filter_balance_free = BTC_read.filter(["BTC_free"])
                BTC_filter_balance_locked = BTC_read.filter(["BTC_locked"])
                BTC_filter_balance_BTCUSDT = BTC_read.filter(["BTC_USDT"])
                BTC_filter_balance_BTCUSDT_price = BTC_read.filter(["BTCUSDT_price"])

                # print(BTC_read)

                # Запуск нейронки
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

                data_read_pandas_BNBBTC = pd.read_csv("BNBBTC.csv")
                data_read_pandas_BNBBTC = data_read_pandas_BNBBTC.tail(3000)
                data_read_pandas_BNBBTC_shape_row, data_read_pandas_BNBBTC_shape_col = data_read_pandas_BNBBTC.shape[0], \
                                                                                       data_read_pandas_BNBBTC.shape[1]
                print(data_read_pandas_BNBBTC.shape)
                print([data_read_pandas_BNBBTC_shape_row, data_read_pandas_BNBBTC_shape_col])

                filter_BNBBTC_price = data_read_pandas_BNBBTC.filter(["symbol_BNBBTC_price"])

                print(filter_BNBBTC_price)

                # create dATEFRAME CLOSE
                data = data_read_pandas_BNBBTC.filter(["symbol_BNBBTC_price"])

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
                plt.savefig("scaled_data_BNBBTC.png")

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

                model = tf.keras.models.load_model(os.path.join("./dnn/", "BNBBTC_model.h5"))
                model.fit(x_train, y_train, batch_size=1, epochs=1)

                model.save(os.path.join("./dnn/", "BNBBTC_model.h5"))
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

                new_df = data_read_pandas_BNBBTC.filter(["symbol_BNBBTC_price"])

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

                time.sleep(5)

                # pred_price = float(symbol_BTCRUB_price) + float(random.randint(-1000,1000))
                # Запуск ордера
                if preset_pred_price <= float(symbol_BNBBTC_price):
                    info = client.get_all_tickers()
                    symbol_BNBBTC = info[2]
                    symbol_BNBBTC_price = symbol_BNBBTC["price"]
                    a = float(1)
                    b = float(balance_btc["free"])
                    ab_sum = a * b
                    data_coin = float(ab_sum) - min_lovume_btc
                    print(data_coin)
                    quantity = float(min_lovume_btc / float(symbol_BNBBTC_price))
                    print(quantity)
                    if data_coin <= 0:
                        print([data_coin, a, b])
                        print(ab_sum)
                        quantity = float(min_lovume_btc / float(symbol_BNBBTC_price))
                        print(quantity)
                        print("Недостаточно  btc")
                    elif data_coin >= 0:
                        print([data_coin, a, b])
                        print("\n" + "BUY Покупать btc  " + str(preset_pred_price))
                        print(a)
                        quantity = float(min_lovume_btc / float(symbol_BNBBTC_price))
                        quantity_start = round(quantity, 3)
                        print(quantity_start)
                        order = client.order_limit_buy(symbol='BNBBTC', quantity=quantity_start,
                                                       price=preset_pred_price)
                        print(order)
                        data_safe_file_BNBBTC = open("BNBBTCorder.csv", "a")
                        data_safe_file_BNBBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(symbol_BNBBTC_price))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['symbol']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['orderId']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['transactTime']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['price']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['origQty']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['side']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write("\n")
                        data_safe_file_BNBBTC.close()



                elif preset_pred_price >= float(symbol_BNBBTC_price):

                    info = client.get_all_tickers()
                    symbol_BNBBTC = info[2]
                    symbol_BNBBTC_price = symbol_BNBBTC["price"]
                    a = float(symbol_BNBBTC_price)
                    b = float(balance_BNB["free"])
                    ab_sum = a * b
                    data_coin = float(ab_sum) - min_lovume_btc
                    print(data_coin)
                    quantity = float(min_lovume_btc / float(symbol_BNBBTC_price))
                    print(quantity)
                    if data_coin <= 0:
                        print([data_coin, a, b])
                        print(ab_sum)
                        quantity = float(min_lovume_btc / float(symbol_BNBBTC_price))
                        print(quantity)
                        print("Недостаточно BNB для продажи")
                    elif data_coin >= 0:
                        print([data_coin, a, b])
                        print("\n" + "BUY Покупать за BNB " + str(preset_pred_price))
                        print(a)
                        quantity = float(min_lovume_btc / float(symbol_BNBBTC_price))
                        quantity_start = round(quantity, 3)
                        print(quantity_start)
                        order = client.order_limit_sell(symbol='BNBBTC', quantity=quantity_start,
                                                        price=preset_pred_price)
                        print(order)
                        data_safe_file_BNBBTC = open("BNBBTCorder.csv", "a")
                        data_safe_file_BNBBTC.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(symbol_BNBBTC_price))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['symbol']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['orderId']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['transactTime']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['price']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['origQty']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write(str(order['side']))
                        data_safe_file_BNBBTC.write(",")
                        data_safe_file_BNBBTC.write("\n")
                        data_safe_file_BNBBTC.close()





        #Запуск асинхронной программы
        ioloop = asyncio.get_event_loop()
        tasks = [ioloop.create_task(start_ETHBTC()),
                 ioloop.create_task(start_LTCBTC()),
                 ioloop.create_task(start_BNBBTC()),
                 ioloop.create_task(control_balance_BTC()),ioloop.create_task(control_balance_RUB()),ioloop.create_task(start_BTCRUB()),ioloop.create_task(BTCRUBorder()),
                 ioloop.create_task(control_balance_AXS()),ioloop.create_task(start_AXSBTC()),ioloop.create_task(AXSBTCorder())
                 ] #
        wait_tasks = asyncio.wait(tasks)
        ioloop.run_until_complete(wait_tasks)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    elif time_hour == str(time.strftime("%H")):
        for i in range(10):
            time.sleep(1)
            print(["pause", i])
        print(["равно h", time_hour, str(time.strftime("%H:%M:%S"))])

#закрытие асинхронной программы
ioloop.close()
