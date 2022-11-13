from pandas.tseries.offsets import Second
import ccxt
import pandas as pd
import os
from pathlib import Path
from slugify import slugify
import datetime
import sys

PATH_EXPORT_FILES = "./export/"
TIMEFRAME_IN_MNS = 15
EXCHANGE = ccxt.bitfinex


def refresh_data_to_now(
    timeframe_in_mns=TIMEFRAME_IN_MNS, exchange=EXCHANGE, folder=PATH_EXPORT_FILES
):
    """Refresh the csv of the cryptocurrences files located in 'folder' so that it download
    the latest prices.

    Parameters
    ----------
    timeframe_in_mns : int
        interval in minutes between each measure.
    exchange: ccxt.exchange
        platform to use to download the data.
    folder: string
        path of the folder where the csv to refresh are located
    """
    files = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            files.append((os.path.join(folder, file), os.path.splitext(file)[0]))

    # For each crypto to update
    for full_path, file in files:
        # obtain the two currencies for the pricing using the filename
        currency_name = file.split("_")[0]
        currency_name = currency_name.replace("-", "/").upper()
        print(full_path, currency_name)

        # load the csv and get the the last timestamp used (last row, second col )
        cs = pd.read_csv(full_path)
        timestamp_iso = int(cs.iloc[-1, 0])
        print(timestamp_iso)
        timestamp_iso += timeframe_in_mns * 60 * 1000

        # download every period since the moment of the last timestamp
        exch = exchange()
        markets = exch.load_markets()
        df = _download_data(exch, currency_name, timeframe_in_mns, timestamp_iso)

        # update the file
        if df is not None:
            print(
                "refresh currency: ",
                currency_name,
                " number of lines updated: ",
                len(df),
            )
            with open(full_path) as f:
                f.write("\n")
            df.to_csv(full_path, mode="a", header=False, index=False)


def get_list_currencies(exchange=EXCHANGE):
    """display a list of all the currencies available for a given exchange
    Parameters
    ----------
    exchange: ccxt.exchange
        platform to use to list the available currencies
    """
    [print(k) for k in EXCHANGE().loadMarkets().keys() if k.endswith("/USD")]


# Limitation of crypto exchange platforms:
# Coinbase has no fetch OHLCV
# Binance, kucoin has no btc/usd rate
# Kraken is limited to 720 OHLCV per fetch
# Bitfinex is fine for the data we need
def get_crypto_data(
    start=datetime.datetime(2020, 10, 10, 00, 00, 00, 00).isoformat(),
    timeframe_in_mns=TIMEFRAME_IN_MNS,
    exchange=EXCHANGE,
    currencies=[
        "BTC/USD",
        "ETH/USD",
        "XRP/USD",
        "ADA/USD",
        "USDT/USD",
    ],
    path_export=PATH_EXPORT_FILES,
):
    """Download the crypto data for a given start date, a timeframe which is interval
    between two measures.

    Parameters
    ----------
    start : datetime in isoformat
        start date corresponding to when we should start to get data about the prices of
        the given currencies.
    timeframe_in_mns: int
        interval in minutes between each measure.
    exchange: ccxt.exchange
        platform to use to download the data.
    currencies: array of String
        List of the currencies whose prices we want to download.
    """
    exch = exchange()
    if not exch.has["fetchOHLCV"]:
        print("No possibility to fetch OLCV with this exchange platform.")
        return

    start_iso = exch.parse8601(start)

    for currency in currencies:

        orders = _download_data(exch, currency, timeframe_in_mns, start_iso)
        if orders is None:
            continue

        Path(path_export).mkdir(exist_ok=True)
        full_path = os.path.join(path_export, f"{slugify(currency)}_export.csv")
        print(
            "downloaded all data for ", currency, "\n exporting the data to ", full_path
        )

        orders.to_csv(full_path, index=False)


def _download_data(exchange, currency, timeframe_in_mns, start_iso):
    """Download the data for a given currency given a timeframe and a
    start timestamp

    Parameters
    ----------
    exchange: ccxt.exchange
        platform to use to download the data.
    currency: string
        currency whose price we want to download.
    timeframe_in_mns: int
        interval in minutes between each measure.

    Returns
    -------
    Panda Dataframe of all the prices provided by exchange if
    it is available.
    """
    timeframe = str(timeframe_in_mns) + "m"
    timeframe_in_ms = timeframe_in_mns * 60 * 1000
    markets = exchange.load_markets()

    if currency in markets.keys():
        all_orders = []
        while start_iso < exchange.milliseconds():
            symbol = currency
            orders = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=start_iso, limit=2000
            )
            if len(orders):
                start_iso = orders[-1][0] + timeframe_in_ms
                all_orders += orders
            else:
                break

        return pd.DataFrame(all_orders)
    else:
        print(currency, " is not possible with the platform ", exchange)


def main(args):
    """Main function that will call either
        get_crypto_data() by default
        get_list_currencies() if the first argument is 'list'
        refresh_data_to_now() if the first argument is 'refresh'

    Parameters
    ----------
    args
        arguments given when the file is called. Only the first one is considered
    currency: string
        currency whose price we want to download.
    timeframe_in_mns: int
        interval in minutes between each measure.

    Returns
    -------
    Panda Dataframe of all the prices provided by exchange if
    it is available.
    """
    if len(args) > 0:
        if args[0] == "refresh":
            refresh_data_to_now()
        elif args[0] == "list":
            get_list_currencies()
    else:
        get_crypto_data()


if __name__ == "__main__":
    main(sys.argv[1:])
