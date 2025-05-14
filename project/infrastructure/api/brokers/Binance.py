from binance.client import Client
import pandas as pd

class BinanceBroker():

    @staticmethod
    def get_history_data(symbols: list, interval: str, start_date: str, end_date: str):
        client = Client()
        columns = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Volume",
            "Taker Buy Quote Volume",
            "Ignore"
        ]

        arr_df = []
        for symbol in symbols:

            data = client.get_historical_klines(symbol = symbol,
                                                interval = interval,
                                                start_str = start_date,
                                                end_str = end_date)
            df = pd.DataFrame(data, columns=columns, dtype=float)
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
            df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
            arr_df.append(df)
        df = pd.concat(arr_df, axis=1)
        return df

if __name__ == '__main__':
    broker = BinanceBroker()
    data = broker.get_history_data(['BTCUSDT'], '1h', '01-01-2020', '02-01-2020')