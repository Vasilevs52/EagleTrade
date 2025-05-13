from binance import Client
import pandas as pd
from typing import List, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(
        symbols: List[str],
        interval: str,
        days: int,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Получает исторические данные по криптовалютам с Binance.

    Параметры:
        symbols (List[str]): Список символов (например, ['BTCUSDT', 'ETHUSDT'])
        interval (str): Интервал времени (например, '1d', '1h', '15m')
        days (int): Количество дней данных для получения
        api_key (Optional[str]): API ключ Binance (если требуется аутентификация)
        api_secret (Optional[str]): Секретный ключ API Binance

    Возвращает:
        pd.DataFrame: DataFrame с историческими данными
    """
    try:
        # Инициализация клиента Binance
        client = Client(api_key=api_key, api_secret=api_secret) if api_key and api_secret else Client()

        # Создание пустого DataFrame с правильными колонками
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore', 'symbol'
        ]
        data = pd.DataFrame(columns=columns)

        for symbol in symbols:
            logger.info(f"Получение данных для {symbol}...")

            # Получение данных с Binance API
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=days
            )

            # Создание DataFrame из полученных данных
            df = pd.DataFrame(klines, columns=columns[:-1])  # Все колонки кроме 'symbol'

            # Преобразование и очистка данных
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Преобразование числовых колонок
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Добавление символа
            df['symbol'] = symbol

            # Конкатенация данных
            data = pd.concat([data, df], axis=0)

        logger.info(f"Успешно получено {len(data)} записей.")
        data = data.drop('timestamp', axis = 1)
        return data

    except Exception as e:
        logger.error(f"Ошибка при получении данных: {str(e)}")
        raise


def shuffle_df(df: pd.DataFrame, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Перемешивает DataFrame случайным образом.

    Параметры:
        df (pd.DataFrame): Входной DataFrame
        random_state (Optional[int]): Seed для воспроизводимости результатов

    Возвращает:
        pd.DataFrame: Перемешанный DataFrame
    """
    try:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Ошибка при перемешивании DataFrame: {str(e)}")
        raise


# Пример использования
if __name__ == "__main__":
    try:
        # Получение данных для BTCUSDT и ETHUSDT за последние 30 дней с дневным интервалом
        data = get_data(
            symbols=['BTCUSDT', 'ETHUSDT'],
            interval='1d',
            days=30
        )
        d = data.iloc[1]
        print(d)



    except Exception as e:
        logger.error(f"Ошибка в примере использования: {str(e)}")

