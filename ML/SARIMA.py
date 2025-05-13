from binance.client import Client
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt

# Константы
SYMBOL = 'BTCUSDT'
DAYS = 730
INTERVAL = '1h'
SEASONALITY = 24  # 24 часа в сутках
TEST_SIZE = 0.2  # 20% данных для теста


def get_data(symbol=SYMBOL, days=DAYS, interval=INTERVAL):
    """Получение данных с Binance API"""
    client = Client()
    data = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        limit=days
    )

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df


def prepare_data(data):
    """Подготовка данных и разделение на train/test"""
    # Создаем копию данных
    df = data[['close']].copy()

    # Разделяем на train/test
    train_size = int(len(df) * (1 - TEST_SIZE))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    return train, test


def evaluate_model(y_true, y_pred):
    """Оценка точности модели"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}


def plot_results(train, test, forecasts, ci, metrics):
    """Визуализация результатов"""
    plt.figure(figsize=(14, 7))

    # Обучающие данные
    plt.plot(train.index, train['close'], label='Train Data', color='blue')

    # Тестовые данные
    plt.plot(test.index, test['close'], label='Test Data', color='green')

    # Прогнозы
    plt.plot(forecasts.index, forecasts, label='Forecast', color='red')

    # Доверительный интервал
    plt.fill_between(ci.index,
                     ci.iloc[:, 0],
                     ci.iloc[:, 1], color='red', alpha=0.3, label='95% CI')

    # Метрики
    metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
    plt.text(0.02, 0.95, metrics_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f'BTC Price Forecast (SARIMA) - Last {DAYS} days, {INTERVAL} interval')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Получение и подготовка данных
    data = get_data()
    train, test = prepare_data(data)

    # Создание и обучение модели
    model = SARIMAX(train['close'],
                    order=(2, 1, 2),  # Изменены параметры для лучшей стабильности
                    seasonal_order=(1, 1, 1, SEASONALITY),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit(disp=False)
    print(results.summary())

    # Прогнозирование на тестовом наборе
    forecast_steps = len(test)
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # Оценка точности
    metrics = evaluate_model(test['close'], forecast_mean)

    # Визуализация
    plot_results(train, test, forecast_mean, confidence_intervals, metrics)

    # Диагностика остатков
    results.plot_diagnostics(figsize=(12, 8))
    plt.show()


if __name__ == '__main__':
    main()