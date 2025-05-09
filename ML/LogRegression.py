import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import talib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime
from joblib import dump, load
# Настройки API Binance
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'


def sliding_window_trend_labeling(series, window_size=5):
    """
    Разметка временного ряда с использованием скользящего окна.

    Параметры:
    - series: pandas Series или numpy array с ценовыми данными
    - window_size: размер окна для анализа (по умолчанию 5)

    Возвращает:
    - Массив меток тренда: 1 (рост), -1 (падение), 0 (стабильность)
    """
    if len(series) < window_size:
        raise ValueError("Длина ряда должна быть не меньше размера окна")

    labels = np.zeros(len(series))

    for i in range(window_size - 1, len(series)):
        # Выделяем окно данных
        window = series.iloc[i - window_size + 1: i + 1] if isinstance(series, pd.Series) \
            else series[i - window_size + 1: i + 1]

        # Создаем признаки (номера точек времени)
        X = np.arange(window_size).reshape(-1, 1)
        y = window.values.reshape(-1, 1) if isinstance(series, pd.Series) \
            else window.reshape(-1, 1)

        # Обучаем линейную регрессию
        model = LinearRegression()
        model.fit(X, y)

        # Определяем наклон (коэффициент β)
        slope = model.coef_[0][0]

        # Присваиваем метку
        labels[i] = 1 if slope > 0 else (-1 if slope < 0 else 0)

    # Первые (window_size - 1) точек не могут быть размечены
    labels[:window_size - 1] = np.nan

    return labels

def download_btc_data(start_date, end_date, timeframe):
    client = Client(API_KEY, API_SECRET)

    # Преобразуем даты в строки в формате Binance
    start_str = start_date.strftime("%d %b, %Y") if isinstance(start_date, datetime) else start_date
    end_str = end_date.strftime("%d %b, %Y") if isinstance(end_date, datetime) else end_date

    # Получаем данные (klines) с Binance
    klines = client.get_historical_klines(
        symbol="BTCUSDT",
        interval=timeframe,
        start_str=start_str,
        end_str=end_str
    )

    # Преобразуем данные в DataFrame
    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]

    df = pd.DataFrame(klines, columns=columns)

    # Преобразуем числовые колонки
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric)

    # Преобразуем временные метки
    df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Date', inplace=True)

    # Оставляем только нужные колонки
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return df



def create_features(data):
    df = data.copy()
    close_prices = df['Close'].values

    df['Target'] = sliding_window_trend_labeling(df['Close'], window_size=10)

    # Технические индикаторы
    df['MA5'] = talib.MA(close_prices, timeperiod=5)
    df['MA10'] = talib.MA(close_prices, timeperiod=10)
    df['MA20'] = talib.MA(close_prices, timeperiod=20)
    df['MA50'] = talib.MA(close_prices, timeperiod=50)
    df['MA100'] = talib.MA(close_prices, timeperiod=100)
    df['MA200'] = talib.MA(close_prices, timeperiod=200)

    df['RSI14'] = talib.RSI(close_prices, timeperiod=14)
    df['RSI30'] = talib.RSI(close_prices, timeperiod=30)

    macd, macdsignal, macdhist = talib.MACD(close_prices)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    df['ATR7'] = talib.ATR(high, low, close, timeperiod=7)
    df['ATR14'] = talib.ATR(high, low, close, timeperiod=14)
    df['ATR21'] = talib.ATR(high, low, close, timeperiod=21)

    # Процентные изменения
    df['Pct_Change1'] = df['Close'].pct_change(1)
    df['Pct_Change3'] = df['Close'].pct_change(3)
    df['Pct_Change7'] = df['Close'].pct_change(7)
    df['Pct_Change30'] = df['Close'].pct_change(30)

    # Логарифмическая доходность
    df['Log_Return1'] = np.log(df['Close'] / df['Close'].shift(1))

    # Объемы
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA20'] = talib.MA(df['Volume'].values, timeperiod=20)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']

    df['Volatility'] = (df['High'] - df['Low']) / df['Open']  # дневная волатильность
    df['Close_Open_Ratio'] = df['Close'] / df['Open']  # соотношение закрытия к открытию

    # Удаляем строки с NaN
    df.dropna(inplace=True)

    return df


def prepare_data(df):
    features = df.drop(['Target', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    target = df['Target']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, features.columns


def train_and_evaluate(X, y):
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\n=== Результаты обучения ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, y_pred, y_pred_proba, X_train, X_test, y_train, y_test, split_idx


def visualize_btc_results(df, model, feature_names, X_test, y_test, y_pred, y_pred_proba, split_idx):
    plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 2, height_ratios=[1.5, 1, 1, 1, 0.5])

    # Заголовок с информацией
    plt.suptitle(
        f"Результаты предсказания тренда BTC\n"
        f"Точность: {accuracy_score(y_test, y_pred):.2%} | "
        f"Период: {df.index[0].strftime('%Y-%m-%d')} - {df.index[-1].strftime('%Y-%m-%d')}",
        y=1.02, fontsize=14
    )

    # 1. График цены с предсказаниями
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(df.index, df['Close'], label='Цена BTC', color='blue')
    ax1.axvline(x=df.index[split_idx], color='red', linestyle='--', label='Начало теста')

    # Цвета для классов
    colors = {1: 'green', -1: 'red', 0: 'gray'}

    # Отображаем предсказания только для тестовой выборки
    test_dates = df.index[split_idx:]

    for trend in [-1, 0, 1]:
        mask = (y_pred == trend)
        ax1.scatter(test_dates[mask], df['Close'][split_idx:][mask],
                    color=colors[trend], label=f'Предсказан: {get_trend_name(trend)}',
                    alpha=0.7)

    # Разметка правильных/неправильных предсказаний
    correct = (y_pred == y_test)
    ax1.scatter(test_dates[correct], df['Close'][split_idx:][correct],
                facecolors='none', edgecolors='lime', s=100,
                label='Правильные предсказания')
    ax1.scatter(test_dates[~correct], df['Close'][split_idx:][~correct],
                marker='x', color='black', s=100,
                label='Ошибки предсказания')

    ax1.set_title('Цена BTC с предсказаниями тренда', pad=20)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True)
    format_axis_dates(ax1)

    # 2. График вероятностей (только если есть вероятности для всех классов)
    if y_pred_proba.shape[1] == 3:  # Для 3 классов
        ax2 = plt.subplot(gs[1, :])

        # Создаем DataFrame с вероятностями
        proba_df = pd.DataFrame(y_pred_proba,
                                columns=['Спад', 'Стабильность', 'Рост'],
                                index=test_dates)

        ax2.stackplot(test_dates, proba_df.T,
                      labels=proba_df.columns,
                      colors=['red', 'gray', 'green'], alpha=0.4)

        ax2.set_title('Вероятности предсказаний', pad=20)
        ax2.legend(loc='upper left')
        ax2.grid(True)
        format_axis_dates(ax2)
    else:
        print("Вероятности не доступны для всех классов")

    # 3. График RSI
    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(df.index, df['RSI14'], color='orange')
    ax3.axhline(70, color='red', linestyle='--')
    ax3.axhline(30, color='green', linestyle='--')
    ax3.set_title('RSI 14')
    ax3.grid(True)

    # 4. График MACD
    ax4 = plt.subplot(gs[2, 1])
    ax4.plot(df.index, df['MACD'], label='MACD')
    ax4.plot(df.index, df['MACD_Signal'], label='Signal')
    ax4.bar(df.index, df['MACD_Hist'], color='gray', alpha=0.3)
    ax4.set_title('MACD')
    ax4.legend()
    ax4.grid(True)

    # 5. График объемов
    ax5 = plt.subplot(gs[3, :])
    ax5.bar(df.index, df['Volume'], color='blue', alpha=0.5)
    ax5.plot(df.index, df['Volume_MA20'], color='red')
    ax5.set_title('Торговые объемы')
    ax5.grid(True)
    format_axis_dates(ax5)

    # 6. Информация о модели
    ax6 = plt.subplot(gs[4, :])
    ax6.axis('off')
    model_info = (
        f"Модель: {model.__class__.__name__}\n"
        f"Точность: {accuracy_score(y_test, y_pred):.2%}\n"
        f"Классы: {pd.Series(y_test).value_counts().to_dict()}\n"
        f"Признаки: {len(feature_names)}"
    )
    ax6.text(0.1, 0.5, model_info, fontsize=12)

    plt.tight_layout()
    plt.show()


def get_trend_name(trend_code):
    names = {-1: 'Спад', 0: 'Стабильность', 1: 'Рост'}
    return names.get(trend_code, 'Неизвестно')


def format_axis_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=45)


def main():
    try:
        # Параметры
        start_date = '2020-01-01'
        end_date = '2021-01-01'
        timeframe = '1h'  # Можно изменить на '1d' для дневных данных

        print("Загрузка данных BTC с Binance...")
        data = download_btc_data(start_date, end_date, timeframe)

        if data.empty:
            print("Ошибка: Не удалось загрузить данные. Проверьте подключение к интернету и API ключи.")
            return

        print("Создание признаков...")
        df = create_features(data)
        print("\nПример данных с признаками:")
        print(df[['Close', 'Target', 'MA50', 'MA200', 'RSI14', 'ATR14', 'Log_Return1']].tail())

        print("\nПодготовка данных для модели...")
        X, y, feature_names = prepare_data(df)

        print("\nОбучение модели...")
        model, y_pred, y_pred_proba, X_train, X_test, y_train, y_test, split_idx = train_and_evaluate(X, y)

        print("\nВизуализация результатов...")
        visualize_btc_results(df, model, feature_names, X_test, y_test, y_pred, y_pred_proba, split_idx)

        save = input("\nСохранить модель? y/n ")
        if save == 'y':
            dump(model, 'models/logistic_regression_trend_1h_model.joblib')
            print('Модель сохронена')
        else:
            print('модель не сохранена')

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()