from binance.client import Client
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Data.Binance import get_data, shuffle_df
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Константы
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
DAYS = 730
INTERVAL = '1h'
TEST_SIZE = 0.2


def sliding_window_trend_labeling(series, window_size=5):
    """Разметка временного ряда с использованием скользящего окна."""
    if len(series) < window_size:
        raise ValueError("Длина ряда должна быть не меньше размера окна")

    labels = np.zeros(len(series))

    for i in range(window_size - 1, len(series)):
        window = series.iloc[i - window_size + 1: i + 1] if isinstance(series, pd.Series) \
            else series[i - window_size + 1: i + 1]
        X = np.arange(window_size).reshape(-1, 1)
        y = window.values.reshape(-1, 1) if isinstance(series, pd.Series) \
            else window.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0][0]
        labels[i] = 1 if slope > 0 else 0  # Only two classes: 0 (down) and 1 (up)

    labels[:window_size - 1] = np.nan
    return labels


def get_features(data: pd.DataFrame):
    arr_df_data = []
    for symbol in data['symbol'].unique():
        df = data[data['symbol'] == symbol].copy()

        # Убедимся, что данные имеют правильный тип
        df = df.apply(pd.to_numeric, errors='ignore')

        # Удалим строки с NaN перед вычислением индикаторов
        df = df.dropna()

        # Проверим, что данных достаточно для расчетов
        if len(df) < 200:  # MA200 требует минимум 200 точек
            continue

        close_prices = df['close'].values.astype('float64')
        high_prices = df['high'].values.astype('float64')
        low_prices = df['low'].values.astype('float64')
        volume_values = df['volume'].values.astype('float64')

        df['target'] = sliding_window_trend_labeling(df['close'], window_size=10)

        # Технические индикаторы (с проверкой на достаточность данных)
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

        # ATR требует float64
        df['ATR7'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=7)
        df['ATR14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['ATR21'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=21)

        # Производные признаки
        df['Pct_Change1'] = df['close'].pct_change(1)
        df['Pct_Change3'] = df['close'].pct_change(3)
        df['Pct_Change7'] = df['close'].pct_change(7)
        df['Pct_Change30'] = df['close'].pct_change(30)
        df['Log_Return1'] = np.log(df['close'] / df['close'].shift(1))
        df['Volume_Change'] = df['volume'].pct_change()
        df['Volume_MA20'] = talib.MA(volume_values, timeperiod=20)
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA20']
        df['Volatility'] = (df['high'] - df['low']) / df['open']
        df['Close_Open_Ratio'] = df['close'] / df['open']

        # Удалим строки с NaN после всех расчетов
        df = df.dropna()

        if not df.empty:  # Добавляем только непустые DataFrame
            arr_df_data.append(df)

    if not arr_df_data:  # Если все DataFrame пустые
        return pd.DataFrame(), pd.Series()

    data = pd.concat([df for df in arr_df_data if not df.empty], axis=0)
    data = shuffle_df(data)
    X = data.drop(['target', 'open', 'high', 'low', 'close', 'volume', 'symbol'], axis=1, errors='ignore')
    y = data['target']
    return X, y


def main():
    data = get_data(SYMBOLS, INTERVAL, DAYS)
    X, y = get_features(data)

    # Проверим баланс классов
    print("\nClass distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

    # Создание DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Улучшенные параметры модели
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 5,
        'learning_rate': 0.05,
        'eval_metric': 'error',  # Use 'error' for binary classification
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42
    }

    # Добавим валидационный набор для ранней остановки
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    # Обучение с ранней остановкой
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # Увеличим количество итераций
        evals=evals,
        early_stopping_rounds=20,  # Остановка если нет улучшений 20 раундов
        verbose_eval=10  # Выводим лог каждые 10 итераций
    )

    # Предсказание
    # Predict probabilities
    y_pred_proba = model.predict(dtest)
    # Convert to class labels
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Оценка модели
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    # Визуализация матрицы ошибок
    # Update confusion matrix labels
    plt.figure(figsize=(6, 4))  # Smaller figure for binary classification
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Визуализация важности признаков
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20, height=0.8)
    plt.title('Feature Importance')
    plt.show()



    # Вывод итоговой точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Model Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()