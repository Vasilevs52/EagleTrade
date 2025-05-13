from core.strategies.metrics import EMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RBUP():

    def __init__(self, settings: dict):
        self.window = settings['window']

    def get_signal(self, data: pd.DataFrame) -> int:


        prices = data['Close']
        ema = EMA(prices, self.window)
        alpha = 2 / (self.window + 1)
        sigma =

    def get_window(self):
        pass