from abc import ABC, abstractmethod
import pandas as pd

class IStrategy(ABC):

    @abstractmethod
    def get_signal(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def get_window(self):
        pass