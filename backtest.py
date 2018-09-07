import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, candle, analysis):
        assert len(candle) == len(analysis), "Each length of candle data and analysis data are different."
        self.candle = candle
        self.analysis = analysis

    def run(self):
        for idx, current in self.candle.iterrows():
            algo(idx)
    
    def alog(self, idx):
        pass
