import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, data):
        self.data = data

    def run(self):
        for idx in range(len(self.data)):
            self.algo(self.data.iloc[:idx])
    
    def algo(self, data):
        # Stochastic Cross strategy
        # Premise : 
        # Entering spot : sto_k under low limit(25) touched before and currently rising and cross the sto_d
        # stop loss : sto_k cross the sto_d down again and value is under low limit(25), sto_k is falling during n tick
        # Exiting spot : 1. rising is ended middle of range
        #                2. sto_k upper high limit(80) touched and sto_k cross the sto_d down again
        pass
