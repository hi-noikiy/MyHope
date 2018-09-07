import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.dates import date2num
from mpl_finance import candlestick_ohlc as candlestick
import datetime
import time
import logging
import logging.handlers
import traceback
import sqlite3
import talib as ta
import zigzag
from indicator import helper as indic
import peakutils

DB_PATH = "d:/workspace/Database/price.db"
con = None
logger = None

# Technical Analysis
SMA_FAST = 5
SMA_SLOW = 10
RSI_PERIOD = 14
RSI_AVG_PERIOD = 15
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K = 14
STOCH_D = 3
SIGNAL_TOL = 3
Y_AXIS_SIZE = 12


class MyContext():
    pass

# get logger
def get_logger():
    global logger
    if not logger:
        logger = logging.getLogger("mylogger")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s]{%(filename)s:%(funcName)s:%(lineno)d}-%(levelname)s - %(message)s')
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

    return logger

# get db connection
def get_db_con():
    try:
        global con
        if not con:
            con = sqlite3.connect(DB_PATH)
        return con
    except:
        traceback.print_exc()

# select data
def get_candles(context):
    try:
        con = get_db_con()
        sql = """SELECT TIME, CLS_PRC, OPEN_PRC, HIGH_PRC, LOW_PRC, PREV_CLS_PRC, VOLUME 
                 FROM {}_CANDLE 
                 WHERE CYCLE = '{}' 
                 AND PERIOD = '{}' 
                 AND TIME BETWEEN '{}' AND '{}'
                 ORDER BY TIME""".format(context.CURRENCY, context.CYCLE, context.CANDLE, context.TIME_FROM, context.TIME_TO)

        data = pd.read_sql(sql, con, index_col=['TIME'])
        data.index = pd.DatetimeIndex(data.index)
        return data
    except:
        traceback.print_exc()
    return None    

def initialize(context):
    context.CURRENCY = "BTC"
    context.CYCLE = "T"
    context.CANDLE = "240"
    context.TIME_FROM = "2018-01-01T00:00:00"
    context.TIME_TO = "2018-09-01T00:00:00"
    context.CAPTITAL_BASE = None
    context.SLIPPAGE = None
    context.COMMISION = None

    context.i = 0
    context.hold = False

    if not context.TIME_TO:
        datetime.datetime.today().strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(context.CURRENCY)

def handle_data(context, data):
    analysis = pd.DataFrame(index = data.index)    
    analysis['price'] = data['CLS_PRC']
    analysis['volume'] = data['VOLUME']
    analysis['sma_f'] = data.CLS_PRC.rolling(window=SMA_FAST,center=False).mean()
    analysis['sma_s'] = data.CLS_PRC.rolling(window=SMA_SLOW,center=False).mean()
    analysis['rsi'] = ta.RSI(data.CLS_PRC.values, RSI_PERIOD)
    analysis['sma_r'] = analysis.rsi.rolling(window=RSI_AVG_PERIOD,center=False).mean()
    #analysis['macd'], analysis['macdSignal'], analysis['macdHist'] = ta.MACD(data.CLS_PRC.as_matrix(), fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
    analysis['stoch_k'], analysis['stoch_d'] = ta.STOCH(data.HIGH_PRC.values, data.LOW_PRC.values, data.CLS_PRC.values, fastk_period=14, slowk_period=3, slowd_period=3)
    #analysis['stoch_k'], analysis['stoch_d'] = ta.STOCHRSI(data.CLS_PRC.values, timeperiod=14, fastk_period=3, fastd_period=3)
    # analysis['stoch_k'], analysis['stoch_d'] = indic.STO(data.HIGH_PRC, data.LOW_PRC, data.CLS_PRC, nK=14, nD=3, nS=3)

    analysis['VOL_CHG'] = data['VOLUME'].pct_change(fill_method='ffill')
    analysis['CDL3INSIDE'] = ta.CDL3INSIDE(data.OPEN_PRC, data.HIGH_PRC, data.LOW_PRC, data.CLS_PRC)
    analysis['ZIGZAG'] = zigzag.peak_valley_pivots(np.array(data['CLS_PRC']), 0.01, -0.01)
    analysis['ZIGZAG_STOCH'] = zigzag.peak_valley_pivots(np.array(analysis['stoch_k']), 0.01, -0.01)
    
    #analysis['sma'] = np.where(analysis.sma_f > analysis.sma_s, 1, 0)
    #analysis['macd_test'] = np.where((analysis.macd > analysis.macdSignal), 1, 0)
    #analysis['stoch_k_test'] = np.where((analysis.stoch_k < 50) & (analysis.stoch_k > analysis.stoch_k.shift(1)), 1, 0)
    #analysis['rsi_test'] = np.where((analysis.rsi < 50) & (analysis.rsi > analysis.rsi.shift(1)), 1, 0)

    #print(analysis)
    return analysis
    
def analyze(context, data, analysis):
    ticker = context.CURRENCY
    date_from = '2018-08-01'
    
    # Data for matplotlib finance plot
    data = data[data.index > date_from]
    data['Date'] = data.index.map(mdates.date2num)
    ohlc = data[['Date','OPEN_PRC','HIGH_PRC','LOW_PRC','CLS_PRC']]

    analysis = analysis[analysis.index > date_from]
    analysis.index = date2num(analysis.index.to_pydatetime())

    # Prepare figure and plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios':[6,2,2]}, sharex=True)
    fig.set_size_inches(15,30)

    # Set ax1
    ax1.set_ylabel(ticker, size=20)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.grid(True)

    # Draw Candlestick
    candlestick(ax1, ohlc.values, width=0.1, colorup='g', colordown='r', alpha=1)

    # Plot ZigZag
    ax1.plot(analysis.ix[analysis.ZIGZAG != 0].index, analysis.price[analysis.ZIGZAG != 0], 'k-')
    ax1.scatter(analysis.ix[analysis.ZIGZAG == 1].index, analysis.price[analysis.ZIGZAG == 1], color='g')
    ax1.scatter(analysis.ix[analysis.ZIGZAG == -1].index, analysis.price[analysis.ZIGZAG == -1], color='r')

    # Draw Moving Averages
    analysis.sma_f.plot(ax=ax1, c='r', label='SMA'+str(SMA_FAST))
    analysis.sma_s.plot(ax=ax1, c='g', label='SMA'+str(SMA_SLOW))
    #ax1.plot(analysis.ix[analysis.sma == 1].index, analysis.sma_f[analysis.sma == 1], '^')

    # Draw 3Inside
    ax1.plot(analysis.ix[analysis.CDL3INSIDE < 0].index, analysis.price[analysis.CDL3INSIDE < 0], 'v')
    ax1.plot(analysis.ix[analysis.CDL3INSIDE > 0].index, analysis.price[analysis.CDL3INSIDE > 0], '^')

    # Draw Support and Resistance Line
    support, resistance = indic.findsnp(data['CLS_PRC'], 10)
    for value in support+resistance:
        ax1.hlines(y=value, xmin=analysis.index.min(), xmax=analysis.index.max(), linewidth=1, color='r', linestyles='--') 

    # Draw Peak and Valley
    # indices = peakutils.indexes(analysis['price'], valley=False, thres=0.3, min_dist=1, thres_abs=False)
    # peak1 = analysis.iloc[indices]

    # peak_indices = indic.detect_peaks(analysis['price'], edge='rising', valley=False, threshold=0.3)
    # valley_indices = indic.detect_peaks(analysis['price'], edge='falling', valley=True, threshold=0.3)
    # peaks = analysis.iloc[peak_indices]
    # valleys = analysis.iloc[valley_indices]
    # peaknvalley_indices = peak_indices.tolist()+valley_indices.tolist()
    # peaknvalley_indices.sort()
    # peaknvalley = analysis.iloc[peaknvalley_indices]

    # ax1.plot(peaknvalley.index, peaknvalley.price, 'k-')    
    # ax1.scatter(peaks.index, peaks.price, color='y')
    # ax1.scatter(valleys.index, valleys.price, color='r')
    # ax1.scatter(peak1.index, peak1.price, color='b', marker='^')
    
    #RSI
    ax2.set_ylabel('RSI', size=Y_AXIS_SIZE)
    analysis.rsi.plot(ax=ax2, c='g', label = 'Period: ' + str(RSI_PERIOD))
    analysis.sma_r.plot(ax=ax2, c='r', label = 'MA: ' + str(RSI_AVG_PERIOD))
    ax2.axhline(y=30, c='b')
    ax2.axhline(y=50, c='black')
    ax2.axhline(y=70, c='b')
    ax2.set_ylim([0,100])
    ax2.legend(loc='upper left')

    '''# Volume
    ax2.bar(analysis.index, analysis.volume, 0.2, align='center', color='b', label='Volume')
    ax2.legend(loc='upper left')
    ax2.grid(True)'''

    '''# Volume Change Rate
    ax3.set_ylabel('VOL_CHG', size=Y_AXIS_SIZE)
    analysis.VOL_CHG.plot(ax=ax3, c='g', label = 'VOL_CHG')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels) 
    ax3.grid()'''

    '''
    # Draw MACD computed with Talib
    ax3.set_ylabel('MACD: '+ str(MACD_FAST) + ', ' + str(MACD_SLOW) + ', ' + str(MACD_SIGNAL), size=Y_AXIS_SIZE)
    analysis.macd.plot(ax=ax3, color='b', label='Macd')
    analysis.macdSignal.plot(ax=ax3, color='g', label='Signal')
    analysis.macdHist.plot(ax=ax3, color='r', label='Hist')
    ax3.axhline(0, lw=2, color='0')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)
    ax3.grid()'''

    # Stochastic plot
    # ax2.set_ylabel('Stoch1', size=Y_AXIS_SIZE)
    # analysis.STO_K.plot(ax=ax2, label='STO_K', color='b')
    # analysis.STO_D.plot(ax=ax2, label='STO_D', color='r')
    # ax2.legend(loc='upper left')
    # ax2.axhline(y=20, c='m')
    # ax2.axhline(y=80, c='m')
    # ax2.grid()

    # Stochastic plot
    ax3.set_ylabel('Stoch', size=Y_AXIS_SIZE)
    analysis.stoch_k.plot(ax=ax3, label='stoch_k', color='b')
    analysis.stoch_d.plot(ax=ax3, label='stoch_d', color='r')
    ax3.legend(loc='upper left')
    ax3.axhline(y=20, c='m')
    ax3.axhline(y=80, c='m')
    ax3.grid()

    # Plot Peak and Valley for Stochastic
    peak_indices = indic.detect_peaks(analysis['stoch_k'], edge='rising', valley=False, threshold=0.3)
    valley_indices = indic.detect_peaks(analysis['stoch_k'], edge='falling', valley=True, threshold=0.3)
    peaks = analysis.iloc[peak_indices]
    valleys = analysis.iloc[valley_indices]
    ax3.scatter(peaks.index, peaks.stoch_k, color='y')
    ax3.scatter(valleys.index, valleys.stoch_k, color='r')

    plt.show()

def main():
    logger = get_logger()
    context = MyContext()

    #initialize
    initialize(context)

    data = get_candles(context)

    if data is None:
        logger.info("!! Data is empty !!")
        return

    analysis = handle_data(context, data)
    if analysis is None:
        logger.info("# Terminate handle data : %d", context.i)
    
    analyze(context, data, analysis)

if __name__ == '__main__':
    main()

