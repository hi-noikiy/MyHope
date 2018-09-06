import requests
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
import json
import traceback
import sqlite3
import logging
import logging.handlers
import time

'https://api.upbit.com/v1/candles/minutes/1?market=market' 'to : yyyy-MM-dd\'T\'HH:mm:ssXXX. 비워서 요청시 가장 최근 캔들' 'Maximum 200'

BASE_URL = 'https://api.upbit.com/v1/candles'
CURRENCYS = ['BTC','ETH','EOS','ONT','NEO']
#MIN_CANDLES = [1, 3, 5, 15, 10, 30, 60, 240]
MIN_CANDLES = [15, 30, 60, 240]
EMPTY_CANDLES = [0]
CYCLES = {
      'D': BASE_URL + '/days?market=KRW-{}&count={}&to={}'
    , 'W': BASE_URL + '/weeks?market=KRW-{}&count={}&to={}'
    , 'M': BASE_URL + '/months?market=KRW-{}&count={}&to={}'
    , 'T': BASE_URL + '/minutes/{}?market=KRW-{}&count={}&to={}'
}
TABLE_NAME = "{}_CANDLE"
CLEANUP = True

DB_PATH = "d:/workspace/Database/price.db"
con = None
cur = None
logger = None

# LOG : get logger
def get_logger():
    global logger

    if not logger:
        logger = logging.getLogger("mylogger")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s]{%(filename)s:%(lineno)d}-%(levelname)s - %(message)s')
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

    return logger

# DB : get db connection
def get_db_con():
    try:
        global con, cur
        if not con:
            con = sqlite3.connect(DB_PATH)

        if not cur:
            cur = con.cursor()
    except:
        traceback.print_exc()

    return con, cur

# DB : create table
def create_table(currency, cycle, period, cleanup=None):
    try:
        con, cur = get_db_con()
        sql = """CREATE TABLE IF NOT EXISTS {table} ( 
                'TIME' TEXT,
                'CYCLE' TEXT,
                'PERIOD' TEXT,
                'CLS_PRC' REAL,
                'OPEN_PRC' REAL,
                'HIGH_PRC' REAL,
                'LOW_PRC' REAL,
                'PREV_CLS_PRC' REAL,
                'VOLUME' REAL,
                CONSTRAINT {table}_PK PRIMARY KEY (TIME, CYCLE, PERIOD) )""".format(table=TABLE_NAME.format(currency))
        cur.execute(sql)

        if cleanup:
            del_candles(currency, cycle, period)
    except:
        traceback.print_exc()

# DB : delete data
def del_candles(currency, cycle, period):
    try:
        con, cur = get_db_con()
        cur.execute("DELETE FROM %s" % TABLE_NAME.format(currency))
        con.commit()
    except sqlite3.Error as e:
        logger.info(str(e))
    except:
        traceback.print_exc()

# DB : insert data
def insert_candles(currency, cycle, period, data):
    try:
        con, cur = get_db_con()
        if data:
            _data = DataFrame(data, columns=['candle_date_time_utc', 'trade_price', 'opening_price', 'high_price', 'low_price', 'prev_closing_price', 'candle_acc_trade_volume'])
            #_data.columns=["TIME", "CLS_PRC", "OPEN_PRC", "HIGH_PRC", "LOW_PRC", "PREV_CLS_PRC", "VOLUME"]
            _data = _data.set_index('candle_date_time_utc')
            #_data.to_sql(table_name, con, if_exists='append')
            sql = "INSERT OR IGNORE INTO {} VALUES(?, '{}', '{}', ?, ?, ?, ?, ?, ?)".format(TABLE_NAME.format(currency), cycle, period)
            cur.executemany(sql, _data.to_records())
            logger.info('# Affected Rows : ' + str(cur.rowcount) + ' / Data to Insert : ' + str(len(_data.to_records())))
    except:
        traceback.print_exc()

# request sending
def send_request(url):
    try:
        if url:
            res = requests.get(url)
            #return json.loads(res.text)
            return res.json()
    except:
        traceback.print_exc()


# upbit 로부터 currency 데이터 가져오기
def get_candles(cycle_type, currency, count, to, period=0):
    try:
        if period != 0:
            url = CYCLES[cycle_type].format(period, currency, count, to)
        else:
            url = CYCLES[cycle_type].format(currency, count, to)
        return send_request(url)
    except:
        traceback.print_exc()

# get all markets
def get_all_market():
    try:
        url = 'https://api.upbit.com/v1/market/all'
        return send_request(url)
    except:
        traceback.print_exc()
        
def main():
    try:
        con, cur = get_db_con()
        logger = get_logger()
        req_count = 100
        today = dt.datetime.now()
       
        for currency in CURRENCYS:
            logger.info("## Currency : " + currency)

            for cycle in CYCLES:
                logger.info("# Cycle : " + cycle)

                if cycle == 'T':
                    PERIODS = MIN_CANDLES
                else:
                    PERIODS = EMPTY_CANDLES

                for period in PERIODS:
                    logger.info("# Period : " + str(period))
                    create_table(currency, cycle, period, cleanup=False)
                    to = today

                    if cycle == 'T':
                        interval = dt.timedelta(minutes=req_count*period)
                    elif cycle == 'D':
                        interval = dt.timedelta(days=req_count)
                    elif cycle == 'W':
                        interval = dt.timedelta(weeks=req_count)
                    elif cycle == 'M':
                        interval = relativedelta(months=req_count)

                    while True:
                        time.sleep(3)
                        logger.info("# Time : " + to.strftime("%Y-%m-%d %H:%M:%S"))
                        data = get_candles(cycle, currency, req_count, to.strftime("%Y-%m-%d %H:%M:%S"), period)
                        if not data:
                            logger.info("! Data empty")
                            break

                        insert_candles(currency, cycle, period, data)

                        if len(data) < req_count:
                            logger.info("! Last data count : %d" % len(data))
                            break

                        if cur.rowcount == 0:
                            logger.info("! Duplicated")
                            break

                        to = to - interval                        
                con.commit()   
    except:
        traceback.print_exc()        
    finally:
        con.close()


if __name__ == '__main__':
    main()