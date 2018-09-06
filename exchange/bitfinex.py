import json
import mylogger
import requests


class Bitfinexpy():
    def __init__(self):
        self.valid_symbols = self._get_valid_symbols()

    def _get_valid_symbols(self):
        valid_symbols = []
        symbols = self.get_symbols()
        for symbol in symbols:
            valid_symbols.append(symbol)
        return valid_symbols

    def _get(self, url, params=None):
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            mylogger.error('get(%s) failed(%d)' % (url, resp.status_code))
            if len(resp.text) > 0:
                mylogger.error('resp: %s' % resp.text)
                raise Exception('requests.get() failed(%s)' % resp.text)
            raise Exception('requests.get() failed(status_code:%d)' % resp.status_code)
        return json.loads(resp.text)

    def _validate_symbol(self, symbol):
        if symbol.lower() not in self.valid_symbols:
            raise Exception('invalid symbol : %s' % symbol)

    ###############################################################
    # REST PUBLIC ENDPOINTS
    ###############################################################

    # https://bitfinex.readme.io/v1/reference#rest-public-ticker
    def get_ticker(self, symbol):
        URL = 'https://api.bitfinex.com/v1/ticker/'
        self._validate_symbol(symbol)
        return self._get(URL + symbol)

    # https://bitfinex.readme.io/v1/reference#rest-public-stats
    def get_stats(self, symbol):
        URL = 'https://api.bitfinex.com/v1/stats/'
        self._validate_symbol(symbol)
        return self._get(URL + symbol)

    # https://bitfinex.readme.io/v1/reference#rest-public-fundingbook
    def get_fundingbook(self, currency='USD', limit_bids=None, limit_asks=None):
        URL = 'https://api.bitfinex.com/v1/lendbook/'
        params = {}
        if limit_asks is not None:
            params['limit_asks'] = limit_asks
        if limit_bids is not None:
            params['limit_bids'] = limit_bids
        return self._get(URL + currency, params)

    # https://bitfinex.readme.io/v1/reference#rest-public-orderbook
    def get_orderbook(self, symbol, limit_bids=None, limit_asks=None, group=None):
        URL = 'https://api.bitfinex.com/v1/book/'
        self._validate_symbol(symbol)
        params = {}
        if limit_bids is not None:
            params['limit_bids'] = limit_bids
        if limit_asks is not None:
            params['limit_asks'] = limit_asks
        if group is not None:
            params['group'] = group
        return self._get(URL + symbol, params)

    # https://bitfinex.readme.io/v1/reference#rest-public-trades
    def get_trades(self, symbol, timestamp=None, limit_trades=None):
        URL = 'https://api.bitfinex.com/v1/trades/'
        self._validate_symbol(symbol)
        params = {}
        if timestamp is not None:
            params['timestamp'] = timestamp
        if limit_trades is not None:
            params['limit_trades'] = limit_trades
        return self._get(URL + symbol, params)

    # https://bitfinex.readme.io/v1/reference#rest-public-lends
    def get_lends(self, currency='USD', timestamp=None, limit_lends=None):
        URL = 'https://api.bitfinex.com/v1/lends/'
        params = {}
        if timestamp is not None:
            params['timestamp'] = timestamp
        if limit_lends is not None:
            params['limit_lends'] = limit_lends
        return self._get(URL + currency, params)

    # https://bitfinex.readme.io/v1/reference#rest-public-symbols
    def get_symbols(self):
        URL = 'https://api.bitfinex.com/v1/symbols'
        return self._get(URL)

    # https://bitfinex.readme.io/v1/reference#rest-public-symbol-details
    def get_symbol_details(self):
        URL = 'https://api.bitfinex.com/v1/symbols_details'
        return self._get(URL)