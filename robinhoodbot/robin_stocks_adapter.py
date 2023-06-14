import robin_stocks.robinhood as rr
from functools import cache
from retry.api import retry_call

class rsa:
    @cache
    def get_fundamentals(stock):
        result = rr.get_fundamentals(stock)
        return result

    @cache
    def get_stock_historicals(stockTicker, interval='hour', span='week', bounds='regular', info=None):
        # print('Cache not found for ' + stockTicker)
        result = retry_call(rsa.try_get_stock_historicals, fargs=[stockTicker, interval, span, bounds, info], tries=5, backoff=5, delay=5)
        return result

    def try_get_stock_historicals(stockTicker, interval, span, bounds, info):
        # print('Trying ' + stockTicker)
        result = rr.get_stock_historicals(stockTicker, interval=interval, span=span, bounds=bounds, info=None)
        test_value = result[0]['close_price']
        return result

    @cache
    def get_instruments_by_symbols(stockTicker, info=None):
        result = retry_call(rsa.try_get_instruments_by_symbols, fargs=[stockTicker], fkwargs={"info": info}, tries=3, backoff=5, delay=5)
        return result

    def try_get_instruments_by_symbols(stockTicker, info=None):
        result = rr.get_instruments_by_symbols(stockTicker, info=info)
        test = len(result)
        return result

    @cache
    def get_instrument_by_url(url, info=None):
        result = retry_call(rsa.try_get_instrument_by_url, fargs=[url], fkwargs={"info": info}, tries=3, backoff=5, delay=5)
        return result

    def try_get_instrument_by_url(url, info=None):
        result = rr.get_instrument_by_url(url, info=info)
        test = len(result)
        return result

    @cache
    def get_watchlist_by_name(name):
        result = retry_call(rsa.try_get_watchlist_by_name, fargs=[name], tries=3, backoff=5, delay=5)
        return result
        
    def try_get_watchlist_by_name(name):
        result = rr.get_watchlist_by_name(name=name)
        return result

    @cache
    def get_all_open_stock_orders(url):
        result = retry_call(rsa.try_get_all_open_stock_orders, fargs=[url], tries=3, backoff=5, delay=5)
        return result
        
    def try_get_all_open_stock_orders(url):
        result = rr.get_all_open_stock_orders(url)
        return result

    @cache
    def get_all_open_stock_orders():
        result = retry_call(rsa.try_get_all_open_stock_orders, tries=3, backoff=5, delay=5)
        return result

    def try_get_all_open_stock_orders():
        result = rr.get_all_open_stock_orders()
        return result
    
    @cache
    def get_name_by_symbol(symbol):
        result = retry_call(rsa.try_get_name_by_symbol(symbol))
        return result

    def try_get_name_by_symbol(symbol):
        result = rr.get_name_by_symbol(symbol)
        return result
    
    @cache
    def get_name_by_url(url):
        result = retry_call(rsa.try_get_name_by_url(url))
        return result
    
    def try_get_name_by_url(url):
        result = rr.get_name_by_url(url)
        return result
    
    @cache
    def get_symbol_by_url(url):
        result = retry_call(rsa.try_get_symbol_by_url(url))
        return result
    
    def try_get_symbol_by_url(url):
        result = retry_call(rr.get_symbol_by_url(url))
        return result
    

