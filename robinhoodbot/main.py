import robin_stocks.robinhood as rr
import pandas as pd
import numpy as np
import ta as t
import smtplib
import ssl
import certifi
import sys
import datetime
import traceback
import time
import os
import json
from retry.api import retry_call
from functools import cache
from pandas.plotting import register_matplotlib_converters
from misc import *
from tradingstats import *
from config import *
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from scipy.stats import linregress
from pyotp import TOTP as otp
from robin_stocks_adapter import rsa, api_tracker, clear_all_caches, cache_stats

# SMS notification state
sms_enabled = False

# JSON Logger for monitor output
class JSONLogger:
    def __init__(self, log_file="log.json"):
        self.log_file = log_file
        self.session_id = str(pd.Timestamp("now"))
        self.logs = []
    
    def log(self, event_type, message, data=None):
        """Log an event to the JSON log file"""
        entry = {
            "timestamp": str(pd.Timestamp("now")),
            "session_id": self.session_id,
            "event_type": event_type,
            "message": message
        }
        if data:
            entry["data"] = data
        
        self.logs.append(entry)
        self._write_to_file(entry)
    
    def _write_to_file(self, entry):
        """Append entry to the log file"""
        try:
            # Read existing logs
            try:
                with open(self.log_file, 'r') as f:
                    all_logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_logs = []
            
            # Append new entry
            all_logs.append(entry)
            
            # Keep only last 10000 entries to prevent file from growing too large
            if len(all_logs) > 10000:
                all_logs = all_logs[-10000:]
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(all_logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

# Global logger instance
json_logger = JSONLogger()


# Console Logger - captures all print output to console_log.json
class ConsoleLogger:
    def __init__(self, log_file="console_log.json"):
        self.log_file = log_file
        self.session_id = str(pd.Timestamp("now"))
        self.original_stdout = sys.stdout
        self.buffer = ""
    
    def write(self, message):
        """Capture print output and log to JSON"""
        # Write to original stdout
        self.original_stdout.write(message)
        
        # Skip empty or whitespace-only messages
        if message.strip():
            self._log_to_file(message.strip())
    
    def flush(self):
        """Flush the output"""
        self.original_stdout.flush()
    
    def _log_to_file(self, message):
        """Append message to the console log file"""
        try:
            entry = {
                "timestamp": str(pd.Timestamp("now")),
                "session_id": self.session_id,
                "message": message
            }
            
            # Read existing logs
            try:
                with open(self.log_file, 'r') as f:
                    all_logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_logs = []
            
            # Append new entry
            all_logs.append(entry)
            
            # Keep only last 10000 entries to prevent file from growing too large
            if len(all_logs) > 10000:
                all_logs = all_logs[-10000:]
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(all_logs, f, indent=2)
        except Exception as e:
            # Use original stdout to avoid recursion
            self.original_stdout.write(f"Warning: Could not write to console log file: {e}\n")

# Redirect stdout to capture all print output
console_logger = ConsoleLogger()
sys.stdout = console_logger


# US Stock Market Holidays
def get_us_market_holidays(year: int) -> set:
    """
    Returns a set of US stock market holidays for a given year.
    NYSE/NASDAQ are closed on these days.
    """
    holidays = set()
    
    # New Year's Day (Jan 1, or observed on nearest weekday)
    new_years = datetime.date(year, 1, 1)
    if new_years.weekday() == 5:  # Saturday
        holidays.add(datetime.date(year - 1, 12, 31))  # Observed Friday before
    elif new_years.weekday() == 6:  # Sunday
        holidays.add(datetime.date(year, 1, 2))  # Observed Monday after
    else:
        holidays.add(new_years)
    
    # Martin Luther King Jr. Day (3rd Monday of January)
    jan_first = datetime.date(year, 1, 1)
    days_to_monday = (7 - jan_first.weekday()) % 7
    first_monday = jan_first + datetime.timedelta(days=days_to_monday)
    mlk_day = first_monday + datetime.timedelta(weeks=2)
    holidays.add(mlk_day)
    
    # Presidents Day (3rd Monday of February)
    feb_first = datetime.date(year, 2, 1)
    days_to_monday = (7 - feb_first.weekday()) % 7
    first_monday = feb_first + datetime.timedelta(days=days_to_monday)
    presidents_day = first_monday + datetime.timedelta(weeks=2)
    holidays.add(presidents_day)
    
    # Good Friday (Friday before Easter Sunday)
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter = datetime.date(year, month, day)
    good_friday = easter - datetime.timedelta(days=2)
    holidays.add(good_friday)
    
    # Memorial Day (last Monday of May)
    may_last = datetime.date(year, 5, 31)
    days_back = (may_last.weekday() - 0) % 7
    memorial_day = may_last - datetime.timedelta(days=days_back)
    holidays.add(memorial_day)
    
    # Juneteenth (June 19, or observed)
    juneteenth = datetime.date(year, 6, 19)
    if juneteenth.weekday() == 5:  # Saturday
        holidays.add(datetime.date(year, 6, 18))
    elif juneteenth.weekday() == 6:  # Sunday
        holidays.add(datetime.date(year, 6, 20))
    else:
        holidays.add(juneteenth)
    
    # Independence Day (July 4, or observed)
    july_fourth = datetime.date(year, 7, 4)
    if july_fourth.weekday() == 5:  # Saturday
        holidays.add(datetime.date(year, 7, 3))
    elif july_fourth.weekday() == 6:  # Sunday
        holidays.add(datetime.date(year, 7, 5))
    else:
        holidays.add(july_fourth)
    
    # Labor Day (1st Monday of September)
    sep_first = datetime.date(year, 9, 1)
    days_to_monday = (7 - sep_first.weekday()) % 7
    labor_day = sep_first + datetime.timedelta(days=days_to_monday)
    holidays.add(labor_day)
    
    # Thanksgiving Day (4th Thursday of November)
    nov_first = datetime.date(year, 11, 1)
    days_to_thursday = (3 - nov_first.weekday()) % 7
    first_thursday = nov_first + datetime.timedelta(days=days_to_thursday)
    thanksgiving = first_thursday + datetime.timedelta(weeks=3)
    holidays.add(thanksgiving)
    
    # Christmas Day (Dec 25, or observed)
    christmas = datetime.date(year, 12, 25)
    if christmas.weekday() == 5:  # Saturday
        holidays.add(datetime.date(year, 12, 24))
    elif christmas.weekday() == 6:  # Sunday
        holidays.add(datetime.date(year, 12, 26))
    else:
        holidays.add(christmas)
    
    return holidays


def is_trading_day(check_date) -> bool:
    """Check if a date is a valid trading day (not weekend, not holiday)."""
    if hasattr(check_date, 'date'):
        check_date = check_date.date()
    elif not isinstance(check_date, datetime.date):
        check_date = pd.to_datetime(check_date).date()
    
    # Skip weekends
    if check_date.weekday() >= 5:
        return False
    
    # Skip holidays
    holidays = get_us_market_holidays(check_date.year)
    if check_date in holidays:
        return False
    
    return True


def is_market_holiday_today() -> bool:
    """Check if today is a market holiday."""
    today = datetime.date.today()
    holidays = get_us_market_holidays(today.year)
    return today in holidays


# Safe divide by zero division function
def safe_division(n, d):
    return n / d if d else 0

# https://stackoverflow.com/questions/19390267/python-3-smtplib-exception-ssl-wrong-version-number-logging-in-to-outlook
def login_to_sms():
    global sms_gateway
    global server
    global sms_enabled
    
    # Log in to gmail with timeout
    SMS_TIMEOUT = 30  # seconds
    try:
        sms_gateway = rh_phone + '@' + rh_company_url  # Phone number to send SMS
        context = ssl.create_default_context(cafile=certifi.where())
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=SMS_TIMEOUT)  # Gmail SMTP server
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(rh_email, rh_mail_password)
        sms_enabled = True
        print("SMS notifications enabled")
    except Exception as e:
        sms_enabled = False
        print(f"⚠️  SMS login failed: {e}")
        print("Continuing without SMS notifications...")


def send_text(message):
    global sms_enabled
    if not sms_enabled:
        print(f"SMS disabled - would have sent: {message[:50]}...")
        return
    try:
        login_to_sms()
        msg = MIMEMultipart()
        msg['From'] = rh_email
        msg['To'] = sms_gateway
        if debug:
            msg['Subject'] = 'DEBUG Robinhood Stocks'
        else:
            msg['Subject'] = 'Robinhood Stocks'
        msg.attach(MIMEText(message+'**', 'plain'))
        sms = msg.as_string()
        server.sendmail(rh_email, sms_gateway, sms)
    except Exception as e:
        print(f"⚠️  SMS send failed: {e}")

def isInExclusionList(symbol):
    """
    Returns true if the symbol is in the exclusion list.
    """
    result = False
    if use_exclusion_watchlist:
        exclusion_list = rsa.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
    for exclusion_item in exclusion_list['results']:
            if exclusion_item['symbol'] == symbol:
                result = True
                return result
    return result


def get_watchlist_symbols(exclude_from_exclusion_list):
    """
    Args: get_watchlist_symbols(exclude_from_exclusionlist): True to exclude symbols from a specified exclusion list watchlist.
    Returns: the symbol for each stock in your watchlist as a list of strings
    """
    exclusion_list = []
    symbols = []
    list = rsa.get_watchlist_by_name(name=watch_list_name)
    # Remove any exclusions.
    if exclude_from_exclusion_list:
        exclusion_list = rsa.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
    skip = False
    for item in list['results']:
        if exclude_from_exclusion_list:
            for exclusion_item in exclusion_list['results']:
                    if exclusion_item['symbol'] == item['symbol']:
                        skip = True
        if skip:
            skip = False
            continue

        symbol = item['symbol']
        symbols.append(symbol)
    x = np.array(symbols)
    symbols = np.unique(x).tolist()
    return symbols


def get_portfolio_symbols():
    """
    Returns: the symbol for each stock in your portfolio as a list of strings
    """
    symbols = []
    holdings_data = rr.get_open_stock_positions()
    for item in holdings_data:
        if not item:
            continue
        instrument_data = rsa.get_instrument_by_url(item.get('instrument'))
        symbol = instrument_data['symbol']
        symbols.append(symbol)
    return symbols

def remove_watchlist_symbols(watchlist_symbols):
    """ Removes all of the symbols from the watchlist.

    Args:
        watchlist_symbols(array): array with all of the symbols in the watchlist.

    Returns:
        Result of the delete request.
    """
    # Night - reset watchlist on Friday (or Thursday if Friday is a holiday)
    begin_time = datetime.time(21, 00)
    end_time = datetime.time(23, 00)
    timenow = datetime.datetime.now().time()
    today = datetime.date.today()
    
    # Check if today is Friday, or Thursday if Friday is a holiday
    is_reset_day = False
    if today.weekday() == 4:  # Friday
        is_reset_day = True
    elif today.weekday() == 3:  # Thursday - check if Friday is a holiday
        friday = today + datetime.timedelta(days=1)
        if not is_trading_day(friday):
            is_reset_day = True
          
    if(timenow >= begin_time and timenow < end_time and is_reset_day):
        print("----- Removing all of this weeks stocks from watchlist -----")
        result = rr.delete_symbols_from_watchlist(watchlist_symbols, name = watch_list_name)
        return result

def get_position_creation_date(symbol, holdings_data):
    """Returns the time at which we bought a certain stock in our portfolio

    Args:
        symbol(str): Symbol of the stock that we are trying to figure out when it was bought
        holdings_data(dict): dict returned by rr.get_current_positions()

    Returns:
        A string containing the date and time the stock was bought, or "Not found" otherwise
    """
    instrument = rsa.get_instruments_by_symbols(symbol)
    url = instrument[0].get('url')
    for dict in holdings_data:
        if(dict.get('instrument') == url):
            return dict.get('created_at')
    return "Not found"


def save_trade_reason(symbol, reason, action="buy", holdings_data=None, price=None, quantity=None, equity=None):
    """ Saves the reason why a stock was bought or sold to a JSON file.
    
    Args:
        symbol(str): Symbol of the stock
        reason(str): Reason for the trade
        action(str): Either "buy" or "sell"
        holdings_data(dict): Optional holdings data dict for sells (from get_modified_holdings())
        price(float): Optional price for buys
        quantity(int): Optional quantity for buys
        equity(float): Optional total equity for buys
    """
    buy_reasons_file = "buy_reasons.json"
    try:
        with open(buy_reasons_file, 'r') as f:
            buy_reasons_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        buy_reasons_data = {}
    
    trade_entry = {
        'reason': reason,
        'action': action,
        'timestamp': str(pd.Timestamp("now")),
        'version': version,
        'config': {
            'debug': debug,
            'price_cap': price_cap,
            'use_price_cap': use_price_cap,
            'min_volume': min_volume,
            'min_market_cap': min_market_cap,
            'purchase_limit_percentage': purchase_limit_percentage,
            'use_purchase_limit_percentage': use_purchase_limit_percentage,
            'investing': investing,
            'premium_account': premium_account,
            'market_tag_for_report': market_tag_for_report
        }
    }
    
    # Add holdings data for sells
    if holdings_data and symbol in holdings_data:
        stock_data = holdings_data[symbol]
        trade_entry['price'] = stock_data.get('price')
        trade_entry['quantity'] = stock_data.get('quantity')
        trade_entry['average_buy_price'] = stock_data.get('average_buy_price')
        trade_entry['equity'] = stock_data.get('equity')
        trade_entry['percent_change'] = stock_data.get('percent_change')
        trade_entry['intraday_percent_change'] = stock_data.get('intraday_percent_change')
        trade_entry['equity_change'] = stock_data.get('equity_change')
        trade_entry['type'] = stock_data.get('type')
        trade_entry['name'] = stock_data.get('name')
        trade_entry['id'] = stock_data.get('id')
        trade_entry['pe_ratio'] = stock_data.get('pe_ratio')
        trade_entry['percentage'] = stock_data.get('percentage')
        trade_entry['bought_at'] = stock_data.get('bought_at')
    # Add buy data
    elif action == "buy":
        if price is not None:
            trade_entry['price'] = str(price)
        if quantity is not None:
            trade_entry['quantity'] = str(quantity)
        if equity is not None:
            trade_entry['equity'] = str(equity)
    
    buy_reasons_data[symbol] = trade_entry
    
    with open(buy_reasons_file, 'w') as f:
        json.dump(buy_reasons_data, f, indent=4)


# Keep backward compatibility
def save_buy_reason(symbol, reason, price=None, quantity=None, equity=None):
    """ Legacy function - calls save_trade_reason with action='buy' """
    save_trade_reason(symbol, reason, action="buy", price=price, quantity=quantity, equity=equity)


def get_trade_reason(symbol):
    """ Retrieves the full trade data for a stock.
    
    Args:
        symbol(str): Symbol of the stock
        
    Returns:
        dict: The trade data including reason, action, timestamp, version, or None if not found
    """
    buy_reasons_file = "buy_reasons.json"
    try:
        with open(buy_reasons_file, 'r') as f:
            buy_reasons_data = json.load(f)
        if symbol in buy_reasons_data:
            return buy_reasons_data[symbol]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


# Keep backward compatibility
def get_buy_reason(symbol):
    """ Legacy function - returns just the reason string """
    trade_data = get_trade_reason(symbol)
    if trade_data:
        return trade_data.get('reason')
    return None


def get_modified_holdings():
    """ Retrieves the same dictionary as rr.build_holdings, but includes data about
        when the stock was purchased, which is useful for the read_trade_history() method
        in tradingstats.py

    Returns:
        the same dict from rr.build_holdings, but with an extra key-value pair for each
        position you have, which is 'bought_at': (the time the stock was purchased)
    """
    holdings = rr.build_holdings()
    holdings_data = rr.get_open_stock_positions()
    for symbol, dict in holdings.items():
        bought_at = get_position_creation_date(symbol, holdings_data)
        bought_at = str(pd.to_datetime(bought_at))
        holdings[symbol].update({'bought_at': bought_at})
        # Add buy reason if available
        buy_reason = get_buy_reason(symbol)
        if buy_reason:
            holdings[symbol].update({'buy_reason': buy_reason})
    return holdings


def get_sma_proximity(stockTicker, n1=None, n2=None):
    """Get the proximity of the short-term SMA to the long-term SMA for death cross analysis.
    
    Args:
        stockTicker(str): Symbol of the stock
        n1(int): Short-term SMA period (default from config.short_sma)
        n2(int): Long-term SMA period (default from config.long_sma)
    
    Returns:
        dict with sma_short, sma_long, gap, gap_pct, and status
    """
    if n1 is None:
        n1 = short_sma
    if n2 is None:
        n2 = long_sma
    try:
        history = rsa.get_stock_historicals(stockTicker, interval='hour', span='3month', bounds='regular')
        if not history or len(history) < n2:
            return None
        
        closingPrices = [float(h['close_price']) for h in history]
        price = pd.Series(closingPrices)
        sma_short = t.volatility.bollinger_mavg(price, n=int(n1), fillna=False)
        sma_long = t.volatility.bollinger_mavg(price, n=int(n2), fillna=False)
        
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        gap = current_sma_short - current_sma_long
        gap_pct = (gap / current_sma_long) * 100 if current_sma_long > 0 else 0
        
        return {
            'sma_short': round(current_sma_short, 2),
            'sma_long': round(current_sma_long, 2),
            'gap': round(gap, 2),
            'gap_pct': round(gap_pct, 2),
            'status': 'above' if gap > 0 else 'DEATH_CROSS'
        }
    except Exception as e:
        return None


def get_last_crossing(df, days, symbol="", direction=""):
    """Searches for a crossing between two indicators for a given stock

    Args:
        df(pandas.core.frame.DataFrame): Pandas dataframe with columns containing the stock's prices, both indicators, and the dates
        days(int): Specifies the maximum number of days that the cross can occur by
        symbol(str): Symbol of the stock we're querying. Optional, used for printing purposes
        direction(str): "above" if we are searching for an upwards cross, "below" if we are searching for a downwaords cross. Optional, used for printing purposes

    Returns:
        1 if the short-term indicator crosses above the long-term one
        0 if there is no cross between the indicators
        -1 if the short-term indicator crosses below the long-term one
    """
    prices = df.loc[:, "Price"]
    shortTerm = df.loc[:, "Indicator1"]
    LongTerm = df.loc[:, "Indicator2"]
    dates = df.loc[:, "Dates"]
    lastIndex = prices.size - 1
    index = lastIndex
    found = index
    recentDiff = (shortTerm.at[index] - LongTerm.at[index]) >= 0
    if((direction == "above" and not recentDiff) or (direction == "below" and recentDiff)):
        return 0, 0, 0, None
    index -= 1
    # Count trading days (exclude weekends and holidays) instead of calendar days
    trading_days_checked = 0
    prev_date = None
    while(index >= 0 and found == lastIndex and not np.isnan(shortTerm.at[index]) and not np.isnan(LongTerm.at[index])
          and trading_days_checked <= days):
        # Track unique trading days (weekdays and non-holidays only)
        current_date = dates.at[index].date() if hasattr(dates.at[index], 'date') else pd.to_datetime(dates.at[index]).date()
        if current_date != prev_date and is_trading_day(current_date):
            trading_days_checked += 1
            prev_date = current_date
        if(recentDiff):
            if((shortTerm.at[index] - LongTerm.at[index]) < 0):
                found = index
        else:
            if((shortTerm.at[index] - LongTerm.at[index]) > 0):
                found = index
        index -= 1
    if(found != lastIndex):
        if((direction == "above" and recentDiff) or (direction == "below" and not recentDiff)):
            last_crossing_report = symbol + ": Short SMA crossed" + (" ABOVE " if recentDiff else " BELOW ") + "Long SMA at " + str(dates.at[found]) + ", which was " + str(
                pd.Timestamp("now", tz='UTC') - dates.at[found]) + " ago", ", price at cross: " + str(prices.at[found]) + ", current price: " + str(prices.at[lastIndex])

            print(last_crossing_report)
        return (1 if recentDiff else -1), prices.at[found], prices.at[lastIndex], dates.at[found]
    else:
        return 0, 0, 0, None


def five_year_check(stockTicker):
    """Figure out if a stock has risen or been created within the last five years.

    Args:
        stockTicker(str): Symbol of the stock we're querying

    Returns:
        True if the stock's current price is higher than it was five years ago, or the stock IPO'd within the last five years
        False otherwise
    """
    instrument = rsa.get_instruments_by_symbols(stockTicker)
    # instrument = retry_call(rsa.get_instruments_by_symbols, fargs=[stockTicker], tries=3, backoff=5, delay=5)

    if(len(instrument) == 0):
        return False

    list_date = instrument[0].get("list_date")
    # If there is no list date then assume that the stocks list date data
    # is just missing i.e. NNOX
    if list_date == None:
        return True
    if ((pd.Timestamp("now") - pd.to_datetime(list_date)) < pd.Timedelta(str(365*5) + " days")):
        return True
    fiveyear = rsa.get_stock_historicals(stockTicker, interval='day', span='5year', bounds='regular')

    closingPrices = []
    for item in fiveyear:
        closingPrices.append(float(item['close_price']))

    # If fiveyear returns an empty list then the stock is either new or has no data.
    # Either way it should fail the five year check.
    if(len(closingPrices) == 0):
        return False
    
    recent_price = closingPrices[len(closingPrices) - 1]
    oldest_price = closingPrices[0]

    return (recent_price > oldest_price)


def golden_cross(stockTicker, n1, n2, days, direction=""):
    """Determine if a golden/death cross has occured for a specified stock in the last X trading days

    Args:
        stockTicker(str): Symbol of the stock we're querying
        n1(int): Specifies the short-term indicator as an X-day moving average.
        n2(int): Specifies the long-term indicator as an X-day moving average.
                 (n1 should be smaller than n2 to produce meaningful results, e.g n1=50, n2=200)
        days(int): Specifies the maximum number of days that the cross can occur by e.g. 10
        direction(str): "above" if we are searching for an upwards cross, "below" if we are searching for a downwaords cross. Optional, used for printing purposes

    Returns:
        1, price if the short-term indicator crosses above the long-term one and the price at cross.
        0 if there is no cross between the indicators
        -1, price if the short-term indicator crosses below the long-term one and price at cross
        False if direction == "above" and five_year_check(stockTicker) returns False, meaning that we're considering whether to
            buy the stock but it hasn't risen overall in the last five years, suggesting it contains fundamental issues
    """
    """ Apparently 5 year historicals are no longer available with hourly intervals.  Only with day intervals now.
    """
    # yearCheck = five_year_check(stockTicker)
    yearCheck = retry_call(five_year_check, fargs=[stockTicker], tries=3, backoff=5, delay=5)

    if(direction == "above" and not yearCheck):
        return False, 0, 0, 0, None, None, None, None

    # print('About to try ' + stockTicker)
    history = rsa.get_stock_historicals(stockTicker, interval='hour', span='3month', bounds='regular')
    # history = retry_call(rsa.get_stock_historicals, fargs=[stockTicker], fkwargs={"interval": "hour","span": "3month","bounds": "regular"}, tries=3, backoff=5, delay=5)
    
    closingPrices = []
    dates = []
    for history_item in history:
        closingPrices.append(float(history_item['close_price']))
        dates.append(history_item['begins_at'])

    price = pd.Series(closingPrices)
    dates = pd.Series(dates)
    dates = pd.to_datetime(dates)
    sma1 = t.volatility.bollinger_mavg(price, n=int(n1), fillna=False)
    sma2 = t.volatility.bollinger_mavg(price, n=int(n2), fillna=False)
    # sma3 = t.volatility.bollinger_mavg(price, n=21, fillna=False)
    # sma4 = t.volatility.bollinger_mavg(price, n=50, fillna=False)
    series = [price.rename("Price"), sma1.rename(
        "Indicator1"), sma2.rename("Indicator2"), dates.rename("Dates")]
    df = pd.concat(series, axis=1)
    cross = get_last_crossing(df, days, symbol=stockTicker, direction=direction)
    
    if(plot):
        # show_plot(price, sma1, sma2, sma3, sma4, dates, symbol=stockTicker,
        #           label1=str(n1)+" day SMA", label2=str(n2)+" day SMA", label3="21 day SMA", label4="50 day SMA")
        show_plot(price, sma1, sma2, dates, symbol=stockTicker,
                  label1=str(n1)+" day SMA", label2=str(n2)+" day SMA")
    # Return: cross_signal, price_at_cross, current_price, price_5hr_ago, current_sma20, current_sma50, current_price, cross_date
    current_sma20 = float(sma1.iloc[-1]) if not pd.isna(sma1.iloc[-1]) else None
    current_sma50 = float(sma2.iloc[-1]) if not pd.isna(sma2.iloc[-1]) else None
    current_price = float(price.iloc[-1]) if len(price) > 0 else None
    cross_date = str(cross[3]) if cross[3] is not None else None
    price_5hr_ago = float(history[len(history)-5]['close_price']) if len(history) >= 5 else current_price
    return cross[0], cross[1], cross[2], price_5hr_ago, current_sma20, current_sma50, current_price, cross_date


def sell_holdings(symbol, holdings_data):
    """ Place an order to sell all holdings of a stock.

    Args:
        symbol(str): Symbol of the stock we want to sell
        holdings_data(dict): dict obtained from get_modified_holdings() method
    """
    shares_owned = int(float(holdings_data[symbol].get("quantity")))
    rr.order_sell_market(symbol, shares_owned,None, 'gfd')
    print("####### Selling " + str(shares_owned) +
          " shares of " + symbol + " #######")
    send_text("SELL: \nSelling " + str(shares_owned) + " shares of " + symbol)

def buy_holdings(potential_buys, cash, equity, holdings_data_length, buy_reasons=None):
    """ Places orders to buy holdings of stocks. This method will try to order
        an appropriate amount of shares such that your holdings of the stock will
        roughly match the average for the rest of your portfoilio. If the share
        price is too high considering the rest of your holdings and the amount of
        buying power in your account, it will not order any shares.
    Args:
        potential_buys(list): List of strings, the strings are the symbols of stocks we want to buy
        cash(str): The amount of cash available to buy stock
        equity(str): Entire amount of invested and uninvested funds
        holdings_data_length(dict): The length of the dict obtained from rr.build_holdings() or get_modified_holdings() method
        buy_reasons(dict): Optional dictionary mapping symbols to their buy reasons (e.g., {"AAPL": "golden_cross"})
    """
    cash = float(cash)
    equity = float(equity)
    portfolio_value = equity - cash
    ideal_position_size = (safe_division(portfolio_value, holdings_data_length)+cash/len(potential_buys))/(2 * len(potential_buys))
    prices = rr.get_latest_price(potential_buys)
    for i in range(0, len(potential_buys)):
        stock_price = float(prices[i])
        if ((stock_price * int(ideal_position_size/stock_price)) > cash):
            num_shares = int(ideal_position_size/stock_price)
            output = "Tried buying " + str(num_shares) + " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " but with only ${:.2f}".format(cash) + " in cash not enough to make this purchase."
            print(output)
            if (len(potential_buys) > 1):
                ideal_position_size = (safe_division(portfolio_value, holdings_data_length)+cash/(len(potential_buys)-1))/(2 * (len(potential_buys)-1))
            continue
        elif ((stock_price * int(ideal_position_size*1.5/stock_price)) > cash):
            num_shares = int(ideal_position_size*1.5/stock_price)
            output = "Tried buying " + str(num_shares) + " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " but with only ${:.2f}".format(cash) + " in cash not enough to make this purchase."
            print(output)
            if (len(potential_buys) > 1):
                ideal_position_size = (safe_division(portfolio_value, holdings_data_length)+cash/(len(potential_buys)-1))/(2 * (len(potential_buys)-1))
            continue
        elif(ideal_position_size < stock_price < ideal_position_size*1.5):
            num_shares = int(ideal_position_size*1.5/stock_price)
        elif (stock_price < ideal_position_size):
            num_shares = int(ideal_position_size/stock_price)
        else:
            num_shares = float(ideal_position_size*1.5/stock_price)
            output = "####### Tried buying " + str(int(ideal_position_size/stock_price)) + " or more shares of " + potential_buys[i] + " at ${:.2f}".format(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " however your account balance of ${:.2f}".format(cash) + " is not enough buying power to purchase at the ideal buying position size. #######"
            print(output)
            if (len(potential_buys) > 1):
                ideal_position_size = (safe_division(portfolio_value, holdings_data_length)+cash/(len(potential_buys)-1))/(2 * (len(potential_buys)-1))
            continue

        # Limit the amount of shares if the purchase price is above the limit set in the config file.
        num_shares = purchase_limiter(num_shares, stock_price, equity)

        print("####### Buying " + str(num_shares) +
                " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) +  " with ${:.2f}".format(cash) + " in cash. #######")

        message = "BUY: \nBuying " + str(num_shares) + " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " with ${:.2f}".format(cash) 

        if not debug:
            result = rr.order_buy_market(potential_buys[i], num_shares)
            if 'detail' in result:
                print(result['detail'])
                message = message +  ". The result is " + result['detail']
                json_logger.log("buy_failed", f"Buy failed for {potential_buys[i]}", {
                    "symbol": potential_buys[i],
                    "num_shares": num_shares,
                    "stock_price": stock_price,
                    "detail": result['detail']
                })
            else:
                # Save buy reason if provided
                if buy_reasons and potential_buys[i] in buy_reasons:
                    save_buy_reason(potential_buys[i], buy_reasons[potential_buys[i]], price=stock_price, quantity=num_shares, equity=equity)
                json_logger.log("buy", f"Buying {potential_buys[i]}", {
                    "symbol": potential_buys[i],
                    "num_shares": num_shares,
                    "stock_price": stock_price,
                    "total_cost": stock_price * num_shares,
                    "cash_available": cash,
                    "reason": buy_reasons.get(potential_buys[i]) if buy_reasons else None
                })
        send_text(message)

def purchase_limiter(num_shares, stock_price, equity):
    result = num_shares
    if use_purchase_limit_percentage:
        purchase = num_shares * stock_price
        limit = equity/purchase_limit_percentage

        if purchase > limit:
            result = int(limit/stock_price)
            print("Purchase limited: purchase was " + str(num_shares) + " shares at " + str(stock_price) + " coming to " + str((num_shares * stock_price)) + ". Now the purchase is " + str(result) + " shares coming to " + str((result * stock_price))) 

    return result

def is_market_in_major_downtrend():
    stockTickerNdaq = 'QQQ'
    stockTickerDow = 'DIA'
    stockTickerSP = 'SPY'
    downtrendNdaq = False
    downtrendDow = False
    downtrendSp = False
    # Nasdaq
    # Using NasDaq as the market downtrend indicator which does not have extended trading hours.
    today_history = rsa.get_stock_historicals(stockTickerNdaq, interval='5minute', span='day', bounds='regular') 
    week_history = rsa.get_stock_historicals(stockTickerNdaq, interval='day', span='week', bounds='regular')   
    if(float(week_history[0]['open_price']) > float(today_history[len(today_history) - 1]['close_price'])):
        downtrendNdaq = True

    # DOW
    # Using Dow as the market downtrend indicator.
    today_history = rsa.get_stock_historicals(stockTickerDow, interval='5minute', span='day', bounds='regular')  
    week_history = rsa.get_stock_historicals(stockTickerDow, interval='day', span='week', bounds='regular')   
    if(float(week_history[0]['open_price']) > float(today_history[len(today_history) - 1]['close_price'])):
        downtrendDow = True

    # S&P Index
    # Using S&P as the market downtrend indicator.
    # day_trades = rr.get_day_trades()
    today_history = rsa.get_stock_historicals(stockTickerSP, interval='5minute', span='day', bounds='regular')    
    week_history = rsa.get_stock_historicals(stockTickerSP, interval='day', span='week', bounds='regular')   
    if(float(week_history[0]['open_price']) > float(today_history[len(today_history) - 1]['close_price'])):
        downtrendSp = True
    
    # If there are atleast two markets in a major downtrend over the past week report return true.
    result = (downtrendNdaq + downtrendDow + downtrendSp) >= 2
    if(result):
        print("The markets are in a major downtrend.")
    return result

def is_market_in_uptrend():
    stockTickerNdaq = 'QQQ'
    stockTickerDow = 'DIA'
    stockTickerSP = 'SPY'
    uptrendNdaq = False
    uptrendDow = False
    uptrendSp = False
    # Nasdaq
    # Using NasDaq as the market uptrend indicator which does not have extended trading hours.
    today_history = rsa.get_stock_historicals(stockTickerNdaq, interval='5minute', span='day', bounds='regular')    
    if(float(today_history[0]['open_price']) < float(today_history[len(today_history) - 1]['close_price'])):
        uptrendNdaq = True
    # DOW
    # Using Dow as the market uptrend indicator.
    today_history = rsa.get_stock_historicals(stockTickerDow, interval='5minute', span='day', bounds='regular')    
    if(float(today_history[0]['open_price']) < float(today_history[len(today_history) - 1]['close_price'])):
        uptrendDow = True
    # S&P Index
    # Using S&P as the market uptrend indicator.
    # day_trades = rr.get_day_trades()
    today_history = rsa.get_stock_historicals(stockTickerSP, interval='5minute', span='day', bounds='regular')    
    if(float(today_history[0]['open_price']) < float(today_history[len(today_history) - 1]['close_price'])):
        uptrendSp = True
    
    result = (uptrendNdaq + uptrendDow + uptrendSp) >= 2
    return result

def get_accurate_gains(portfolio_symbols, watchlist_symbols, profileData):
    '''
    Robinhood includes dividends as part of your net gain. This script removes
    dividends from net gain to figure out how much your stocks/options have paid
    off.
    Note: load_portfolio_profile() contains some other useful breakdowns of equity.
    Print profileData and see what other values you can play around with.
    '''

    allTransactions = rr.get_bank_transfers()
    cardTransactions= rr.get_card_transactions()

    deposits = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'completed'))
    withdrawals = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'withdraw') and (x['state'] == 'completed'))
    debits = sum(float(x['amount']['amount']) for x in cardTransactions if (x['direction'] == 'debit' and (x['transaction_type'] == 'settled')))
    reversal_fees = sum(float(x['fees']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'reversed'))

    money_invested = investing or (deposits + reversal_fees - (withdrawals - debits))
    dividends = rr.get_total_dividends()
    percentDividend = 0
    if not money_invested == 0:
        percentDividend = dividends/money_invested*100

    equity_amount = float(profileData['equity'])
    buying_power = float(profileData['equity']) - float(profileData['market_value'])
    totalGainMinusDividends = equity_amount - dividends - money_invested
    percentGain = totalGainMinusDividends/money_invested*100

    bankTransfered = "The total money invested is ${:.2f}".format(money_invested)
    equity = "The total equity is ${:.2f}".format(equity_amount)
    withdrawable_amount = "The buying power is ${:.2f}".format(buying_power)
    equityAndWithdrawable = "The total account value of ${:.2f}".format(float(equity_amount))
    dividendIncrease = "The net worth has increased {:0.3f}% due to dividends that amount to ${:0.2f}".format(percentDividend, dividends)
    gainIncrease = "The net worth has increased {:0.3f}% due to other gains that amount to ${:0.2f}\n".format(percentGain, totalGainMinusDividends)

    print("For accurate numbers the [investing] value in the config should be set to the initial Investing number on Robinhoods main graph.\n")
    print(bankTransfered)
    print(equity)
    print(withdrawable_amount)
    print(equityAndWithdrawable)
    print(dividendIncrease)
    print(gainIncrease)
    
    """ Send a text message with the days metrics """

    if debug: 
        print("----- Scanning market reports to add stocks to watchlist -----")
        market_tag_report = get_market_tag_stocks_report()
        # If the market tag report has some stock values...
        if len(market_tag_report) > 0 and market_tag_report[0] != '':
            send_text(market_tag_report[0])
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)
        print("----- End market reports scan -----")

    # Evening Morning report
    begin_time = datetime.time(8, 30)
    end_time = datetime.time(9, 30)
    timenow = datetime.datetime.now().time()

    if(timenow >= begin_time and timenow < end_time):
        if(timenow >= begin_time and timenow < datetime.time(9, 00)):
            print("Sending morning report.")
            send_text(bankTransfered + "\n" + withdrawable_amount)
            time.sleep(2)
            send_text(equity)      
            time.sleep(2)
            send_text(equityAndWithdrawable + "\n" + gainIncrease)
        # Get interesting stocks report.
        market_tag_report = get_market_tag_stocks_report()
        if len(market_tag_report) > 0 and market_tag_report[0] != '':
            # If the market tag report has some stock values...
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)

    # Evening report
    begin_time = datetime.time(17, 30)
    end_time = datetime.time(18, 30)

    if(timenow >= begin_time and timenow < end_time):
        if(timenow >= begin_time and timenow < datetime.time(18, 00)):
            print("Sending evening report.")
            send_text(bankTransfered + "\n" + withdrawable_amount)
            time.sleep(2)
            send_text(equity)      
            time.sleep(2)
            send_text(equityAndWithdrawable + "\n" + gainIncrease)
        # Get interesting stocks report.
        market_tag_report = get_market_tag_stocks_report()
        if len(market_tag_report) > 0 and market_tag_report[0] != '':
            # If the market tag report has some stock values...
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)

    # Morning auto-invest
    begin_time = datetime.time(10, 00)
    end_time = datetime.time(11, 00)
    timenow = datetime.datetime.now().time()
          
    if(timenow >= begin_time and timenow < end_time):
        print("----- Scanning market reports to add stocks to watchlist -----")
        market_tag_report = get_market_tag_stocks_report()
        # If the market tag report has some stock values...
        if len(market_tag_report) > 0 and market_tag_report[0] != '':
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)
        print("----- End market reports scan -----")    

    # Afternoon auto-invest
    begin_time = datetime.time(13, 00)
    end_time = datetime.time(14, 00)
    timenow = datetime.datetime.now().time()
          
    if(timenow >= begin_time and timenow < end_time):
        print("----- Scanning market reports to add stocks to watchlist -----")
        market_tag_report = get_market_tag_stocks_report()
        # If the market tag report has some stock values...
        if len(market_tag_report) > 0 and market_tag_report[0] != '':
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)
        print("----- End market reports scan -----") 

def is_market_open_now():
    begin_time = datetime.time(9, 30)
    end_time = datetime.time(16, 00)
    timenow = datetime.datetime.now().time()

    # Check if it's a trading day (weekday and not a holiday)
    if is_trading_day(datetime.date.today()):
        if(timenow >= begin_time and timenow < end_time):
            return True

    return False
        
def profit_before_eod(stock, holdings_data):
    """ Return true if there is a profit before the close of the day.
    Args:
        symbol(str): The symbol of the stock.
    """    
    end_time = datetime.time(16,00)
    eod = is_eod()
    
    # If it is after the begin time and before the end time then check to see if there is a profit available.
    if(eod):
        average_buy_price = float(holdings_data[stock]['average_buy_price'])
        price = float(holdings_data[stock]['price'])

        # Perhaps use average buy price and price in holdings_data?
        # If this stock was traded today use the intraday percent change.
        percent_change = float(holdings_data[stock]['intraday_percent_change'])
        if(percent_change > 0):
            message = "The price of " + stock + " after " + str(end_time) + " has increased " + str(percent_change) + "%."
            print(message)
            return True
        elif (price > average_buy_price):
            message = "The price of " + stock + " after " + str(end_time) + " ($" + str(price) + ") is greater than it was purchased at ($" + str(average_buy_price) + ")."
            print(message)
            return True
    return False

def is_eod():
    """ Return true if the time of the day is after 1:30pm. After this time the price is subject to inflection probability increases significantly. """
    begin_time = datetime.time(13, 30)
    end_time = datetime.time(16,00)
    
    timenow = datetime.datetime.now().time()

    eod = timenow >= begin_time and timenow < end_time and is_trading_day(datetime.date.today())
    return eod


def sudden_drop(symbol, percent, hours_apart):
    """ Return true if the price drops more than the percent argument in the span of hours_apart.

    Args:
        symbol(str): The symbol of the stock.
        percent(float): The amount of percentage drop from the previous close price.
        hours_apart(float): Number of hours away from the current to check.

    Returns:
        True if there is a sudden drop.
    """
    historicals = rsa.get_stock_historicals(symbol, interval='hour', span='month')
    if len(historicals) == 0:
        return False

    if (len(historicals) - 1 - hours_apart) < 0:
        return False
        
    percentage = (percent/100) * float(historicals[len(historicals) - 1 - hours_apart]['close_price'])
    target_price = float(historicals[len(historicals) - 1 - hours_apart]['close_price']) - percentage

    if float(historicals[len(historicals) - 1]['close_price']) <= target_price:
        message = "The " + symbol + " has dropped from " + str(float(historicals[len(historicals) - 1 - hours_apart]['close_price'])) + " to " + str(float(historicals[len(historicals) - 1]['close_price'])) + " which is more than " + str(percent) + "% (" + str(target_price) + ") in the span of " + str(hours_apart) + " hour(s)."
        print(message)
        send_text(message)
        return True
    
    return False


def get_drop_percentages(symbol):
    """Get the percentage drop over 1 hour and 2 hours for a stock.
    
    Args:
        symbol(str): The symbol of the stock.
    
    Returns:
        dict with drop_1hr and drop_2hr percentages (negative = drop, positive = gain)
    """
    try:
        historicals = rsa.get_stock_historicals(symbol, interval='hour', span='month')
        if len(historicals) < 3:
            return None
        
        current_price = float(historicals[-1]['close_price'])
        
        # 1 hour ago
        price_1hr_ago = float(historicals[-2]['close_price']) if len(historicals) >= 2 else current_price
        drop_1hr = ((current_price - price_1hr_ago) / price_1hr_ago) * 100 if price_1hr_ago > 0 else 0
        
        # 2 hours ago
        price_2hr_ago = float(historicals[-3]['close_price']) if len(historicals) >= 3 else current_price
        drop_2hr = ((current_price - price_2hr_ago) / price_2hr_ago) * 100 if price_2hr_ago > 0 else 0
        
        return {
            'price_1hr_ago': round(price_1hr_ago, 2),
            'price_2hr_ago': round(price_2hr_ago, 2),
            'current_price': round(current_price, 2),
            'change_1hr': round(drop_1hr, 2),
            'change_2hr': round(drop_2hr, 2),
            'trigger_1hr': -15,  # 15% drop in 1 hour triggers sell
            'trigger_2hr': -10   # 10% drop in 2 hours triggers sell
        }
    except Exception as e:
        return None


def sudden_increase(symbol, percent, minutes_apart):
    """ Return true if the price increases more than the percent argument in the span of two hours_apart.

    Args:
        symbol(str): The symbol of the stock.
        percent(float): The amount of percentage drop from the previous close price.
        hours_apart(float): Number of hours away from the current to check.

    Returns:
        True if there is a sudden drop.
    """
    minutes_apart_5_min = int(minutes_apart/5)
    historicals = rsa.get_stock_historicals(symbol, interval='5minute', span='day')
    if len(historicals) == 0:
        return False

    if (len(historicals) - 1 - minutes_apart_5_min) < 0:
        return False
        
    percentage = (percent/100) * float(historicals[len(historicals) - 1 - minutes_apart_5_min]['close_price'])
    target_price = float(historicals[len(historicals) - 1 - minutes_apart_5_min]['close_price']) + percentage

    if float(historicals[len(historicals) - 1]['close_price']) >= target_price:
        message = "The " + symbol + " has increased from " + str(float(historicals[len(historicals) - 1 - minutes_apart_5_min]['close_price'])) + " to " + str(float(historicals[len(historicals) - 1]['close_price'])) + " which is more than " + str(percent) + "% (" + str(target_price) + ") in the span of " + str(minutes_apart) + " minute(s)."
        print(message)
        return True
    
    return False

def percent_increase(symbol, percent, buy_price, current_price):
    """ Return true if the price increases more than the percent argument in the span of two hours_apart.

    Args:
        symbol(str): The symbol of the stock.
        percent(float): The amount of percentage increase from the previous close price.

    Returns:
        True if there is a sudden increase.
    """        
    percentage = (percent/100) * buy_price
    target_price = buy_price + percentage

    if current_price >= target_price:
        message = "The " + symbol + " has increased from " + str(buy_price) + " to " + str(current_price) + " which is more than " + str(percent) + "% (" + str(target_price) + ")."
        print(message)
        return True
    
    return False

def auto_invest(stock_array, portfolio_symbols, watchlist_symbols):
    try:
        invest = True

        # If the previous stock that we added to the watchlist is still here
        # or the stock is in an exclusion list if one has been set
        # then dont auto invest any other stocks for now to prevent just adding
        # all stocks to the investment pool thus diluting the investment potential
        # in the previous stock that has been autoinvested.
        exclusion_list = rsa.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
        stock_array_numpy = np.array(stock_array)
        stock_array = np.unique(stock_array_numpy).tolist()
        stock_array_copy = stock_array.copy()
        for stock in stock_array:
            now = datetime.datetime.now()
            print(now)
            removed = False
            if (stock in portfolio_symbols):
                # The code below was meant to prevent too many purchases of stock in the hopes
                # but this has now been commented out in the hopes of experiementing with the
                # benefits of multiple investments.
                # invest = False
                # message_skip = stock + " is still in the recomended list. Auto-Invest will skip this interval in order to allow time between stock generation."
                # print(message_skip)
                # send_text(message_skip)
                if (stock in stock_array_copy):
                    stock_array_copy.remove(stock)
                    removed = True
                    print(stock + " removed from auto-invest because it is already in the portfolio.")
            if (use_exclusion_watchlist):
                for exclusion_result in exclusion_list['results']:
                    if (stock == exclusion_result['symbol']):
                        if (stock in stock_array_copy):
                            stock_array_copy.remove(stock)
                            removed = True
                            print(stock + " removed from auto-invest because it was in the exclusion list.")
            if (stock in watchlist_symbols):
                if stock in stock_array_copy:
                    stock_array_copy.remove(stock)
                    removed = True
                    print(stock + " removed from auto-invest because it is already in the watchlist.")
            if (not removed):
                # If this stock is untradeable on the robin hood platform
                # take it out of the list of stocks under consideration.
                stock_info = rsa.get_instruments_by_symbols(stock, info='tradeable')
                if (len(stock_info) == 0 or not stock_info[0]):
                    if stock in stock_array_copy:
                        stock_array_copy.remove(stock)
                        removed = True
                        print(stock + " removed from auto-invest because RobinHood has marked this stock as untradeable.")
            fundamentals = rsa.get_fundamentals(stock)
            if (not removed):
                average_volume = float(fundamentals[0]['average_volume'] or 0)
                if(average_volume < min_volume):
                    if stock in stock_array_copy:
                        stock_array_copy.remove(stock)
                        removed = True
                        print(stock + " removed from auto-invest because the average volume of this stock is less than " + str(min_volume) + ".")
            if (not removed):
                market_cap = float(fundamentals[0]['market_cap'])
                if(market_cap < min_market_cap):
                    if stock in stock_array_copy:
                        stock_array_copy.remove(stock)
                        removed = True
                        print(stock + " removed from auto-invest because the market cap of this stock is less than " + str(min_market_cap) + ".")
            if (not removed and use_price_cap):
                # If a price cap has been set remove any stocks
                # that go above the cap or if the stock does not have
                # any history for today.
                history = rsa.get_stock_historicals(stock, interval='day')
                if (len(history) == 0 or float(history[len(history) - 1]['close_price']) > price_cap):
                    if stock in stock_array_copy:
                        stock_array_copy.remove(stock)
                        removed = True
                        if (len(history) == 0):
                            print(stock + " removed from auto-invest because it has no stock history to analyze.")
                        else:
                            print(stock + " removed from auto-invest because its price of " + str(float(history[len(history) - 1]['close_price'])) + " was greater than your price cap of " + str(price_cap))
            if (not removed):
                print(stock + " has survived.")

        if (invest):
            stock_array = stock_array_copy

            # Lowest price.
            # symbol_and_price = find_symbol_with_lowest_price(stock_array)
            # selected_symbol = symbol_and_price[0]
            # lowest_price = symbol_and_price[1]
            # message = "Auto-Invest is adding " + selected_symbol + " at ${:.2f}".format(lowest_price) + " to the " + watch_list_name + " watchlist."

            # Greatest slope for today.
            selected_symbol = find_symbol_with_greatest_slope(stock_array)

            # Highest volume.
            # selected_symbol = find_symbol_with_highest_volume(stock_array)
            
            if(selected_symbol == ''):
                return

            message = "Auto-Invest is adding " + selected_symbol + " to the " + watch_list_name + " watchlist."
            send_text(message)
            print(message)
            if not debug:
                rr.post_symbols_to_watchlist(selected_symbol, watch_list_name)

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    except ValueError:
        print("Could not convert data to an integer.")
    except Exception as e:
        print("Unexpected error could not generate interesting stocks report:", str(e))

        login_to_sms()
        send_text("Unexpected error could not generate interesting stocks report:" + str(e) + "\n Trace: " + traceback.format_exc())
        print(traceback.format_exc())

def find_symbol_with_greatest_slope(stock_array):
    linregressResults = []
    for stockTicker in stock_array:
        # Load stock numbers.
        history = rsa.get_stock_historicals(stockTicker, interval='5minute', span='day', bounds='regular')
        closingPrices = []
        dates = []
        i = 0
        for history_item in history:
            closingPrices.append(float(history_item['close_price']))
            # dates.append(history_item['begins_at'])
            i = i + 1
            dates.append(i)
        # Determine slopes.
        linregressResult = linregress(dates, closingPrices)
        linregressResults.append(linregressResult.slope)
    # Find index.
    sorted_lineregress = sorted(linregressResults)
    if(len(sorted_lineregress) == 0):
        return ''
    highest_slope = sorted_lineregress[len(sorted_lineregress) - 1]
    index_of_highest_slope = [float(i) for i in linregressResults].index(highest_slope)
    symbol_of_highest_slope = stock_array[index_of_highest_slope]
    return symbol_of_highest_slope

def find_symbol_with_highest_volume(stock_array):
    volume_array = []
    for stock in stock_array:
        volumes = rsa.get_stock_historicals(stock, interval='day', span='week', bounds='regular', info='volume')
        if len(volumes) == 0:
            continue
        volume_array.append(volumes[len(volumes) - 1])
    stock_and_volume_float_array = [float(i) for i in volume_array]
    sorted_volume_array = sorted(stock_and_volume_float_array, key=float)
    highest_volume = sorted_volume_array[len(sorted_volume_array) - 1]
    # Convert the string price array to float and find the index of the 
    # stock with the highest volume.
    index_of_highest_volume = [float(i) for i in volume_array].index(highest_volume)
    symbol_of_highest_volume = stock_array[index_of_highest_volume]
    return symbol_of_highest_volume

def find_stock_with_lowest_price(stock_array):
    # Find stock with the lowest stock price.
    price_array = rr.get_latest_price(stock_array)
    stock_and_price_float_array = [float(i) for i in price_array]
    sorted_price_array = sorted(stock_and_price_float_array, key=float)
    lowest_price = sorted_price_array[0]
    # Convert the string price array to float and find the index of the 
    # stock with the lowest price.
    index_of_lowest_price = [float(i) for i in price_array].index(lowest_price)
    symbol_of_lowest_price = stock_array[index_of_lowest_price]
    return symbol_of_lowest_price, index_of_lowest_price

def get_market_tag_stocks_report():
    try:
        report_string = ""
        market_tag_for_report_array = market_tag_for_report.split(',')
        stock_array = []

        for market_tag_for_report_item in market_tag_for_report_array:
            all_market_tag_stocks = rr.get_all_stocks_from_market_tag(market_tag_for_report_item, info = 'symbol')
            print(market_tag_for_report_item + " " + str(len(all_market_tag_stocks)) + " items.")
            for market_tag_stock in all_market_tag_stocks:
                # cross = golden_cross(market_tag_stock, n1=short_sma, n2=long_sma, days=5, direction="above")
                cross = retry_call(golden_cross, fargs=[market_tag_stock], fkwargs={"n1": short_sma,"n2": long_sma,"days": 5, "direction": "above"}, tries=3, backoff=5, delay=2)
                if(cross[0] == 1):
                    report_string = report_string + "\n" + market_tag_stock + "{:.2f}".format(cross[2])
                    stock_array.append(market_tag_stock)

        if(report_string != ""):
            return market_tag_for_report + "\n" + report_string, stock_array
        return "", stock_array

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    except ValueError:
        print("Could not convert data to an integer.")
    except Exception as e:
        print("Unexpected error could not generate interesting stocks report:", str(e))

        login_to_sms()
        send_text(
            "Unexpected error could not generate interesting stocks report:" + str(e) + "\n Trace: " + traceback.format_exc())

def order_symbols_by_slope(portfolio_symbols):
    """ This method orders an array of symbols by their slope in descending order
    """ 
    try:
        w, h = 2, 0
        Matrix = [[0 for x in range(w)] for y in range(h)] 
        for stockTicker in portfolio_symbols:
            # Load stock numbers.
            history = rsa.get_stock_historicals(stockTicker, interval='5minute', span='day', bounds='regular')
            closingPrices = []
            dates = []
            i = 0
            for history_item in history:
                closingPrices.append(float(history_item['close_price']))
                # dates.append(history_item['begins_at'])
                i = i + 1
                dates.append(i)
            # Determine slopes.
            linregressResult = linregress(dates, closingPrices)
            Matrix.append([stockTicker, linregressResult.slope])
        sorted_matrix = sorted(Matrix, key=lambda l:l[1], reverse=True)
        result_matrix = [[0 for x in range(2)] for y in range(0)]
        for row in sorted_matrix:
            # Only return rows that have a positive slope. We dont need to invest 
            # in stocks that have a negative slope in the current trading day.
            if row[1] > 0.0008:
                result_matrix.append(row)

        just_first_column = [row[0] for row in result_matrix]
        return just_first_column
    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    except ValueError:
        print("Could not convert data to an integer.")
    except Exception as e:
        print("Unexpected error could not generate interesting stocks report:", str(e))

        login_to_sms()
        send_text(
            "Unexpected error could not generate interesting stocks report:" + str(e) + "\n Trace: " + traceback.format_exc())

def build_pheonix_profile_data(profile_data_with_dividend):
    """Builds a dictionary of important information regarding the user account.

    :returns: Returns a dictionary that has total equity, extended hours equity, cash, and divendend total.

    """
    profile_data = {}

    pheonix_account = rr.load_phoenix_account()

    profile_data['equity'] = pheonix_account['results'][0]['total_equity']['amount']
    if (pheonix_account['results'][0]['total_extended_hours_equity']):
        profile_data['extended_hours_equity'] = pheonix_account['results'][0]['total_extended_hours_equity']['amount']
    profile_data['cash'] = pheonix_account['results'][0]['uninvested_cash']['amount']

    profile_data['dividend_total'] = profile_data_with_dividend['dividend_total']

    return profile_data

def scan_stocks():
    """ The main method. Sells stocks in your portfolio if their 50 day moving average crosses
        below the 200 day, and buys stocks in your watchlist if the opposite happens.

        ###############################################################################################
        WARNING: Comment out the sell_holdings and buy_holdings lines if you don't actually want to execute the trade.
        ###############################################################################################

        If you sell a stock, this updates tradehistory.txt with information about the position,
        how much you've earned/lost, etc.
    """

    try:

        # Log in to Robinhood
        # Put your username in the config file.
        login = rr.authentication.login(username=rh_username)
        login_to_sms()

        # Clear caches at the start of each scan to get fresh data
        clear_all_caches()

        if debug:
            print("----- DEBUG MODE -----\n")

        print("----- Version " + version + " -----\n")
        json_logger.log("scan_start", f"Starting scan - Version {version}", {"version": version, "debug": debug})
        
        print("----- Starting scan... -----\n")
        register_matplotlib_converters()
        watchlist_symbols = get_watchlist_symbols(False)
        portfolio_symbols = get_portfolio_symbols()
        holdings_data = get_modified_holdings()
        
        json_logger.log("portfolio_status", "Portfolio and watchlist loaded", {
            "portfolio_symbols": portfolio_symbols,
            "watchlist_symbols": watchlist_symbols,
            "portfolio_count": len(portfolio_symbols),
            "watchlist_count": len(watchlist_symbols)
        })
        profileData = rr.load_portfolio_profile()
        potential_buys = []
        buy_reasons = {}  # Track reasons for buying each stock
        sells = []
        sell_reasons = {}  # Track reasons for selling each stock
        print("Current Portfolio: " + str(portfolio_symbols) + "\n")
        print("Current Watchlist: " + str(watchlist_symbols) + "\n")
        print("----- Scanning portfolio for stocks to sell -----\n")
        market_uptrend = is_market_in_uptrend()        
        market_in_major_downtrend = is_market_in_major_downtrend()

        if(not market_uptrend):
                print("The market(s) in general are in a downtrend.  Setting the sell day period to 14 days.")
                n1 = 14

        if(market_in_major_downtrend):
                print("The market(s) are in a major downtrend.  Setting the sell day period to 14 days.")
                n1 = 14

        open_stock_orders = rsa.get_all_open_stock_orders()
        is_market_open = is_market_open_now()

        day_trade_message = ""
        for symbol in portfolio_symbols:
            # If the market is open and there are open stock orders still pending
            # 5 at least minutes after the opening bell, cancel them.  This is meant to remove orders
            # that are still pending because the buy price has gone far higher
            # than what the price was when the order was originally when the markets 
            # were closed.
            continue_outer_loop = False
            if(len(open_stock_orders) > 0):
                for open_stock_order in open_stock_orders:
                    instrument = rsa.get_instrument_by_url(open_stock_order['instrument'])
                    if(instrument['symbol'] == symbol):
                        five_minutes_after_market_open_hours = datetime.datetime.now().hour >= 9 and datetime.datetime.now().minute > 35 and datetime.datetime.now().hour < 11
                        if(is_market_open and five_minutes_after_market_open_hours):
                            rr.cancel_stock_order(open_stock_order['id'])
                        else:
                            # If the markets are closed then continue to the next symbol
                            # as this pending market action for this symbol should be 
                            # allowed to proceed.
                            continue_outer_loop = True
            if(continue_outer_loop):
                print("Skipping " + symbol + " because there is a order currently pending for this symbol.")
                continue

            n1 = short_sma
            n2 = long_sma
            # If we are not in a market uptrend, tighten the belt and use
            # the downtrend short-term SMA from config.
            if use_dynamic_sma and ((not market_uptrend) or market_in_major_downtrend):
                n1 = short_sma_downtrend
            tradeable_stock_info = rsa.get_instruments_by_symbols(symbol)
            if (len(tradeable_stock_info) == 0 or not tradeable_stock_info[0]['tradeable']):
                continue
            # sudden_increase an increase of 10% or more over the course of 2 hours then drops by at least 5% in an hour then set the short term to 5 and the long term to 7.
            # is_sudden_increase = sudden_increase(symbol, 10, 2) or sudden_increase(symbol, 15, 1)
            # if(is_sudden_increase):
            #     n1 = 5
            #     n2 = 7
            #     print("For " + symbol + " setting the short term period to " + str(n1) + " and setting the long term period to " + str(n2) + ".")
            is_traded_today = traded_today(symbol, profileData)
            is_take_profit = take_profit(symbol, holdings_data, take_profit_percent)
            # If we have surpassed the take profit threshold and the stock was traded today
            # make it less likely to sell by simply changing the periods and not immediately 
            # selling in order to try our best not to hit our day trade limit.
            if use_dynamic_sma and is_take_profit and is_traded_today:
                n1 = short_sma_take_profit
                n2 = long_sma_take_profit
                is_take_profit = False
                print("For " + symbol + " setting the short term period to " + str(n1) + " and setting the long term period to " + str(n2) + ".")
            is_sudden_drop = sudden_drop(symbol, 10, 2) or sudden_drop(symbol, 15, 1)
            # If there is a profit before the end of day then sell because there is usually an inflection point after 2pm.
            is_profit_before_eod = profit_before_eod(symbol, holdings_data)
            cross = golden_cross(symbol, n1=n1, n2=n2, days=10, direction="below")
            
            # Check stop-loss
            is_stop_loss = False
            stop_loss_pct = 0
            if use_stop_loss:
                avg_buy_price = float(holdings_data[symbol]['average_buy_price'])
                current_price = float(holdings_data[symbol]['price'])
                stop_loss_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0
                if stop_loss_pct <= -stop_loss_percent:
                    is_stop_loss = True
                    print(f"🛑 STOP-LOSS triggered for {symbol}: {stop_loss_pct:.2f}% (threshold: -{stop_loss_percent}%)")
            
            # Determine sell reasons - track all that apply
            sell_reasons_list = []
            if cross[0] == -1:
                sell_reasons_list.append("death_cross")
            if is_sudden_drop:
                sell_reasons_list.append("sudden_drop")
            if is_take_profit:
                sell_reasons_list.append("take_profit")
            if is_profit_before_eod:
                sell_reasons_list.append("profit_before_eod")
            if is_stop_loss:
                sell_reasons_list.append("stop_loss")
            
            # Join all reasons or use None if empty
            sell_reason = ",".join(sell_reasons_list) if sell_reasons_list else None
            
            if(cross[0] == -1 or is_sudden_drop or is_take_profit or is_profit_before_eod or is_stop_loss):
                day_trades = get_day_trades(profileData)
                if ((day_trades <= 1) or (not is_traded_today)):
                    print("Day trades currently: " + str(day_trades))
                    print("Traded today: " + str(is_traded_today))
                    sell_holdings(symbol, holdings_data)
                    sells.append(symbol)
                    sell_reasons[symbol] = sell_reason
                    save_trade_reason(symbol, sell_reason, action="sell", holdings_data=holdings_data)
                    json_logger.log("sell", f"Selling {symbol}", {
                        "symbol": symbol,
                        "reason": sell_reason,
                        "day_trades": day_trades,
                        "holdings_data": holdings_data.get(symbol, {})
                    })
                    genetic_generation_add(symbol, is_take_profit)
                else:
                    day_trade_message = "Unable to sell " + symbol + " because there are " + str(day_trades) + " day trades and/or this stock was traded today."
                    print(day_trade_message)
                    send_text(day_trade_message)
                    json_logger.log("sell_blocked", f"Cannot sell {symbol} - day trade limit", {
                        "symbol": symbol,
                        "reason": sell_reason,
                        "day_trades": day_trades,
                        "traded_today": is_traded_today
                    })
            else:
                # Log why this stock is NOT being sold
                hold_reasons = []
                hold_data = {"symbol": symbol}
                
                # Add market trend info
                market_status = "uptrend" if market_uptrend else "downtrend"
                if market_in_major_downtrend:
                    market_status = "major_downtrend"
                hold_data["market_trend"] = {
                    "status": market_status,
                    "uptrend": market_uptrend,
                    "major_downtrend": market_in_major_downtrend,
                    "sma_period_used": n1
                }
                
                if cross[0] != -1:
                    # Show death_cross metrics: SMA values and gap
                    sma_data = get_sma_proximity(symbol, n1, n2)
                    if sma_data:
                        sma_data["sma_period"] = n1
                        hold_reasons.append(f"no death_cross (SMA{n1}=${sma_data['sma_short']}, SMA{n2}=${sma_data['sma_long']}, gap={sma_data['gap_pct']:.2f}%, market={market_status})")
                        hold_data["death_cross_metrics"] = sma_data
                    else:
                        hold_reasons.append(f"no death_cross (market={market_status})")
                if not is_sudden_drop:
                    # Show sudden_drop metrics: percentage change over 1hr and 2hr
                    drop_data = get_drop_percentages(symbol)
                    if drop_data:
                        hold_reasons.append(f"no sudden_drop (1hr={drop_data['change_1hr']:.2f}% need<-15%, 2hr={drop_data['change_2hr']:.2f}% need<-10%)")
                        hold_data["sudden_drop_metrics"] = drop_data
                    else:
                        hold_reasons.append("no sudden_drop")
                if not is_take_profit:
                    # Show take_profit metrics: current price, buy price, percent change, and the threshold
                    avg_buy_price = float(holdings_data[symbol]['average_buy_price'])
                    current_price = float(holdings_data[symbol]['price'])
                    intraday_pct = float(holdings_data[symbol]['intraday_percent_change'])
                    total_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0
                    hold_reasons.append(f"no take_profit (buy=${avg_buy_price:.2f}, now=${current_price:.2f}, intraday={intraday_pct:.2f}%, total={total_pct:.2f}%, need={take_profit_percent}%)")
                    hold_data["take_profit_metrics"] = {
                        "avg_buy_price": avg_buy_price,
                        "current_price": current_price,
                        "intraday_pct": intraday_pct,
                        "total_pct": total_pct,
                        "threshold": take_profit_percent
                    }
                if not is_profit_before_eod:
                    # Show profit_before_eod metrics: is it EOD time, current price, buy price, intraday percent change
                    is_eod_time = is_eod()
                    avg_buy_price_eod = float(holdings_data[symbol]['average_buy_price'])
                    current_price_eod = float(holdings_data[symbol]['price'])
                    intraday_pct_eod = float(holdings_data[symbol]['intraday_percent_change'])
                    current_time = datetime.datetime.now().strftime("%H:%M")
                    hold_reasons.append(f"no profit_before_eod (is_eod={is_eod_time}, time={current_time}, buy=${avg_buy_price_eod:.2f}, now=${current_price_eod:.2f}, intraday={intraday_pct_eod:.2f}%, need=is_eod+profit)")
                    hold_data["profit_before_eod_metrics"] = {
                        "is_eod": is_eod_time,
                        "current_time": current_time,
                        "eod_window": "13:30-16:00",
                        "avg_buy_price": avg_buy_price_eod,
                        "current_price": current_price_eod,
                        "intraday_pct": intraday_pct_eod,
                        "has_profit": intraday_pct_eod > 0 or current_price_eod > avg_buy_price_eod,
                        "need": "is_eod=True AND (intraday>0% OR price>buy_price)"
                    }
                if not is_stop_loss:
                    # Show stop-loss metrics
                    sl_status = "enabled" if use_stop_loss else "disabled"
                    hold_reasons.append(f"no stop_loss ({sl_status}, loss={stop_loss_pct:.2f}%, trigger=-{stop_loss_percent}%)")
                    hold_data["stop_loss_metrics"] = {
                        "enabled": use_stop_loss,
                        "current_loss_pct": round(stop_loss_pct, 2),
                        "trigger_pct": -stop_loss_percent,
                        "would_trigger_at": round(float(holdings_data[symbol]['average_buy_price']) * (1 - stop_loss_percent/100), 2)
                    }
                hold_data["reasons"] = hold_reasons
                print(f"HOLD {symbol}: {', '.join(hold_reasons)}")
                json_logger.log("hold", f"Holding {symbol}", hold_data)
        profile_data_with_dividend_total = rr.build_user_profile()
        profile_data = build_pheonix_profile_data(profile_data_with_dividend_total)
        ordered_watchlist_symbols = order_symbols_by_slope(watchlist_symbols)
        print("\n----- Scanning watchlist for stocks to buy -----\n")
        for symbol in ordered_watchlist_symbols:
            if(symbol not in portfolio_symbols):
                cross = golden_cross(symbol, n1=20, n2=50, days=golden_cross_buy_days, direction="above")
                if(cross[0] == 1):
                        # If the current price is greater than the price at cross,
                        # meaning that the price is still rising then buy.
                    if(float(cross[2]) > float(cross[1])):
                        # If the current price is greater than the price 5 hours ago,
                        # meaning we have less of a chance of the stock showing a 
                        # death cross soon then buy.
                        if(float(cross[2]) > float(cross[3])):
                            if(market_uptrend and not market_in_major_downtrend):
                                day_trades = get_day_trades(profileData)
                                if day_trades <= 1 or not traded_today(symbol, profileData):
                                    if is_market_open or premium_account:
                                        eod = is_eod()
                                        # If it is not after 2:30pm then we can buy.
                                        # Buying after 2:30pm may not be a good ideas as 
                                        # inflection in the price usually occurs at the 
                                        # end of the day.  Possibly day traders start
                                        # exiting their positions.
                                        if (not eod):
                                            potential_buys.append(symbol)
                                            buy_reasons[symbol] = "golden_cross"
                                        else:
                                            eod_message = symbol + ": It is after 1:30pm EST and as such trading has ended for today as the probability for inflection change usually occurs after this time of day up to and including market close."
                                            print(eod_message)
                                            send_text(eod_message)
                                            json_logger.log("skip_buy", f"Skipping {symbol} - end of day", {
                                                "symbol": symbol,
                                                "reason": "eod_trading_paused",
                                                "golden_cross": True,
                                                "golden_cross_date": cross[7] if len(cross) > 7 else None,
                                                "price_rising": True,
                                                "price_above_5hr_ago": True,
                                                "current_price": float(cross[2]),
                                                "sma20": round(cross[4], 4) if cross[4] else None,
                                                "sma50": round(cross[5], 4) if cross[5] else None,
                                                "market_uptrend": market_uptrend,
                                                "is_eod": eod,
                                                "current_time": datetime.datetime.now().strftime("%H:%M"),
                                                "need": "Wait until before 1:30pm EST (trading window: 9:30am-1:30pm)"
                                            })
                                    else:
                                        print("If the market is closed and you do not have a premium account then buying is paused until market open.")
                                        json_logger.log("skip_buy", f"Skipping {symbol} - market closed", {
                                            "symbol": symbol,
                                            "reason": "market_closed_no_premium",
                                            "golden_cross": True,
                                            "golden_cross_date": cross[7] if len(cross) > 7 else None,
                                            "price_rising": True,
                                            "price_above_5hr_ago": True,
                                            "current_price": float(cross[2]),
                                            "sma20": round(cross[4], 4) if cross[4] else None,
                                            "sma50": round(cross[5], 4) if cross[5] else None,
                                            "market_uptrend": market_uptrend,
                                            "is_market_open": is_market_open,
                                            "premium_account": premium_account,
                                            "need": "Market must be open (9:30am-4:00pm EST) OR have premium_account=True"
                                        })
                                else:
                                    day_trade_message = "Unable to buy " + symbol + " because there are " + str(day_trades) + " day trades."
                                    print(day_trade_message)
                                    send_text(day_trade_message)
                                    json_logger.log("skip_buy", f"Skipping {symbol} - day trade limit", {
                                        "symbol": symbol,
                                        "reason": "day_trade_limit",
                                        "golden_cross": True,
                                        "golden_cross_date": cross[7] if len(cross) > 7 else None,
                                        "price_rising": True,
                                        "price_above_5hr_ago": True,
                                        "current_price": float(cross[2]),
                                        "sma20": round(cross[4], 4) if cross[4] else None,
                                        "sma50": round(cross[5], 4) if cross[5] else None,
                                        "market_uptrend": market_uptrend,
                                        "day_trades": day_trades,
                                        "traded_today": traded_today(symbol, profileData),
                                        "need": f"day_trades <= 1 (currently {day_trades}) OR stock not traded today"
                                    })
                            else:
                                print("But the markets on average are not in an uptrend.")
                                json_logger.log("skip_buy", f"Skipping {symbol} - market downtrend", {
                                    "symbol": symbol,
                                    "reason": "market_not_uptrend",
                                    "golden_cross": True,
                                    "golden_cross_date": cross[7] if len(cross) > 7 else None,
                                    "price_rising": True,
                                    "price_above_5hr_ago": True,
                                    "current_price": float(cross[2]),
                                    "sma20": round(cross[4], 4) if cross[4] else None,
                                    "sma50": round(cross[5], 4) if cross[5] else None,
                                    "market_uptrend": market_uptrend,
                                    "market_in_major_downtrend": market_in_major_downtrend,
                                    "need": "At least 2 of 3 indexes (QQQ, DIA, SPY) must be up today AND not in major weekly downtrend"
                                })
                        else:
                            print("But the price is lower than it was 5 hours ago.")
                            json_logger.log("skip_buy", f"Skipping {symbol} - price below 5hr ago", {
                                "symbol": symbol,
                                "reason": "price_below_5hr_ago",
                                "golden_cross": True,
                                "golden_cross_date": cross[7] if len(cross) > 7 else None,
                                "price_rising": True,
                                "current_price": float(cross[2]),
                                "price_5hr_ago": float(cross[3]),
                                "sma20": round(cross[4], 4) if cross[4] else None,
                                "sma50": round(cross[5], 4) if cross[5] else None,
                                "need": f"Current price (${cross[2]:.2f}) must be > price 5hr ago (${cross[3]:.2f})"
                            })
                    else:
                        print("But the price is lower than it was when the golden cross formed " + str(cross[2]) + " < " + str(cross[1]))
                        json_logger.log("skip_buy", f"Skipping {symbol} - price below cross price", {
                            "symbol": symbol,
                            "reason": "price_below_cross_price",
                            "golden_cross": True,
                            "golden_cross_date": cross[7] if len(cross) > 7 else None,
                            "current_price": float(cross[2]),
                            "price_at_cross": float(cross[1]),
                            "sma20": round(cross[4], 4) if cross[4] else None,
                            "sma50": round(cross[5], 4) if cross[5] else None,
                            "need": f"Current price (${cross[2]:.2f}) must be > price at golden cross (${cross[1]:.2f})"
                        })
                else:
                    # No golden cross detected
                    json_logger.log("skip_buy", f"Skipping {symbol} - no golden cross", {
                        "symbol": symbol,
                        "reason": "no_golden_cross",
                        "golden_cross": False,
                        "golden_cross_date": None,
                        "cross_result": cross[0] if cross else None,
                        "current_price": cross[6] if cross and len(cross) > 6 else None,
                        "sma20": round(cross[4], 4) if cross and len(cross) > 4 and cross[4] else None,
                        "sma50": round(cross[5], 4) if cross and len(cross) > 5 and cross[5] else None,
                        "need": f"SMA(20) must cross above SMA(50) within {golden_cross_buy_days} days"
                    })
                    
        if(len(potential_buys) > 0):
            equity = profile_data.get('equity')
            cash = profile_data.get('cash')
            holdings_data_length = len(holdings_data)
            buy_holdings_succeeded = buy_holdings(potential_buys, cash, equity, holdings_data_length, buy_reasons)
        if(len(sells) > 0):
            file_name = trade_history_file_name
            if debug:
                file_name = "robinhoodbot/tradehistory-debug.json"
            update_trade_history(sells, holdings_data, file_name, sell_reasons)

        print("----- Scanning metric updates and stocks to add to watch list -----\n")

        # Get the metrics report.
        get_accurate_gains(portfolio_symbols, watchlist_symbols, profileData)

        # Remove all from watchlist_symbols if Friday evening.
        # Dont remove watchlist symbols that are in the exclusion list.
        if(reset_watchlist):
            watchlist_symbols_to_remove = get_watchlist_symbols(True)
            remove_watchlist_symbols(watchlist_symbols_to_remove)
        
        print("----- Version " + version + " -----\n")

        # Print API request stats
        api_tracker.print_stats()
        
        # Print cache stats
        cache_stats.print_stats()
        
        # Log scan completion with API stats
        api_stats = api_tracker.get_stats()
        cache_stats_data = cache_stats.get_stats()
        json_logger.log("scan_complete", f"Scan completed - Version {version}", {
            "version": version,
            "sells_count": len(sells),
            "sells": sells,
            "potential_buys_count": len(potential_buys),
            "potential_buys": potential_buys,
            "api_stats": api_stats,
            "cache_stats": cache_stats_data
        })

        print("----- Scan over -----\n")

        # Sign out of the email server.
        server.quit()

        if debug:
            print("----- DEBUG MODE -----\n")

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
        json_logger.log("error", f"IOError: {str(e)}", {"error_type": "IOError", "details": str(sys.exc_info()[0])})
    except Exception as e:
        print("Unexpected error:", str(e))
        json_logger.log("error", f"Unexpected error: {str(e)}", {"error_type": type(e).__name__, "traceback": traceback.format_exc()})

        login_to_sms()
        send_text("Unexpected error:" + str(traceback.format_exc()))
        raise

def traded_today(stock, profileData):
    # If the equity in the account is above the day trading violation
    # rules then return false to disable day trading violation protection.
    if(disable_day_trading_violation_prevention(profileData)):
        return False

    stock_list = rr.get_open_stock_positions()
    for stock_item in stock_list:
        instrument = rsa.get_instrument_by_url(stock_item['instrument'])
        stock_item_creation_date = stock_item['updated_at']
        stock_item_symbol = instrument['symbol']
        # If the stock was traded already and the date it was traded on was today then return true
        if (stock_item_symbol == stock):
            # Or maybe use the intraday properties which may be a better way to tell if a stock was traded
            # in the same day?
            if (stock_item_creation_date.split('T')[0] == datetime.datetime.today().strftime('%Y-%m-%d')):
                print(stock_item_symbol + " was already traded today " + stock_item_creation_date)
                return True

    return False

def take_profit(stock, holdings_data, percentage_limit):
    hours_apart = (datetime.datetime.now() - datetime.datetime.today().replace(hour = 10)).seconds//60//60

    if hours_apart >= 23:
        hours_apart = 1

    minutes_apart = hours_apart * 60
    
    average_buy_price = float(holdings_data[stock]['average_buy_price'])
    price = float(holdings_data[stock]['price'])

    # Perhaps use average buy price and price in holdings_data?
    # If this stock was traded today use the intraday percent change.
    if float(holdings_data[stock]['intraday_percent_change']) > 0.0:
        percent_change = float(holdings_data[stock]['intraday_percent_change'])
        if(percent_change >= percentage_limit):
            message = "Changing the period. " + stock + " has achieved the " + str(percentage_limit) + "% take profit limit at the next possible opportunity."
            print(message)
            return True
    if float(holdings_data[stock]['intraday_percent_change']) < 0.0:
        return False
    elif(percent_increase(stock, percentage_limit, average_buy_price, price)):
        message = "Changing the period. " + stock + " has achieved the " + str(percentage_limit) + "% take profit limit at the next possible opportunity."
        print(message)
        return True
    return False

def get_day_trades(profileData):
    """Gets the day trades that count towards a day trading violation

    :returns: Returns the number of day trades that count towards a day trading violation.  If the user has over $25,000 this method
              returns 0.

    """
    if(disable_day_trading_violation_prevention(profileData)):
        return 0
    else:
        day_trades = rr.get_day_trades()['equity_day_trades']
        return len(day_trades)

def disable_day_trading_violation_prevention(profileData):
    if(float(profileData['equity']) > 25000):
        return True
    else:
        return False

def genetic_generation_add(symbol, is_take_profit):
    if(is_take_profit):
        if not debug:
            rr.post_symbols_to_watchlist(symbol, watch_list_name)
        message = symbol + " has survived the take profit limit and has been added to the current genetic generation."
        print(message)
        send_text(message)
        return True
    else:
        return False

# execute the scan
scan_stocks()
