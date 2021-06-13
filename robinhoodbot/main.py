import robin_stocks as r
import robin_stocks.robinhood as rr
import pandas as pd
import numpy as np
import ta as t
import smtplib
import sys
import datetime
import traceback
import time
from pandas.plotting import register_matplotlib_converters
from misc import *
from tradingstats import *
from config import *
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from scipy.stats import linregress

# Safe divide by zero division function


def safe_division(n, d):
    return n / d if d else 0


def login_to_sms():
    global sms_gateway
    global server
    
    # Log in to Robinhood
    sms_gateway = rh_phone + '@' + rh_company_url  # Phone number to send SMS
    server = smtplib.SMTP("smtp.gmail.com", 587)  # Gmail SMTP server
    server.starttls()
    server.login(rh_email, rh_mail_password)


def send_text(message):
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

def isInExclusionList(symbol):
    """
    Returns true if the symbol is in the exclusion list.
    """
    result = False
    if use_exclusion_watchlist:
        exclusion_list = rr.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
    for exclusion_item in exclusion_list['results']:
            if exclusion_item['symbol'] == symbol:
                result = True
                return result
    return result


def get_watchlist_symbols():
    """
    Returns: the symbol for each stock in your watchlist as a list of strings
    """
    exclusion_list = []
    symbols = []
    list = rr.get_watchlist_by_name(name=watch_list_name)
    # Remove any exclusions.
    if use_exclusion_watchlist:
        exclusion_list = rr.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
    skip = False
    for item in list['results']:
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
        instrument_data = rr.get_instrument_by_url(item.get('instrument'))
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
    # Night
    begin_time = datetime.time(21, 00)
    end_time = datetime.time(23, 00)
    timenow = datetime.datetime.now().time()
          
    if(timenow >= begin_time and timenow < end_time and datetime.datetime.today().weekday() == 4):
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
    instrument = rr.get_instruments_by_symbols(symbol)
    url = instrument[0].get('url')
    for dict in holdings_data:
        if(dict.get('instrument') == url):
            return dict.get('created_at')
    return "Not found"


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
    return holdings


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
        return 0,0,0
    index -= 1
    while(index >= 0 and found == lastIndex and not np.isnan(shortTerm.at[index]) and not np.isnan(LongTerm.at[index])
          and ((pd.Timestamp("now", tz='UTC') - dates.at[index]) <= pd.Timedelta(str(days) + " days"))):
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
        return (1 if recentDiff else -1), prices.at[found], prices.at[lastIndex]
    else:
        return 0,0,0


def five_year_check(stockTicker):
    """Figure out if a stock has risen or been created within the last five years.

    Args:
        stockTicker(str): Symbol of the stock we're querying

    Returns:
        True if the stock's current price is higher than it was five years ago, or the stock IPO'd within the last five years
        False otherwise
    """
    instrument = rr.get_instruments_by_symbols(stockTicker)
    if(len(instrument) == 0):
        return False

    list_date = instrument[0].get("list_date")
    # If there is no list date then assume that the stocks list date data
    # is just missing i.e. NNOX
    if list_date == None:
        return True
    if ((pd.Timestamp("now") - pd.to_datetime(list_date)) < pd.Timedelta("5 Y")):
        return True
    fiveyear = rr.get_stock_historicals(
        stockTicker, interval='day', span='5year', bounds='regular')
    closingPrices = []
    for item in fiveyear:
        closingPrices.append(float(item['close_price']))
    recent_price = closingPrices[len(closingPrices) - 1]
    oldest_price = closingPrices[0]
    # if(recent_price <= oldest_price and verbose == True):
    #     print("The stock " + stockTicker + " IPO'd, more than 5 years ago, on " + list_date +
    #           " with a price 5 years ago of " + str(oldest_price) +
    #           " and a current price of " + str(recent_price) + "\n")
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
    yearCheck = five_year_check(stockTicker)

    if(direction == "above" and not yearCheck):
        return False,0,0

    history = rr.get_stock_historicals(stockTicker, interval='hour', span='3month', bounds='regular')
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
    return cross[0], cross[1], cross[2], history[len(history)-5]['close_price']


def sell_holdings(symbol, holdings_data):
    """ Place an order to sell all holdings of a stock.

    Args:
        symbol(str): Symbol of the stock we want to sell
        holdings_data(dict): dict obtained from get_modified_holdings() method
    """
    shares_owned = int(float(holdings_data[symbol].get("quantity")))
    if not debug:
        rr.order_sell_market(symbol, shares_owned)
    print("####### Selling " + str(shares_owned) +
          " shares of " + symbol + " #######")
    send_text("SELL: \nSelling " + str(shares_owned) + " shares of " + symbol)

def buy_holdings(potential_buys, profile_data, holdings_data):
    """ Places orders to buy holdings of stocks. This method will try to order
        an appropriate amount of shares such that your holdings of the stock will
        roughly match the average for the rest of your portfoilio. If the share
        price is too high considering the rest of your holdings and the amount of
        buying power in your account, it will not order any shares.
    Args:
        potential_buys(list): List of strings, the strings are the symbols of stocks we want to buy
        symbol(str): Symbol of the stock we want to sell
        holdings_data(dict): dict obtained from rr.build_holdings() or get_modified_holdings() method
    """
    cash = float(profile_data.get('cash'))
    portfolio_value = float(profile_data.get('equity')) - cash
    ideal_position_size = (safe_division(portfolio_value, len(holdings_data))+cash/len(potential_buys))/(2 * len(potential_buys))
    prices = rr.get_latest_price(potential_buys)
    for i in range(0, len(potential_buys)):
        stock_price = float(prices[i])
        if ((stock_price * int(ideal_position_size/stock_price)) > cash):
            num_shares = int(ideal_position_size/stock_price)
            output = "Tried buying " + str(num_shares) + " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " but with only ${:.2f}".format(cash) + " in cash not enough to make this purchase."
            print(output)
            if (len(potential_buys) > 1):
                ideal_position_size = (safe_division(portfolio_value, len(holdings_data))+cash/(len(potential_buys)-1))/(2 * (len(potential_buys)-1))
            continue
        elif ((stock_price * int(ideal_position_size*1.5/stock_price)) > cash):
            num_shares = int(ideal_position_size*1.5/stock_price)
            output = "Tried buying " + str(num_shares) + " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " but with only ${:.2f}".format(cash) + " in cash not enough to make this purchase."
            print(output)
            if (len(potential_buys) > 1):
                ideal_position_size = (safe_division(portfolio_value, len(holdings_data))+cash/(len(potential_buys)-1))/(2 * (len(potential_buys)-1))
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
                ideal_position_size = (safe_division(portfolio_value, len(holdings_data))+cash/(len(potential_buys)-1))/(2 * (len(potential_buys)-1))
            continue

        print("####### Buying " + str(num_shares) +
                " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) +  " with ${:.2f}".format(cash) + " in cash. #######")

        message = "BUY: \nBuying " + str(num_shares) + " shares of " + potential_buys[i] + " at " + str(stock_price) + " costing ${:.2f}".format(stock_price * num_shares) + " with ${:.2f}".format(cash) 

        if not debug:
            result = rr.order_buy_market(potential_buys[i], num_shares)
            if 'detail' in result:
                print(result['detail'])
                message = message +  ". The result is " + result['detail']
        send_text(message)

def is_market_in_uptrend():
    stockTickerNdaq = 'NDAQ'
    stockTickerDow = 'DIA'
    stockTickerSP = 'SPY'
    uptrendNdaq = False
    uptrendDow = False
    uptrendSp = False
    # Nasdaq
    # Using NasDaq as the market uptrend indicator which does not have extended trading hours.
    today_history = rr.get_stock_historicals(stockTickerNdaq, interval='5minute', span='day', bounds='regular')    
    if(float(today_history[0]['open_price']) < float(today_history[len(today_history) - 1]['close_price'])):
        uptrendNdaq = True
    # DOW
    # Using Dow as the market uptrend indicator.
    today_history = rr.get_stock_historicals(stockTickerDow, interval='5minute', span='day', bounds='regular')    
    if(float(today_history[0]['open_price']) < float(today_history[len(today_history) - 1]['close_price'])):
        uptrendDow = True
    # S&P Index
    # Using S&P as the market uptrend indicator.
    # day_trades = rr.get_day_trades()
    today_history = rr.get_stock_historicals(stockTickerSP, interval='5minute', span='day', bounds='regular')    
    if(float(today_history[0]['open_price']) < float(today_history[len(today_history) - 1]['close_price'])):
        uptrendSp = True
    
    result = (uptrendNdaq + uptrendDow + uptrendSp) >= 2
    return result

def get_accurate_gains(portfolio_symbols, watchlist_symbols):
    '''
    Robinhood includes dividends as part of your net gain. This script removes
    dividends from net gain to figure out how much your stocks/options have paid
    off.
    Note: load_portfolio_profile() contains some other useful breakdowns of equity.
    Print profileData and see what other values you can play around with.
    '''

    profileData = rr.load_portfolio_profile()
    allTransactions = rr.get_bank_transfers()
    cardTransactions= rr.get_card_transactions()

    deposits = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'completed'))
    withdrawals = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'withdraw') and (x['state'] == 'completed'))
    debits = sum(float(x['amount']['amount']) for x in cardTransactions if (x['direction'] == 'debit' and (x['transaction_type'] == 'settled')))
    reversal_fees = sum(float(x['fees']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'reversed'))

    money_invested = deposits + reversal_fees - (withdrawals - debits)
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
    gainIncrease = "The net worth has increased {:0.3f}% due to other gains that amount to ${:0.2f}".format(percentGain, totalGainMinusDividends)

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
        if market_tag_report[0] != '':
            send_text(market_tag_report[0])
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)
        print("----- End market reports scan -----")

    # Evening Morning report
    begin_time = datetime.time(8, 30)
    end_time = datetime.time(9, 30)
    timenow = datetime.datetime.now().time()

    if(timenow >= begin_time and timenow < end_time):
        print("Sending morning report.")
        send_text(bankTransfered + "\n" + withdrawable_amount)
        time.sleep(2)
        send_text(equity)      
        time.sleep(2)
        send_text(equityAndWithdrawable + "\n" + gainIncrease)
        # Get interesting stocks report.
        market_tag_report = get_market_tag_stocks_report()
        if market_tag_report[0] != '':
            # If the market tag report has some stock values...
            send_text(market_tag_report[0])
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)

    # Evening report
    begin_time = datetime.time(17, 30)
    end_time = datetime.time(18, 30)

    if(timenow >= begin_time and timenow < end_time):
        print("Sending evening report.")
        send_text(bankTransfered + "\n" + withdrawable_amount)
        time.sleep(2)
        send_text(equity)      
        time.sleep(2)
        send_text(equityAndWithdrawable + "\n" + gainIncrease)
        # Get interesting stocks report.
        market_tag_report = get_market_tag_stocks_report()
        if market_tag_report[0] != '':
            # If the market tag report has some stock values...
            send_text(market_tag_report[0])
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
        if market_tag_report[0] != '':
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
        if market_tag_report[0] != '':
            if market_report_auto_invest:
                auto_invest(market_tag_report[1], portfolio_symbols, watchlist_symbols)
        print("----- End market reports scan -----") 

def sudden_drop(symbol, percent, hours_apart):
    """ Return true if the price drops more than the percent argument in the span of two hours.

    Args:
        symbol(str): The symbol of the stock.
        percent(float): The amount of percentage drop from the previous close price.
        hours_apart(float): Number of hours away from the current to check.

    Returns:
        True if there is a sudden drop.
    """
    historicals = rr.get_stock_historicals(symbol, interval='hour', span='month')
    if len(historicals) == 0:
        return False
        
    percentage = (percent/100) * float(historicals[len(historicals) - 1 - hours_apart]['close_price'])
    target_price = float(historicals[len(historicals) - 1 - hours_apart]['close_price']) - percentage

    if float(historicals[len(historicals) - 1]['close_price']) <= target_price:
        message = "The " + symbol + " has dropped from " + str(float(historicals[len(historicals) - 1 - hours_apart]['close_price'])) + " to " + str(float(historicals[len(historicals) - 1]['close_price'])) + " which is more than " + str(percent) + "% (" + str(target_price) + ") in the span of " + str(hours_apart) + " hour(s)."
        print(message)
        send_text(message)
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
        exclusion_list = rr.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
        stock_array_copy = stock_array.copy()
        for stock in stock_array:
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
                stock_info = rr.get_instruments_by_symbols(stock)
                if (not stock_info[0]['tradeable']):
                    if stock in stock_array_copy:
                        stock_array_copy.remove(stock)
                        removed = True
                        print(stock + " removed from auto-invest because RobinHood has marked this stock as untradeable.")
            fundamentals = rr.get_fundamentals(stock)
            if (not removed):
                average_volume = float(fundamentals[0]['average_volume'])
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
                history = rr.get_stock_historicals(stock, interval='day')
                if (len(history) == 0 or float(history[len(history) - 1]['close_price']) > price_cap):
                    if stock in stock_array_copy:
                        stock_array_copy.remove(stock)
                        removed = True
                        if (len(history) == 0):
                            print(stock + " removed from auto-invest because it has no stock history to analyze.")
                        else:
                            print(stock + " removed from auto-invest because its price of " + str(float(history[len(history) - 1]['close_price'])) + " was greater than your price cap of " + str(price_cap))

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
        # send_text("Unexpected error could not generate interesting stocks report:" + str(e))

def find_symbol_with_greatest_slope(stock_array):
    linregressResults = []
    for stockTicker in stock_array:
        # Load stock numbers.
        history = rr.get_stock_historicals(stockTicker, interval='5minute', span='day', bounds='regular')
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
        volumes = rr.get_stock_historicals(stock, interval='day', span='week', bounds='regular', info='volume')
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
            print(market_tag_for_report_item + str(len(all_market_tag_stocks)))
            for market_tag_stock in all_market_tag_stocks:
                cross = golden_cross(market_tag_stock, n1=21, n2=50, days=10, direction="above")
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
            history = rr.get_stock_historicals(stockTicker, interval='5minute', span='day', bounds='regular')
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

    profile_data['equity'] = pheonix_account['total_equity']['amount']
    if (pheonix_account['total_extended_hours_equity']):
        profile_data['extended_hours_equity'] = pheonix_account['total_extended_hours_equity']['amount']
    profile_data['cash'] = pheonix_account['uninvested_cash']['amount']

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
        # Put your username and password in a config.py file in the same directory (see sample file)
        login = rr.login(rh_username, rh_password)
        login_to_sms()

        if debug:
            print("----- DEBUG MODE -----\n")

        print("----- Starting scan... -----\n")
        register_matplotlib_converters()
        watchlist_symbols = get_watchlist_symbols()
        portfolio_symbols = get_portfolio_symbols()
        holdings_data = get_modified_holdings()
        potential_buys = []
        sells = []
        print("Current Portfolio: " + str(portfolio_symbols) + "\n")
        print("Current Watchlist: " + str(watchlist_symbols) + "\n")
        print("----- Scanning portfolio for stocks to sell -----\n")
        market_uptrend = is_market_in_uptrend()
        open_stock_orders = []
        for symbol in portfolio_symbols:
            is_sudden_drop = sudden_drop(symbol, 10, 2) or sudden_drop(symbol, 15, 1)
            cross = golden_cross(symbol, n1=21, n2=50, days=30, direction="below")
            if(cross[0] == -1 or is_sudden_drop):
                open_stock_orders = rr.get_all_open_stock_orders()
                # If there are any open stock orders then dont buy more.  This is to avoid 
                # entering multiple orders of the same stock if the order has not yet between
                # filled.
                if(len(open_stock_orders) == 0):
                    day_trades = rr.get_day_trades()['equity_day_trades']
                    if len(day_trades) <= 1:
                        if (not isInExclusionList(symbol)):
                            send_text("Attempting to sell " + symbol)
                            sell_holdings(symbol, holdings_data)
                            sells.append(symbol)
                        else:
                            print("Unable to sell " + symbol + " is in the exclusion list.")
                    else:
                        print("Unable to sell " + symbol + " because there are " + str(len(day_trades)) + " day trades.")
                else:
                    print("Unable to sell " + symbol + " because there are open stock orders.")
        profile_data_with_dividend_total = rr.build_user_profile()
        profile_data = build_pheonix_profile_data(profile_data_with_dividend_total)
        ordered_watchlist_symbols = order_symbols_by_slope(watchlist_symbols)
        print("\n----- Scanning watchlist for stocks to buy -----\n")
        for symbol in ordered_watchlist_symbols:
            if(symbol not in portfolio_symbols):
                cross = golden_cross(symbol, n1=21, n2=50, days=10, direction="above")
                if(cross[0] == 1):
                    open_stock_orders = rr.get_all_open_stock_orders()
                    # If there are any open stock orders then dont buy more.  This is to avoid 
                    # entering multiple orders of the same stock if the order has not yet between
                    # filled.
                    if(len(open_stock_orders) == 0):
                        # If the current price is greater than the price at cross,
                        # meaning that the price is still rising then buy.
                        if(float(cross[2]) > float(cross[1])):
                            # If the current price is greater than the price 5 hours ago,
                            # meaning we have less of a chance of the stock showing a 
                            # death cross soon then buy.
                            if(float(cross[2]) > float(cross[3])):
                                if(market_uptrend):
                                    day_trades = rr.get_day_trades()['equity_day_trades']
                                    if len(day_trades) <= 1:
                                        potential_buys.append(symbol)
                                    else:
                                        print("Unable to buy " + symbol + " because there are " + str(len(day_trades)) + " day trades.")
                                else:
                                    print("But the markets on average are not in an uptrend.")
                            else:
                                print("But the price is lower than it was 5 hours ago.")
                        else:
                            print("But the price is lower than it was when the golden cross formed " + str(cross[2]) + " < " + str(cross[1]))
                    else:
                        pending_order_message = "But there are " + str(len(open_stock_orders)) + " current pending orders."
                        print(pending_order_message)
                        # send_text("Wanted to buy " + symbol + ". " + pending_order_message)
        if(len(potential_buys) > 0):
            buy_holdings_succeeded = buy_holdings(potential_buys, profile_data, holdings_data)
        if(len(sells) > 0):
            file_name = trade_history_file_name
            if debug:
                file_name = "robinhoodbot/tradehistory-debug.txt"
            update_trade_history(sells, holdings_data, file_name)

        # Get the metrics report.
        get_accurate_gains(portfolio_symbols, watchlist_symbols)

        # Remove all from watchlist_symbols if Friday evening.
        if(reset_watchlist):
            remove_watchlist_symbols(watchlist_symbols)
        
        print("----- Scan over -----\n")

        # Sign out of the email server.
        server.quit()

        if debug:
            print("----- DEBUG MODE -----\n")

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    except Exception as e:
        print("Unexpected error:", str(e))

        login_to_sms()
        send_text("Unexpected error:" + str(e))
        raise

# execute the scan
scan_stocks()
