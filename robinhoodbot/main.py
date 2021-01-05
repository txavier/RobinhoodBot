import robin_stocks as r
import pandas as pd
import numpy as np
import ta as t
import smtplib
import sys
import datetime
import traceback
from pandas.plotting import register_matplotlib_converters
from misc import *
from tradingstats import *
from config import *
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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


def get_watchlist_symbols():
    """
    Returns: the symbol for each stock in your watchlist as a list of strings
    """
    my_list_names = []
    symbols = []
    list = r.get_watchlist_by_name(name=watch_list_name)
    # Remove any exclusions.
    if use_exclusion_watchlist:
        exclusion_list = r.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
        for list_item in list['results']:
            for exclusion_item in exclusion_list['results']:
                if exclusion_item['symbol'] == list_item['symbol']:
                    list_item.remove(exclusion_item['symbol'])
    for item in list['results']:
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
    holdings_data = r.get_open_stock_positions()
    for item in holdings_data:
        if not item:
            continue
        instrument_data = r.get_instrument_by_url(item.get('instrument'))
        symbol = instrument_data['symbol']
        symbols.append(symbol)
    return symbols


def get_position_creation_date(symbol, holdings_data):
    """Returns the time at which we bought a certain stock in our portfolio

    Args:
        symbol(str): Symbol of the stock that we are trying to figure out when it was bought
        holdings_data(dict): dict returned by r.get_current_positions()

    Returns:
        A string containing the date and time the stock was bought, or "Not found" otherwise
    """
    instrument = r.get_instruments_by_symbols(symbol)
    url = instrument[0].get('url')
    for dict in holdings_data:
        if(dict.get('instrument') == url):
            return dict.get('created_at')
    return "Not found"


def get_modified_holdings():
    """ Retrieves the same dictionary as r.build_holdings, but includes data about
        when the stock was purchased, which is useful for the read_trade_history() method
        in tradingstats.py

    Returns:
        the same dict from r.build_holdings, but with an extra key-value pair for each
        position you have, which is 'bought_at': (the time the stock was purchased)
    """
    holdings = r.build_holdings()
    holdings_data = r.get_open_stock_positions()
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
        return 0
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
        return (1 if recentDiff else -1)
    else:
        return 0


def five_year_check(stockTicker):
    """Figure out if a stock has risen or been created within the last five years.

    Args:
        stockTicker(str): Symbol of the stock we're querying

    Returns:
        True if the stock's current price is higher than it was five years ago, or the stock IPO'd within the last five years
        False otherwise
    """
    instrument = r.get_instruments_by_symbols(stockTicker)
    if(len(r.get_instruments_by_symbols(stockTicker)) == 0):
        return False

    list_date = instrument[0].get("list_date")
    if ((pd.Timestamp("now") - pd.to_datetime(list_date)) < pd.Timedelta("5 Y")):
        return True
    fiveyear = r.get_stock_historicals(
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
        1 if the short-term indicator crosses above the long-term one
        0 if there is no cross between the indicators
        -1 if the short-term indicator crosses below the long-term one
        False if direction == "above" and five_year_check(stockTicker) returns False, meaning that we're considering whether to
            buy the stock but it hasn't risen overall in the last five years, suggesting it contains fundamental issues
    """
    """ Apparently 5 year historicals are no longer available with hourly intervals.  Only with day intervals now.
    """
    yearCheck = five_year_check(stockTicker)

    if(direction == "above" and not yearCheck):
        return False

    history = r.get_stock_historicals(stockTicker, interval='hour', span='3month', bounds='regular')
    closingPrices = []
    dates = []
    for history_item in history:
        closingPrices.append(float(history_item['close_price']))
        dates.append(history_item['begins_at'])

    # If we are in extended hours then add extended hours close_prices.
    # extended_hours = False
    # begin_time = datetime.time(17, 30)
    # end_time = datetime.time(9, 0)
    # timenow = datetime.datetime.now().time()

    # if(timenow >= begin_time or timenow < begin_time):
    #     extended_hours = True

    # if(extended_hours):
    #     today_history = r.get_stock_historicals(stockTicker, interval='hour', span='day', bounds='extended')
    #     for today_history_item in today_history:
    #         closingPrices.append(float(today_history_item['close_price']))
    #         dates.append(today_history_item['begins_at'])

    price = pd.Series(closingPrices)
    dates = pd.Series(dates)
    dates = pd.to_datetime(dates)
    sma1 = t.volatility.bollinger_mavg(price, n=int(n1), fillna=False)
    sma2 = t.volatility.bollinger_mavg(price, n=int(n2), fillna=False)
    series = [price.rename("Price"), sma1.rename(
        "Indicator1"), sma2.rename("Indicator2"), dates.rename("Dates")]
    df = pd.concat(series, axis=1)
    cross = get_last_crossing(
        df, days, symbol=stockTicker, direction=direction)
    # if(verbose == True and cross == 1 and direction == "above" and yearCheck):
    #     print("We're considering whether to buy the " + stockTicker +
    #               " but it hasn't risen overall in the last 5 years and it hasn't IPO'd in the last 5 years, suggesting it contains fundamental issues.\n")
    if(plot):
        show_plot(price, sma1, sma2, dates, symbol=stockTicker,
                  label1=str(n1)+" day SMA", label2=str(n2)+" day SMA")
    return cross


def sell_holdings(symbol, holdings_data):
    """ Place an order to sell all holdings of a stock.

    Args:
        symbol(str): Symbol of the stock we want to sell
        holdings_data(dict): dict obtained from get_modified_holdings() method
    """
    shares_owned = int(float(holdings_data[symbol].get("quantity")))
    if not debug:
        r.order_sell_market(symbol, shares_owned)
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
        holdings_data(dict): dict obtained from r.build_holdings() or get_modified_holdings() method

    Returns: 
        False if order has not been placed because there was not enough buying power.
    """
    cash = float(profile_data.get('cash'))
    portfolio_value = float(profile_data.get('equity')) - cash
    ideal_position_size = (safe_division(portfolio_value, len(
        holdings_data))+cash/len(potential_buys))/(2 * len(potential_buys))
    prices = r.get_latest_price(potential_buys)
    buying_power = r.load_account_profile(info='buying_power')
    order_placed = False
    for i in range(0, len(potential_buys)):
        stock_price = float(prices[i])
        if (float(buying_power) < ideal_position_size):
            output = "####### Tried buying shares of " + potential_buys[i] + ", but at ${:.2f}".format(ideal_position_size) + " your account balance of ${:.2f}".format(float(buying_power)) + " is not enough buying power to do so#######"
            print(output)
            break
        elif(ideal_position_size < stock_price < ideal_position_size*1.5):
            num_shares = int(ideal_position_size*1.5/stock_price)
        elif (stock_price < ideal_position_size):
            num_shares = int(ideal_position_size/stock_price)
        else:
            output = "####### Tried buying shares of " + potential_buys[i] + ", but not enough buying power to do so#######"
            print(output)
            send_text(output)
            break
        print("####### Buying " + str(num_shares) +
              " shares of " + potential_buys[i] + " #######")

        send_text("Attempting to buy " + potential_buys[i])

        message = "BUY: \nBuying " + str(num_shares) + " shares of " + potential_buys[i]

        if not debug:
            result = r.order_buy_market(potential_buys[i], num_shares)
            if 'detail' in result:
                message = message +  ". The result is " + result['detail']
        order_placed = True
        send_text(message)
    return order_placed
    

def get_accurate_gains(portfolio_symbols):
    '''
    Robinhood includes dividends as part of your net gain. This script removes
    dividends from net gain to figure out how much your stocks/options have paid
    off.
    Note: load_portfolio_profile() contains some other useful breakdowns of equity.
    Print profileData and see what other values you can play around with.
    '''

    profileData = r.load_portfolio_profile()
    allTransactions = r.get_bank_transfers()
    cardTransactions= r.get_card_transactions()

    deposits = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'completed'))
    withdrawals = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'withdraw') and (x['state'] == 'completed'))
    debits = sum(float(x['amount']['amount']) for x in cardTransactions if (x['direction'] == 'debit' and (x['transaction_type'] == 'settled')))
    reversal_fees = sum(float(x['fees']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'reversed'))

    money_invested = deposits + reversal_fees - (withdrawals - debits)
    dividends = r.get_total_dividends()
    percentDividend = dividends/money_invested*100

    equity_amount = float(profileData['equity'])
    buying_power = float(profileData['equity']) - float(profileData['market_value'])
    totalGainMinusDividends = equity_amount - dividends - money_invested + buying_power
    percentGain = totalGainMinusDividends/money_invested*100

    bankTransfered = "The total money invested is {:.2f}".format(money_invested)
    equity = "The total equity is {:.2f}".format(equity_amount)
    withdrawable_amount = "The buying power is {:.2f}".format(buying_power)
    equityAndWithdrawable = "For a total account value of  {:.2f}".format(float(equity_amount + buying_power))
    dividendIncrease = "The net worth has increased {:0.2}% due to dividends that amount to {:0.2f}".format(percentDividend, dividends)
    gainIncrease = "The net worth has increased {:0.3}% due to other gains that amount to {:0.2f}".format(percentGain, totalGainMinusDividends)

    print(bankTransfered)
    print(equity)
    print(withdrawable_amount)
    print(invested)
    print(equity)
    print(equityAndWithdrawable)
    print(dividendIncrease)
    print(gainIncrease)

    """ Send a text message with the days metrics """

    # Evening Morning report
    begin_time = datetime.time(8, 30)
    end_time = datetime.time(9, 0)
    timenow = datetime.datetime.now().time()

    if debug: 
        market_tag_report = get_market_tag_stocks_report()
        send_text(market_tag_report[0])
        if market_report_auto_invest:
            auto_invest(market_tag_report[1], portfolio_symbols)
            
    if(timenow >= begin_time and timenow < end_time):
        print("Sending morning report.")
        send_text(bankTransfered + "\n" +
                  withdrawable_amount)
        send_text(equity + "\n" + equityAndWithdrawable + "\n" + gainIncrease)
        # Get interesting stocks report.
        market_tag_report = get_market_tag_stocks_report()
        send_text(market_tag_report[0])
        if market_report_auto_invest:
            auto_invest(market_tag_report[1], portfolio_symbols)

    # Evening report
    begin_time = datetime.time(17, 30)
    end_time = datetime.time(18, 0)

    if(timenow >= begin_time and timenow < end_time):
        print("Sending evening report.")
        send_text(bankTransfered + "\n" +
                  withdrawable_amount)
        send_text(equity + "\n" + equityAndWithdrawable + "\n" + gainIncrease)
        # Get interesting stocks report.
        market_tag_report = get_market_tag_stocks_report()
        send_text(market_tag_report[0])

def auto_invest(stock_array, portfolio_symbols):
    try:
        invest = True

        # If the previous stock that we added to the watchlist is still here
        # or the stock is in an exclusion list if one has been set
        # then dont auto invest any other stocks for now to prevent just adding
        # all stocks to the investment pool thus diluting the investment potential
        # in the previous stock that has been autoinvested.
        exclusion_list = r.get_watchlist_by_name(name=auto_invest_exclusion_watchlist)
        for stock in stock_array:
            if (stock in portfolio_symbols):
                invest = False
                print(stock + " is still in the recomended list. Auto-Invest will skip this interval in order to allow time between stock generation.")
            if (use_exclusion_watchlist):
                for exclusion_result in exclusion_list['results']:
                    if (stock == exclusion_result['symbol']):
                        stock_array.remove(stock)

        if (invest):
            # Lowest price.
            # symbol_and_price = find_symbol_with_lowest_price(stock_array)
            # selected_symbol = symbol_and_price[0]
            # lowest_price = symbol_and_price[1]
            # message = "Auto-Invest is adding " + selected_symbol + " at ${:.2f}".format(lowest_price) + " to the " + watch_list_name + " watchlist."

            # Highest volume.
            selected_symbol = find_symbol_with_highest_volume(stock_array)
            message = "Auto-Invest is adding " + selected_symbol + " to the " + watch_list_name + " watchlist."

            send_text(message)
            print(message)
            if not debug:
                r.post_symbols_to_watchlist(selected_symbol, watch_list_name)

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    except ValueError:
        print("Could not convert data to an integer.")
    except Exception as e:
        print("Unexpected error could not generate interesting stocks report:", str(e))

        login_to_sms()
        send_text(
            "Unexpected error could not generate interesting stocks report:" + str(e) + "\n Trace: " + traceback.print_exc())

def find_symbol_with_highest_volume(stock_array):
    volume_array = []
    for stock in stock_array:
        volumes = r.get_stock_historicals(stock, interval='day', span='week', bounds='regular', info='volume')
        volume_array.append(volumes[len(volumes) - 1])
    stock_and_volume_float_array = [float(i) for i in volume_array]
    sorted_volume_array = sorted(stock_and_volume_float_array, key=float)
    highest_volume = sorted_volume_array[len(sorted_volume_array) - 1]
    # Convert the string price array to float and find the index of the 
    # stock with the lowest price.
    index_of_highest_volume = [float(i) for i in volume_array].index(highest_volume)
    symbol_of_highest_volume = stock_array[index_of_highest_volume]
    return symbol_of_highest_volume

def find_stock_with_lowest_price(stock_array):
    # Find stock with the lowest stock price.
    price_array = r.get_latest_price(stock_array)
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
            all_market_tag_stocks = r.get_all_stocks_from_market_tag(market_tag_for_report_item, info = 'symbol')
            for market_tag_stock in all_market_tag_stocks:
                cross = golden_cross(market_tag_stock, n1=34, n2=84, days=10, direction="above")
                if(cross == 1):
                    report_string = report_string + " \n " + market_tag_stock
                    stock_array.append(market_tag_stock)

        if(report_string != ""):
            return market_tag_for_report + " \n\n " + report_string, stock_array
        return ""

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    except ValueError:
        print("Could not convert data to an integer.")
    except Exception as e:
        print("Unexpected error could not generate interesting stocks report:", str(e))

        login_to_sms()
        send_text(
            "Unexpected error could not generate interesting stocks report:" + str(e) + "\n Trace: " + traceback.print_exc())


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
        login = r.login(rh_username, rh_password)
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
        for symbol in portfolio_symbols:
            cross = golden_cross(symbol, n1=34, n2=84, days=30, direction="below")
            if(cross == -1):
                send_text("Attempting to sell " + symbol)
                sell_holdings(symbol, holdings_data)
                sells.append(symbol)
        profile_data = r.build_user_profile()
        print("\n----- Scanning watchlist for stocks to buy -----\n")
        for symbol in watchlist_symbols:
            # If more money has been added then strengthen position of well performing portfolio holdings if the funds allow.
            # the below has been commented out to make the algorithm less aggressive in fear of violating day-trading policies.
            # if(symbol in portfolio_symbols):
            #     cross = golden_cross(symbol, n1=34, n2=84, days=10, direction="above")
            #     if(cross == 1):
            #         potential_buys.append(symbol)
            #         if(verbose == True):
            #             print("Strengthen position of " + symbol +
            #                   " as the golden cross is within 10 days.")
            if(symbol not in portfolio_symbols):
                cross = golden_cross(symbol, n1=34, n2=84, days=10, direction="above")
                if(cross == 1):
                    potential_buys.append(symbol)
        if(len(potential_buys) > 0):
            buy_holdings_succeeded = buy_holdings(potential_buys, profile_data, holdings_data)
            if buy_holdings_succeeded:
                new_holdings = get_modified_holdings()
                update_trade_history(potential_buys, new_holdings, trade_history_file_name)
        if(len(sells) > 0):
            update_trade_history(sells, holdings_data, trade_history_file_name)

        # Get the metrics report.
        get_accurate_gains(portfolio_symbols)
        
        print("----- Scan over -----\n")

        # Sign out of the email server.
        server.quit()

        if debug:
            print("----- DEBUG MODE -----\n")

    except IOError as e:
        print(e)
        print(sys.exc_info()[0])
    # except ValueError:
    #     print("Could not convert data to an integer.")
    except Exception as e:
        print("Unexpected error:", str(e))

        login_to_sms()
        send_text("Unexpected error:" + str(e))
        raise

# execute the scan
scan_stocks()
