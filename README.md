# RobinhoodBot
Trading bot for Robinhood accounts

For more info:
https://medium.com/@kev.guo123/building-a-robinhood-stock-trading-bot-8ee1b040ec6a


5/1/19: Since Robinhood has updated it's API, you now have to enter a 2 factor authentication code whenever you run the script. To do this, go to the Robinhood mobile app and enable two factor authentication in your settings. You will now receive an SMS code when you run the script, which you have to enter into the script.



This project supports Python 3.7+


To Install:

```bash
git clone https://github.com/txavier/RobinhoodBot.git
cd RobinhoodBot/
pip install -r requirements.txt
cp config.py.sample config.py # add auth info and watchlist name to monitor after copying
```

To Run:
In RobinHood create a watchlist named "Exclusion".  This will be the watchlist that you will use to tell the bot to ignore the stock tickers contained within.

```python
cd RobinboodBot/robinhoodbot (If outside of root directory)
python3 main.py
```

To loop: 1 once an hour

```python
cd RobinboodBot/robinhoodbot (If outside of root directory)
./run.sh # uses bash
```

# Changes to robin_stocks library
 - Added @cache to def get_name_by_symbol(symbol):
 - Added @cache to def get_name_by_url(url):
 - Added @cache to def get_symbol_by_url(url):
 - Added None Parameter to order
Afterwards, be sure to run /> pip install . 

# VENV
## In the event you want to use Venv instead of anaconda, activate it with the follwing command.
venv:
/home/theo/dev/.venv/bin/activate
~/> source dev/.venv/bin/activate

