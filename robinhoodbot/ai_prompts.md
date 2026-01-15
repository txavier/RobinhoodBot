#AI Prompts

/> run the genetic optimizer intraday with 3 stocks

/dev/RobinhoodBot/robinhoodbot/>  python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL --days 60 --generations 15 --population 20 --seed 42

/dev/RobbinhoodBot/robinhoodbot/> python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,INTC,CRM,ORCL,ADBE,PYPL,QCOM,AVGO,CSCO,IBM,MU,SHOP --max-positions 10 --validate-real

/> Give me the best intraday configuration in a table compared to the current config values and also give me your recomendation based on your analysis of the genetic optimizers best intraday configuration and all the logs that have been taken thus far.
/> log these changes in the ai_changelog