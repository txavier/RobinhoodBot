#AI Prompts

/> run the genetic optimizer intraday with 3 stocks

/robinhoodbot/>  python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL --days 60 --generations 15 --population 20 --seed 42 

/robinhoodbot/> python genetic_optimizer_intraday.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,INTC,CRM,ORCL,ADBE,PYPL,QCOM,AVGO,CSCO,IBM,MU,SHOP --max-positions 10 --generations 20 --population 30 --validate-real --use-ray --resume

/> cd /home/theo/dev/RobinhoodBot/robinhoodbot && source /home/theo/dev/.venv/bin/activate && python3 genetic_optimizer_intraday.py --symbols AAPL,MSFT,NVDA --max-positions 10 --generations 20 --population 30 --resume --validate-real --real-data

/> Give me the best configuration found in a table compared to the current config values and also give me your recomendation based on your analysis of the genetic optimizers best intraday configuration and all the logs, ai_suggested_config_changelog.json, buy_reasons.json, genetic_optimization_intraday_result.json, log.json, tradehistory-real.json, that have been taken thus far.
/> log these changes in the ai_changelog

/> Analyze all the logs, including  ai_suggested_config_changelog.json, buy_reasons.json, genetic_optimization_intraday_result.json, log.json, tradehistory-real.json.  Analyze today's entries to find the effect of the config modifications recommended by the genetic_optimizer.  Compare it to previous logs to find out all of the positive effects and negative effects.

Tuning the optimizer.
/> what are ranges for the values evaluated inthe genetic optimizer?
/> Based on the optimizer best values which ranges should be adjusted in order to theoretically give a possibility for better results?