# "Insider Training": An Exploration of Insider Trades
According to [Investopedia](https://www.investopedia.com/terms/i/insider.asp), an **insider** is "anyone who trades a company's shares based on material nonpublic knowledge."

In this repository, we try to see if we can mine insider data and use machine learning to generate investment insights, in lieu of having any technical knowledge of investment strategies, the Greeks, etc. I know what "bullish" and "bearish" mean, but not much more than that :)

This project is neither an investment pitch, nor does it constitute financial advice. It's an exploratoration driven by data and curiosity!

### Where can the data be found?
Insider trade filings are published online by the SEC, and they can be found using the EDGAR API. However, it is much easier to pull data from sites that gather the filing data into tabular form such as [Benzinga](https://www.benzinga.com/).

Historic ticker data (e.g. daily opening, low, high, close, volume) was obtained with the Yahoo Finance API via the [yfinance](https://pypi.org/project/yfinance/) Python package.

### Which machine learning model is used?
So far, I have explored using two model types. One is [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html), which is a gradient-boosting decision tree model, and the other is a neural network with one dense hidden layer and a dropout layer, implemented with the Keras Sequential API.

### Which evaluation metrics are used?
You can see my commentary on model performance in the Jupyter notebook files "decision_tree" and "neural_net". Also, in order to evaluate the models in a way that is more demonstrative, I develop a simple trading strategy in "strateg_simulation" and benchmark it against the performance of the S&P500 ETF fund, [SPY](https://www.google.com/finance/quote/SPY:NYSEARCA), across the given time period.
