# "Insider Training": A Novel Approach to Stock-Trading
According to [Investopedia](https://www.investopedia.com/terms/i/insider.asp), an **insider** is "anyone who trades a company's shares based on material nonpublic knowledge."

In this repository's notebooks, we mine public insider trade data and use machine learning to generate a novel trading strategy that outperforms the S&P 500 during a two-week period in October 2022.\*

## Where can the data be found?
Insider trade filings are published online by the SEC, and they can be found using the EDGAR API. However, it is much easier to pull data from sites that gather the filing data into tabular form such as [Benzinga](https://www.benzinga.com/).

Historic ticker data (e.g. daily opening, low, high, close, volume) was obtained with the Yahoo Finance API via the [yfinance](https://pypi.org/project/yfinance/) Python package.

## Which machine learning model is used?
I use [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html), which is a gradient-boosting decision tree model, and a Keras Sequential neural network with one dense hidden layer.

## Which evaluation metrics are used?
You can see my commentary on model performance in *nb3_decision_tree.ipynb* and *nb4_neural_net.ipynb*. I create a custom-weighted mean-squared-error metric that penalizes overestimated valuation losses and underestimated large valuation gains more harshly.

## What strategy is used?
I develop a simple trading strategy in *nb5_strategy_simulation.ipynb* based on my model's valuation-percentage-increase predictions. I benchmark it against the performance of the S&P 500 ETF fund, [SPY](https://www.google.com/finance/quote/SPY:NYSEARCA), across the given time period for testing.  
<br />
<br />
*\*This project is neither an investment pitch nor financial advice.*
