'''
This file contains helpful functions.

There are two variable names you will see a lot. They are:
    - historicDat (dict): maps a ticker (str) to a DataFrame containing daily Open, Close, 
                            High, Low, and Volume, indexed by 'YYYY-MM-DD' date (str)

    - insiderDat (pd.DataFrame): each row represents an insider trade. Initially contains 
        - 'FilingDate', 
        - 'TradeDate', 
        - 'Ticker', 
        - 'CompanyName', 
        - 'InsiderName', 
        - 'Title' of the insider, 
        - 'TradeType' (purchase, sale, or sale + options exercise), 
        - 'Price' per share, 
        - 'Qty' of shares bought/sold in the trade, 
        - shares 'Owned' by the insider after the trade, 
        - 'DeltaOwn' of the insider's position in term of %, and 
        - total monetary 'Value' of the trade,
    although more features are added in the "Feature Creation" section below.
'''


############################################################################
############################# Imports ######################################
############################################################################
import re
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



############################################################################
############################ Miscellaneous #################################
############################################################################
class my_misc:
    
    def save_obj(obj, name):
        '''
        Save data as a pickle object.
        '''
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        '''
        Load pickled data.
        '''
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def validate(date_text):
        '''
        Ensure that all dates have a valid format.
        '''
        try:
            dt.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        
        

############################################################################
#####3########### Cleaning and Formatting CSV Data #########################
############################################################################
import yfinance as yf
from pandas_datareader import data as pdr
from time import sleep

yf.pdr_override()  # supposedly faster for getting stock data

class my_cleaning:
    
    def cleanAndFormatDF(csv_loc, clean_csv_loc, historicDat_loc, newORload='load', startDate=None, endDate=None):
        '''
        Produces a DataFrame of insider trades and a dictionary of historic ticker data.

        IN:
            csv_loc (str): location of CSV with insider data, without '.csv'
            clean_csv_loc (str): location to save cleaned insider data to CSV, without '.csv'
            historicDat_loc (str): location containing historic ticker data, without '.pkl'
            newORload (str): must be either 'new' (pulls new historic ticker data with yfinance and saves to
                                historicDat_loc.pkl) or 'load' (loads historic ticker data from historicDat.pkl)
            startDate (str): 'YYYY-MM-DD' indicating when to start pulling historic ticker data
            endDate (str): 'YYYY-MM-DD' indicating when to stop pulling historic ticker data

        OUT:
            insiderDat (pd.DataFrame): see top of file
            historicDat (dict): see top of file
        '''

        insiderDat = pd.read_csv(csv_loc + '.csv')
        insiderDat = insiderDat.rename(columns={'Filing Date': 'FilingDate', 
                                                'Trade Date': 'TradeDate', 
                                                'Company Name': 'CompanyName', 
                                                'Insider Name': 'InsiderName', 
                                                'Trade Type': 'TradeType'},
                                       errors='raise')

        '''
        NOTE: I am stripping away the filing date's time. Assume we can't take advantage of a filing's
        information and buy until opening next day.
        '''
        insiderDat.FilingDate = [dt.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S').date() 
                                 for d in insiderDat.FilingDate]  # convert to date object
        insiderDat.TradeDate = [dt.datetime.strptime(str(d), '%Y-%m-%d').date()
                                for d in insiderDat.TradeDate]  # convert to date object
        insiderDat.Price = [float(p.replace(',','').replace('$','')) 
                            for p in insiderDat.Price]
        insiderDat.Qty = [float(q[1]) if '#' in q[0] else float(q[0].replace(',','')) 
                          for q in zip(insiderDat.Qty, insiderDat.Qty2)]  # fixes format errors in the 'Qty' column
                                                                           # of the CSV
        insiderDat.Owned = [float(o.replace(',','')) 
                            for o in insiderDat.Owned]
        insiderDat.Value = [float(v[1]) if '#' in v[0] else float(v[0].replace(',','').replace('$','')) 
                            for v in zip(insiderDat.Value, insiderDat.Value2)] # fixes format errors in the 'Value' 
                                                                                # column of the CSV

        # replace infinite and too-large DeltaOwn with a sufficiently large number
        insiderDat = insiderDat.replace({'DeltaOwn': {'New': '0', '>999%': '0'}})
        insiderDat.DeltaOwn = [float(d.replace('%','')) 
                               for d in insiderDat.DeltaOwn]
        insiderDat = insiderDat.replace({'DeltaOwn': {'0': 10*max(insiderDat.DeltaOwn)}})

        insiderDat = insiderDat.replace({'Ticker': {'FB': 'META'}})  # address a major name change

        insiderDat = insiderDat.sort_values(by='FilingDate')



        allTickers = insiderDat.Ticker.unique().tolist()
        print('There are ' + str(len(allTickers)) + ' unique tickers.')

        if newORload == 'new':
            historicDat = my_retrieval.getHistoricDat(allTickers, startDate, endDate)
            my_misc.save_obj(historicDat, historicDat_loc)
        elif newORload == 'load':
            historicDat = my_misc.load_obj(historicDat_loc)
            historicDat.update(
                my_retrieval.getHistoricDat(allTickers, startDate, endDate, historicDat_loc, historicDat=historicDat)
            )
            my_misc.save_obj(historicDat, historicDat_loc)
        else:
            raise ValueError('newORload must be ''new'' or ''load''')


        exampleTick = next(iter(historicDat))
        print(f'\nExample ticker data for {exampleTick}:')
        print(historicDat[exampleTick])


        '''
        Now we want to remove trades for tickers that no longer exist from insiderDat.
        historicDat contains empty DataFrames for these tickers.

        For convenience, we also want to remove trades for tickers that were listed after our start date,
        indicated either by the first listed date being after startDate or by the presence of NaNs.
        '''
        tickersToRemove = set()
        for tick in allTickers:
            if ( historicDat[tick].empty
                or (historicDat[tick].index[0] > dt.datetime.strptime(startDate, '%Y-%m-%d'))
                or np.any(np.isnan(historicDat[tick])) ):
                tickersToRemove.add(tick)

        print('\nThere are ' + str(len(tickersToRemove)) + ' tickers that no longer exist or were listed on ' +
              f'an exchange after {startDate} and are being removed.')

        insiderDat = insiderDat[insiderDat.Ticker.isin(tickersToRemove)==False]
        insiderDat = insiderDat.drop(['Qty2', 'Value2'], axis=1).reset_index().drop(['index'], axis=1)
        insiderDat.to_csv(clean_csv_loc + '.csv', index=False)

        return insiderDat, historicDat



############################################################################
################ Retrieving and Generating Data ############################
############################################################################
class my_retrieval:

    def getHistoricDat(ticks, startDate, endDate, historicDat_loc, historicDat={}):
        '''
        Download stock data from the Yahoo Finance API between two dates.

        IN:
            ticks (List[str]): list of ticker names
            startDate (str): YYYY-MM-DD on which to begin pulling data
            endDate (str): YYYY-MM-DD on which to stop pulling data
            historicDat_loc (str): location containing historic ticker data, without '.pkl'
            historicDat (dict): see top of file

        OUT:
            historicDat (dict): input updated with new tickers
        '''
        my_misc.validate(startDate)
        my_misc.validate(endDate)

        newTicks = [t for t in ticks if t not in historicDat.keys()]
        if len(newTicks) == 0: return historicDat  # nothing to download!
        
        print(f'{len(newTicks)} tickers to download.')
        
        waitBeforeSleep = 100  # sleep every 100 requests; server is finnicky
        
        numBatches = len(newTicks)//waitBeforeSleep + 1

        for batch in range(numBatches):
            batchStartIdx = batch*waitBeforeSleep
            batchEndIdx = min(len(newTicks), batchStartIdx+waitBeforeSleep)

            batchTicks = newTicks[batchStartIdx:batchEndIdx]

            tickDat = pdr.get_data_yahoo(
                batchTicks,
                start=startDate, 
                end=endDate,
                threads=2,
                progress=True, 
                show_errors=False,
                group_by='ticker',
                timeout=10
            )

            for t in batchTicks: 
                try: historicDat.update({t: tickDat[t]})
                except KeyError: 
                    if len(batchTicks) == 1: historicDat.update({t: tickDat})
                    else: historicDat.update({t: pd.DataFrame()})

            my_misc.save_obj(historicDat, historicDat_loc)

            if batch < numBatches-1: sleep(30)


        return historicDat


    def returnDataOnDate(tick, tickDat, startDate, delta=0, dataName='Close', searchDirection=1):
        '''
        Returns a ticker's value associated with 'dataName', 'delta' days after 'startDate'. In the event that the 
        stock market was not open on this future date, function returns the nearest viable date in the 
        direction of 'searchDirection', looking forward or backward one day at a time.

        IN:
            tick (str): a ticker
            tickDat (pd.DataFrame): data from historicDat[tick] (see top of file)
            startDate (str): YYYY-MM-DD
            delta (int): days to look forward (nonnegative)
            dataName (str): 'Open', 'Close', 'High', 'Low', Volume'
            searchDirection (int): either +1 or -1
        OUT:
            val (float): the value of 'dataName' for 'tick' on 'futureDate'
            futureDate (dt.date object): the date for which we are returning 'val'
        '''

        futureDate = dt.date.strftime(
            dt.datetime.strptime(startDate, '%Y-%m-%d') + dt.timedelta(days=delta), '%Y-%m-%d'
        )  # first attempt at a future date

        e = 'KeyError'
        while e is not None:  # continually add 'searchDirection' to 'futureDate' until we can return a value
            try:
                val = tickDat.loc[futureDate][dataName]
                e = None

            except KeyError:
                futureDate = dt.date.strftime(
                    dt.datetime.strptime(futureDate, '%Y-%m-%d') + dt.timedelta(days=searchDirection), '%Y-%m-%d'
                )

                # flag if we start searching infinitely backwards
                if dt.datetime.strptime(futureDate, '%Y-%m-%d') < dt.datetime.strptime('2000-01-01', '%Y-%m-%d'):
                    raise ValueError('Out-of-bounds date caused by ' + tick + ' on ' + startDate)

        return val, dt.datetime.strptime(futureDate, '%Y-%m-%d').date()


    def returnPriceDiff(insiderDat, historicDat, SP500Dat, delta, priceTime):
        '''
        Returns difference between price at 'priceTime' on the initial filing date and price at 'priceTime',
        'delta' days later for each trade in 'insiderDat'.

        IN:
            insiderDat (pd.DataFrame): see top of file
            historicDat (dict): see top of file
            SP500Dat (dict): same format as historicDat, but for SPY data
            delta (int): nonnegative number of days
            priceTime (str): 'Open', 'Close'
        OUT:
            closingDiff (dict): maps a trade (str: a nonnegative integer appended to a ticker name) to a
                                percentage price difference tuple for ticker and SPY (float, float)
        '''

        closingDiff = {}

        for tradeNum, trade in insiderDat.iterrows():
            tick = trade['Ticker']
            tickDat = historicDat[tick]
            SPY_Dat = SP500Dat['SPY']
            startDate = str(trade['FilingDate'])

            startPrice, _ = my_retrieval.returnDataOnDate(
                tick, tickDat, startDate, dataName=priceTime, searchDirection=-1
            )
            futurePrice, _ = my_retrieval.returnDataOnDate(
                tick, tickDat, startDate, dataName=priceTime, delta=delta
            )
            startPrice_SP500, _ = my_retrieval.returnDataOnDate(
                'SPY', SPY_Dat, startDate, dataName=priceTime, searchDirection=-1
            )
            futurePrice_SP500, _ = my_retrieval.returnDataOnDate(
                'SPY', SPY_Dat, startDate, dataName=priceTime, delta=delta
            )

            closingDiff[tick + str(tradeNum)] = (100*(futurePrice - startPrice) / startPrice, 
                                                 100*(futurePrice_SP500 - startPrice_SP500) / startPrice_SP500)

        return closingDiff


    def returnVolatilities(tickDat, refDate, priceTime, daysToLookBack):
        '''
        Returns volume and price volatility over the past daysToLookBack days.

        IN:
            tickDat (pd.DataFrame): data from historicDat[tick] (see top of file)
            refDate (str): YYYY-MM-DD, the reference date
            priceTime (str): 'Open', 'Close', 'High', 'Low'
            daysToLookBack (int): how many past days to consider for volatility; nonnegative
        OUT:
            volumeVolatility (float)
            priceVolatility (float)
        '''    
        maxPastDate = refDate - dt.timedelta(days=daysToLookBack)

        pastTickVols = tickDat[maxPastDate:refDate]['Volume']
        pastTickPrices = tickDat[maxPastDate:refDate][priceTime]

        avgVol = np.mean(pastTickVols)
        avgPrice = np.mean(pastTickPrices)

        if avgVol == 0: volumeVolatility = 0
        else: volumeVolatility = np.std(pastTickVols) / avgVol

        if avgPrice == 0: priceVolatility = 0
        else: priceVolatility = np.std(pastTickPrices) / avgPrice

        return volumeVolatility, priceVolatility


    def returnBestPriceChange(tick, tickDat, refDate, priceTime, windowLen, daysToLookForward):
        '''
        Returns best median % price change across 'windowLen' days, in minDaysToLookForward to maxDaysToLookForward days.

        IN:
            tick (str): a ticker
            tickDat (pd.DataFrame): data from historicDat[tick] (see top of file)
            refDate (str): YYYY-MM-DD, the reference date
            priceTime (str): 'Open', 'Close', 'High', 'Low'
            minDaysToLookForward (int): min days to look forward in computing max price change
            maxDaysToLookForward (int): max days to look forward in computing max price chang
        OUT:
            meanPercentChange (float)
        '''
        currentPrice, dateUsed = my_retrieval.returnDataOnDate(
            tick, tickDat, dt.date.isoformat(refDate), dataName=priceTime, searchDirection=-1
        )

        minFutureDate = dateUsed + dt.timedelta(days=1)
        maxFutureDate = dateUsed + dt.timedelta(days=daysToLookForward)

        priceChanges = tickDat[minFutureDate:maxFutureDate][priceTime]

        bestMedian = max(
            [np.median(priceChanges[idx:idx+dt.timedelta(days=windowLen)])
             for idx in priceChanges.index[:-windowLen]]
        )

        medianPercentChange = 100*(bestMedian-currentPrice) / currentPrice


        return medianPercentChange



############################################################################
########################### Generating Plots ###############################
############################################################################
from random import random

class my_plots:

    def plotPriceDifference(diffDat, delta, labelThresh, ax):
        '''
        Makes a scatter plot of percentage price change 'delta' days later for each insider trade represented 
        in 'diffDat'.

        IN:
            diffDat (dict): maps a trade (represented by a string: ticker + 'index' in insiderDat) to a tuple
                            of price increases for (ticker, SPY)
            delta (int): nonnegative
            labelThresh (float): absolute value price change threshold above which plot points are labeled
            ax (pyplot axes object)
        OUT:
            ax (pyplot axes object)
        '''

        tickPrices = [val[0] for val in diffDat.values()]
        SP500Prices = [val[1] for val in diffDat.values()]

        ax.plot(diffDat.keys(), tickPrices, '.b', markersize=8)
        ax.plot(diffDat.keys(), SP500Prices, '--r', label='S&P500')
        ax.set_xticklabels([])

        for key in diffDat.keys():
            if (abs(diffDat[key][0]) > labelThresh) and random() < 0.25:  # label outlying points with 1/3 probability
                ax.annotate(re.sub(r'\d+', '', key), (key, diffDat[key][0]))

        ax.set_xticks([])
        ax.set_xlabel(f'({len(diffDat)} Trades)')
        ax.set_ylabel(f'% change after {delta} days')
        ax.set_title(f'Price % change after {delta} days')
        ax.legend()

        #mplcursors.cursor(multiple = True).connect(
        #    "add", lambda sel: sel.annotation.set_text(diffDat.keys()[sel.target.index]))

        return ax



    def plotOutlyingPriceDifference(outlierClosings, numDays, delta, ax):
        '''
        Makes a line plot of percentage price change for each ticker whose price 'delta' days after a trade
        is outlying.

        IN:
            outlierClosings (dict): maps 'pos' or 'neg' (representing the direction of outlying price change) to
                                    a numpy array, where each row contains daily ticker price for a given trade
            numDays (int): number of future days to plot
            delta (int): number of days after which price is outlying; nonnegative
            ax (pyplot axes object)
        OUT:
            ax (pyplot axes object)
        '''

        for row in outlierClosings['pos']:
            ax.plot(list(range(numDays)), row, '-b', markersize=6)
        for row in outlierClosings['neg']:
            ax.plot(list(range(numDays)), row, '-r', markersize=6)

        ymin, ymax = ax.get_ylim()

        ax.set_ylim(top=min(ymax, 400))
        ax.set_xlabel(f'Days after insider trade')
        ax.set_ylabel(f'Price % change')
        ax.set_title(f'Outlying {delta}-day ticker prices')

        return ax


    def plotVolatilityPriceScatter(volatilities, priceChanges, daysToLookForward, daysToLookBack):
        '''
        Creates scatter plots of 'avg % price change in minDaysToLookForward to maxDaysToLookForward days' vs 
        'volume/price volatility over the past daysToLookBack days'.

        IN:
            volatilities (List[(float, float)])
            priceChanges (list)
            minDaysToLookForward (int)
            maxDaysToLookForward (int)
            daysToLookBack (int)
        '''

        fig, axs = plt.subplots(1, 2, figsize=(6.4*2, 4.8))
        for i, name in enumerate(['Volume', 'Price']):
            axs[i].plot([vol[i] for vol in volatilities], priceChanges, '.b', markersize=8)

            axs[i].set_ylim(top=500)
            axs[i].set_xlabel(f'{name} volatility in previous {daysToLookBack} days')
            axs[i].set_ylabel(f'Best median % price change in {daysToLookForward} days')
            axs[i].set_title(f'Price Change vs {name} Volatility')

        plt.show()


    def plotPriceWithTrades(tick, tickDat, insiderDat, startDate, endDate):
        '''
        Plots a ticker's price history with insider trades overlaid.

        IN:
            tick (str): a ticker
            tickDat (pd.DataFrame): data from historicDat[tick] (see top of file)
            insiderDat (pd.DataFrame): see top of file
            startDate (str): YYYY-MM-DD representing when to start plotting
            endDate (str): YYYY-MM-DD representing when to stop plotting     
        '''

        groups = insiderDat.groupby('TradeType')

        # store only the ticker data needed for plotting
        plotDat = pd.DataFrame(columns=tickDat.columns, index=tickDat.index).astype(float)
        for d in pd.date_range(start=startDate, end=endDate):
            try: plotDat.loc[d] = tickDat.loc[d]
            except KeyError: pass

        # average daily price
        fin_av = [(plotDat.High[i] + plotDat.Low[i])/2 for i in range(len(plotDat))]

        fig, ax = plt.subplots(1, 1)
        ax.plot(plotDat.index, fin_av)
        ax.fill_between(plotDat.index, plotDat.Low, plotDat.High, color='b', alpha=.1)

        cmap = {'P - Purchase': 'g', 'S - Sale': 'r', 'S - Sale+OE': 'y'}
        for name, group in groups:
            ax.plot(group.FilingDate, group.Price, marker='o', linestyle='', label=name+', filed', color=cmap[name])

        ax.set_xlabel('Price, $')
        ax.legend()

        plt.xticks(rotation=75)
        plt.title(tick + ' price in June')


    def plotPredictedVsActual(train_pred, train_true, cv_pred, cv_true):
        '''
        Makes a scatter plot of model-predicted price change vs actual price change.  
        '''
        ls = [[train_pred, train_true], [cv_pred, cv_true]]
        
        fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 4.8))
        for i in [0,1]:
            y_pred = ls[i][0]
            y_true=ls[i][1]
            
            axs[i].scatter(y_pred, y_true)
            axs[i].plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], '--r', label='Predicted = Actual')

            ymin, ymax = axs[i].get_ylim()

            axs[i].set_ylim(top=min(ymax, 500))
            axs[i].set_xlabel('Predicted % price increase')
            axs[i].set_ylabel('Actual % price increase')
            axs[i].set_title('Predicted vs. Actual % Increase')
            axs[i].legend()
        
        plt.show()
    
    
    
############################################################################
########################## Feature Creation ################################
############################################################################
class my_features:
    
    def createAllFeatures(insiderDat, historicDat, daysToLookForward, windowLen, daysToLookBack, minOutput, maxOutput):
        '''
        Completes the feature engineering for insider trade data.

        IN:
            insiderDat (pd.DataFrame): see top of file
            historicDat (dict): see top of file
            minDaysToLookForward (int): min number of days to look ahead for forward-looking features
            maxDaysToLookForward (int): max number of days to look ahead for forward-looking features
            daysToLookBack (int): max number of days to look back for backward-looking features
            minOutput (float): change all lower price changes to this value
            maxOutput (float): change all higher price changes to this value

        OUT:
            insiderDat (pd.DataFrame): input modified to contain engineered features
        '''

        insiderDat['FilingDate'] = pd.to_datetime(insiderDat['FilingDate']).dt.date
        insiderDat['TradeDate'] = pd.to_datetime(insiderDat['TradeDate']).dt.date

        # create new features
        insiderDat[['NumTrades','TradeToFileTime','ValueOwned','VolumeVolatility',
                    'PriceVolatility','%FuturePriceChange']] = 0

        startDate = min(insiderDat.FilingDate)
        endDate = max(insiderDat.FilingDate)
        delta = endDate - startDate

        for tradeNum, trade in insiderDat.iterrows():
            print(f'Processing trade: {tradeNum}', end='\r')
            tick = trade['Ticker']
            tickDat = historicDat[tick]
            tradeDate = trade['TradeDate']
            fileDate = trade['FilingDate']

            # skip the first daysToLookBack days so we have data to look back at
            if (fileDate - dt.timedelta(days=daysToLookBack)) < startDate:
                continue


            # compute percentage change in shares owned by insider
            owned = insiderDat.at[tradeNum, 'Owned']
            shareChange = insiderDat.at[tradeNum, 'Qty']
            price = insiderDat.at[tradeNum, 'Price']
            if owned != shareChange:
                insiderDat.at[tradeNum, 'DeltaOwn'] = 100*shareChange / (owned-shareChange)


            # compute total value of insider's trade
            insiderDat.at[tradeNum, 'Value'] = shareChange*price


            # compute total value of insider's shares
            insiderDat.at[tradeNum, 'ValueOwned'] = owned*price


            # compute time gaps between trades and filings
            tradeToFileTime = (fileDate - tradeDate).days
            insiderDat.at[tradeNum, 'TradeToFileTime'] = tradeToFileTime


            # compute the number of same-ticker trades in the last daysToLookBack days, taking
            # advantage of supposed backend speedup with pd.eval
            prevDate = fileDate-dt.timedelta(days=daysToLookBack)
            recentTrades = pd.DataFrame()
            recentTrades = pd.eval('isRecentTrade = (insiderDat.Ticker==tick) and (insiderDat.FilingDate <= fileDate)' +
                          ' and (insiderDat.FilingDate >= prevDate)', target=recentTrades)

            insiderDat.at[tradeNum, 'NumTrades'] = len(recentTrades[recentTrades['isRecentTrade'] == True].index)


            # compute volatilities in the last daysToLookBack days
            # compute avg price percentage change in the next minDaysToLookForward to maxDaysToLookForward days
            volatilities = my_retrieval.returnVolatilities(
                tickDat, fileDate, 'Close', daysToLookBack
            )

            priceChange = my_retrieval.returnBestPriceChange(
                tick, tickDat, fileDate, 'Close', windowLen, daysToLookForward
            )

            insiderDat.at[tradeNum, 'VolumeVolatility'] = volatilities[0]
            insiderDat.at[tradeNum, 'PriceVolatility'] = volatilities[1]
            
            # cap the outputs in the interval [-minOutput, maxOutput), as discussed in exploratory_analysis
            insiderDat.at[tradeNum, '%FuturePriceChange'] = min(max(priceChange, minOutput), maxOutput-1e-6)
        
        print('Processing trade: Done!')
        return insiderDat



############################################################################
########################## Model Preparation ###############################
############################################################################
from imblearn.over_sampling import RandomOverSampler

class my_model_prep:
    
    def prepareForModel(insiderDat):
        '''
        Assigns all insider titles to categories and ensures that each feature is the proper dtype.
        '''
        def fixTitle(title):
            '''
            Assigns an insider title to a category. Splits people into people likely to have insider knowledge
            vs all others (e.g. 10% holders).
            '''

            importantKeywords = ['Dir', 'VP', 'Vice', 'V.P.', 'Pres',  # directors to the left; C-suite below
                                 'CEO', 'C.E.O' 'COO', 'C.O.O', 'CHRO', 'C.H.R.O', 'CFO', 'C.F.O', 'CTO', 'C.T.O', 'Chief',
                                 'COB', 'C.O.B.', 'Chair']  # chair keywords

            if any([re.search(key, title, re.IGNORECASE) for key in importantKeywords]): newTitle = 1
            else: newTitle = 0

            return newTitle
        
        def fixTradeType(tradeType):
            '''
            Assigns an insider title to a category. Splits people into people likely to have insider knowledge
            vs all others (e.g. 10% holders).
            '''

            importantKeywords = ['Dir', 'VP', 'Vice', 'V.P.', 'Pres',  # directors to the left; C-suite below
                                 'CEO', 'C.E.O' 'COO', 'C.O.O', 'CHRO', 'C.H.R.O', 'CFO', 'C.F.O', 'CTO', 'C.T.O', 'Chief',
                                 'COB', 'C.O.B.', 'Chair']  # chair keywords

            if tradeType == 'S - Sale': newType = 0
            elif tradeType == 'S - Sale+OE': newType = 1
            elif tradeType == 'P - Purchase': newType = 2
            else: raise ValueError('Trade type couldn''t be assigned')

            return newType
        
        #################################################################

        if 'Title' in insiderDat.columns:
            insiderDat['TitleCat'] = None
            insiderDat.TitleCat = [fixTitle(t) for t in insiderDat.Title]
            insiderDat = insiderDat.drop('Title', axis=1)
            
        if 'TradeType' in insiderDat.columns:
            insiderDat['TradeTypeCat'] = None
            insiderDat.TradeTypeCat = [fixTradeType(t) for t in insiderDat.TradeType]
            insiderDat = insiderDat.drop('TradeType', axis=1)

        insiderDat.FilingDate = pd.to_datetime(insiderDat['FilingDate']).dt.date
        
        insiderDat = insiderDat.astype({'Price': 'float', 
                                        'DeltaOwn': 'float', 
                                        'Value': 'float', 
                                        'NumTrades': 'int', 
                                        'TradeToFileTime': 'int', 
                                        'VolumeVolatility': 'float',
                                        'PriceVolatility': 'float',
                                        '%FuturePriceChange': 'float'})

        return insiderDat
    
    
    def oversample(data_XY, binStarts):
        '''
        Oversamples data_XY to account for the distribution of price changes. 
        IN:
            data_XY (pd.DataFrame)
            binStarts (List[float]): a list of values at which our 'bins' for oversampling begin
        OUT:
            data_XY_resampled (pd.DataFrame): same as data_XY, but with rows duplicated via oversampling
        '''
        data_XY.assign(PriceChangeCAT=None)
        
        binEnds = binStarts[1:] + [max(data_XY['%FuturePriceChange']) + 1e-6]  # account for rounding error
        
        for binStart, binEnd in zip(binStarts, binEnds):
            data_XY.loc[
                (data_XY['%FuturePriceChange'] >= binStart) & (data_XY['%FuturePriceChange'] < binEnd), 
                'PriceChangeCAT'
            ] = f'{round(binStart)} to {round(binEnd)}'
            
        data_XY_resampled, _ = RandomOverSampler().fit_resample(data_XY, data_XY.PriceChangeCAT)
        
        return data_XY_resampled.drop(columns=['PriceChangeCAT'])


    def returnXandY(insiderDat, startDate, endDate, binStarts=[]):
        '''
        Drops features not used for training and splits trade data into input and output sets.

        IN:
            insiderDat (pd.DataFrame): see top of file
            startDate (str): start date for data
            endDate (str): end date for data
            binStarts (List[int]): a list of values at which our 'bins' for oversampling begin.
                                    Empty list skips oversampling.
        OUT:
            data_XY (pd.DataFrame): contains input and output data together, along with 
                                    'FilingDate' and 'Ticker'
            data_X (pd.DataFrame): input data for model
            data_Y (pd.DataFrame): output data for model
        '''
        dateRange = pd.date_range(start=startDate, end=endDate).date

        insiderDat = insiderDat.drop(columns=['CompanyName', 'TradeDate', 'InsiderName', 'Owned', 'Qty'])

        # Get rid of a column that I created when re-indexing the dataframe
        if 'Unnamed: 0' in insiderDat.columns: insiderDat = insiderDat.drop(columns=['Unnamed: 0'])

        # Use one-hot encoding for trade type
        #dummies_data = pd.get_dummies(insiderDat, columns=['TradeType'], prefix=[None])

        data_XY = insiderDat[insiderDat['FilingDate'].isin(dateRange)]
        
        if binStarts: data_XY = my_model_prep.oversample(data_XY, binStarts)
        
        data_XY = data_XY.dropna()
        data_X = data_XY.drop(columns=['FilingDate', '%FuturePriceChange', 'Ticker'])
        data_Y = data_XY['%FuturePriceChange']

        assert np.any(np.isnan(data_X)) == False
        assert np.all(np.isfinite(data_X)) == True
        assert np.any(np.isnan(data_Y)) == False
        assert np.all(np.isfinite(data_Y)) == True

        return data_XY, data_X, data_Y
    

###########################################################################
############################# Models ######################################
###########################################################################
import xgboost as xgb
from operator import itemgetter
from sklearn.model_selection import RandomizedSearchCV

class my_models:
    
    def XGBReg(train_X, train_Y, cv_X, cv_Y, objective, metric, parameters, earlyStopping, n_estimators):
        '''
        Uses a randomized search to get the best parameters for an XGBoost model, and then fits the model with these
        parameters.
        
        IN:
            train_X, train_Y, cv_X, cv_Y (pd.DataFrame)
            objective (function): defined appropriately according to XGBoost docs
            metric (function): evaluation metric, also defined appropriately
            parameters (dict): maps parameter names to lists of values to try
            earlyStopping (int): number of rounds to wait for loss improvement
            n_estimators (int): number of estimators to be used by the model
        OUT:
            xgb_model (an Fit object)
        '''
        xgb_reg = xgb.XGBRegressor(
            verbosity=0, 
            objective=objective,
            eval_metric=metric,
            early_stopping_rounds=earlyStopping,
            n_estimators=n_estimators,
            tree_method='hist'
        )
        
        xgb_rscv = RandomizedSearchCV(
            xgb_reg,
            param_distributions=parameters,
            verbose=0,
            random_state=40,
            error_score='raise'
        )

        xgb_rscv_model = xgb_rscv.fit(
            train_X, 
            train_Y, 
            eval_set=[(train_X, train_Y), (cv_X, cv_Y)],
            verbose=0
        )

        print('Best learning_rate, min_split_loss, max_depth, max_delta_step, colsample_bytree, subsample,\n' +
              'reg_lambda, min_child_weight:\n' +
              f'''{itemgetter(
                'learning_rate', 'min_split_loss', 'max_depth', 'max_delta_step', 'colsample_bytree', 'subsample', 
                'reg_lambda', 'min_child_weight'
               )(xgb_rscv_model.best_estimator_.get_params())
              }''')
        
        xgb_model = xgb.XGBRegressor(**xgb_rscv_model.best_estimator_.get_params()).fit(
            train_X, train_Y, 
            eval_set=[(train_X, train_Y), (cv_X, cv_Y)], verbose=0
        )
        
        return xgb_model
        
        

###########################################################################
########################## Trade Simulation ###############################
###########################################################################
class my_sims:
    
    def runTradeSimulation(data_XY, predName, historicDat, startDate, endDate, buyThresh, sellThresh):
        '''
        Runs the basic investment simulation outlined in strategy_simulation.

        IN:
            data_XY (pd.DataFrame): contains input and output data together, along with 
                                    'FilingDate' and 'Ticker'
            predName (str): name of data_XY's prediction column, e.g. 'XGB_Prediction'
            historicDat (dict): see top of file
            startDate (str): date on which to begin simulation
            endDate (str): date on which to end simulation
            buyThresh (float): threshold for predicted % increase, above which we buy shares of the ticker
            sellThresh (float): threshold for realized % increase, above which we sell shares of the ticker
        '''

        myTrades = {}  # logs purchases and sales in the simulation
        totalInvested = 0
        totalProfit = 0

        for d in pd.date_range(start=startDate, end=endDate):    
            currDate = dt.date.strftime(d.date(), '%Y-%m-%d')

            # Check each trade's performance prediction. If high enough, purchase at next day's opening.
            for tradeNum, trade in data_XY[data_XY['FilingDate'] == d.date()].iterrows():
                if trade[predName] < buyThresh: continue

                tick = trade['Ticker']

                buyPrice, buyDate = my_retrieval.returnDataOnDate(
                    tick, historicDat[tick], currDate, delta=1, dataName='Open'
                )
                #buyDate = dt.date.strftime(buyDate, '%Y-%m-%d')

                totalInvested += 1

                print(f'''Buying {tick} on {buyDate}, currently ${round(buyPrice, 2)}''')

                if tick in myTrades.keys():
                    myTrades[tick]['BuyPrice'].append(buyPrice)
                    myTrades[tick]['SellPrice'].append(None)
                else:
                    myTrades[tick] = {'BuyPrice': [buyPrice], 'SellPrice': [None]}


            # Check already-purchased stocks. If any ticker's value has risen enough, sell all shares  at closing.
            for tick, elem in myTrades.items():
                for buyNum, buyPrice in enumerate(elem['BuyPrice']):
                    try:
                        currPrice = historicDat[tick].loc[currDate]['Close']

                        # sell if the price has risen enough and the shares haven't been sold yet
                        if (currPrice > (1. + sellThresh/100)*buyPrice) and (elem['SellPrice'][buyNum] is None):
                            elem['SellPrice'][buyNum] = currPrice
                            profit = (currPrice-buyPrice) / buyPrice
                            totalProfit += profit
                            print(f'Selling {tick} on {currDate}, currently ${round(currPrice, 2)}, ' +
                                         f'for {round(100*profit, 2)}% profit')

                    except KeyError: pass  # we can't sell today

                # If it's the last day of the simulation, sell everything (for performance evaluation)
                # !!!Make sure that this is a day that the market is open!!!
                if d.date() == dt.datetime.strptime(endDate, '%Y-%m-%d').date():
                    try: currPrice = historicDat[tick].loc[currDate]['Close']
                    except: raise ValueError('Market must be open on simulation end date!')
                        
                    totalProfit += sum([(currPrice-elem['BuyPrice'][idx])/elem['BuyPrice'][idx] 
                                        for idx, val in enumerate(elem['SellPrice']) if val is None])
                    myTrades[tick]['SellPrice'] = [currPrice if val is None else val for val in elem['SellPrice']]


        print('\n-----------------------------------------\n')

        '''Determine total profit in the given time period.'''
        print(f'We invested ${totalInvested}. Our portfolio is now worth ${round(totalInvested+totalProfit, 2)}, ' +
              f'giving a return of {round(100*totalProfit/totalInvested, 2)}%.')