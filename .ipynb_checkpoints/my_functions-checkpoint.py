'''
This file contains helpful functions.

There are two variable names you will see a lot in this notebook. They are:
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
import yfinance as yf
import datetime as dt
import pickle
import matplotlib.pyplot as plt

from random import random

plt.style.use('fivethirtyeight')



############################################################################
############################ Miscellaneous #################################
############################################################################
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
    print('There are ' + str(len(allTickers)) + ' unique tickers.\nGetting historic data for these tickers...')
    
    if newORload == 'new':
        historicDat = getHistoricDat(allTickers, startDate, endDate)
        save_obj(historicDat, historicDat_loc)
    elif newORload == 'load':
        historicDat = load_obj(historicDat_loc)
    else:
        raise ValueError('newORload must be ''new'' or ''load''')
        
        
    exampleTick = next(iter(historicDat))
    print(f'\nExample ticker data for {exampleTick}:')
    print(historicDat[exampleTick])
    

    '''
    Now we want to remove trades for tickers that no longer exist from insiderDat.
    historicDat contains empty DataFrames for these tickers.
    
    For convenience, we also want to remove trades for tickers that were listed after our start date.
    '''
    tickersToRemove = set()
    for tick in allTickers:
        if historicDat[tick].empty or (historicDat[tick].index[0] > dt.datetime.strptime(startDate, '%Y-%m-%d')):
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
def getHistoricDat(ticks, startDate, endDate):
    '''
    Download stock data from the Yahoo Finance API between two dates.
    
    IN:
        ticks (List[str]): list of ticker names
        startDate (str): YYYY-MM-DD on which to begin pulling data
        endDate (str): YYYY-MM-DD on which to stop pulling data
        
    OUT:
        historicDat (dict): see top of file
    '''
    validate(startDate)
    validate(endDate)
    
    historicDat = {}
    
    for t in ticks:
        stockDat = yf.download(t, start=startDate, end=endDate, progress=False)
        historicDat[t] = stockDat
        print(str(len(historicDat))+'/'+str(len(ticks)) + ' done', end='\r')  # print progress
        
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
    
    futureDate = dt.date.strftime(dt.datetime.strptime(startDate, '%Y-%m-%d') 
                                  + dt.timedelta(days=delta), '%Y-%m-%d')  # first attempt at a future date
    
    e = 'KeyError'
    while e is not None:  # continually add 'searchDirection' to 'futureDate' until we can return a value
        try:
            val = tickDat.loc[futureDate][dataName]
            e = None

        except KeyError:
            futureDate = dt.date.strftime(dt.datetime.strptime(futureDate, '%Y-%m-%d') 
                                               + dt.timedelta(days=searchDirection), '%Y-%m-%d')
            
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
        
        startPrice, _ = returnDataOnDate(tick, tickDat, startDate, dataName=priceTime, searchDirection=-1)
        futurePrice, _ = returnDataOnDate(tick, tickDat, startDate, dataName=priceTime, delta=delta)
        startPrice_SP500, _ = returnDataOnDate('SPY', SPY_Dat, startDate, dataName=priceTime, searchDirection=-1)
        futurePrice_SP500, _ = returnDataOnDate('SPY', SPY_Dat, startDate, dataName=priceTime, delta=delta)
        
        closingDiff[tick + str(tradeNum)] = (100*(futurePrice - startPrice) / startPrice, 
                                             100*(futurePrice_SP500 - startPrice_SP500) / startPrice_SP500)

    return closingDiff


def returnVolumeAndPriceChange(tick, tickDat, refDate, priceTime, daysToLookForward, daysToLookBack):
    '''
    Returns percentage volume change for 'tick' over the last 'daysToLookBack' days, and the max percentage
    price increase over the next 'daysToLookForward' days.
    
    IN:
        tick (str): a ticker
        tickDat (pd.DataFrame): data from historicDat[tick] (see top of file)
        refDate (str): YYYY-MM-DD, the reference date
        priceTime (str): 'Open', 'Close', 'High', 'Low'
        daysToLookForward (int): how many future days to consider for max price change; nonnegative
        daysToLookBack (int): how many past days to consider for volume change; nonnegative
    '''
    
    # 'dateUsed' is the actual reference date used for time-of-filing data
    currentVol, dateUsed = returnDataOnDate(tick, 
                                            tickDat, 
                                            dt.date.isoformat(refDate), 
                                            dataName='Volume', 
                                            searchDirection=-1)
    
    previousVol, _ = returnDataOnDate(tick,
                                      tickDat,  
                                      dt.date.isoformat(dateUsed-dt.timedelta(days=daysToLookBack)), 
                                      dataName='Volume', 
                                      searchDirection=-1)
    
    if previousVol == 0 and currentVol == 0: percentChangeVol = 0  
    elif previousVol == 0: percentChangeVol = 9999  # assign a large increase instead of infinity
    else: percentChangeVol = 100*(currentVol-previousVol) / previousVol
    
    currentPrice = tickDat.loc[dt.date.isoformat(dateUsed)][priceTime]
    
    '''
    Determine the max percentage price change, starting the day after 'dateUsed'.
    '''
    highestPrice = 0
    for i in range(1, daysToLookForward):
        tempPrice, _ = returnDataOnDate(tick, 
                                        tickDat, 
                                        dt.date.isoformat(dateUsed), 
                                        delta=i)
        if tempPrice > highestPrice:
            highestPrice = tempPrice
            
    percentChangePrice = 100*(highestPrice-currentPrice) / currentPrice
    
    return percentChangeVol, percentChangePrice



############################################################################
########################### Generating Plots ###############################
############################################################################
def createDifferencePlot(diffDat, delta, labelThresh, ax):
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
    
    
    
def createOutlyingDifferencePlots(outlierClosings, numDays, delta, ax):
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

    ax.set_xlabel(f'Days after insider trade')
    ax.set_ylabel(f'Price % change after {delta} days')
    ax.set_title(f'Outlying {delta}-day ticker prices')
    
    return ax
    
    
def createVolumePriceScatters(volPriceDat, daysToLookForward, daysToLookBack):
    '''
    Creates a scatter plot of 'max percentage price change in daysToLookForward days' vs 'percentage
    volume change in daysToLookBack days'.
    
    IN:
        volPriceDat (List[List[float]]): first index represents trade ID; second index represents 
                                            (volume change, price change)
        daysToLookForward (int)
        daysToLookBack (int)
    '''
    
    fig, ax = plt.subplots(1, 1)
    ax.plot([val[0] for val in volPriceDat], [val[1] for val in volPriceDat], '.b', markersize=8)
    
    plt.xlabel(f'% volume change in previous {daysToLookBack} days')
    plt.ylabel(f'Max % price change in {daysToLookForward} days')
    plt.xscale('symlog')
    plt.title(f'Price Change vs Volume Change')
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
    
    
    
############################################################################
########################## Feature Creation ################################
############################################################################
def createAllFeatures(insiderDat, historicDat, daysToLookForward=90, daysToLookBack=1):
    '''
    Completes the feature engineering for insider trade data.
    
    IN:
        insiderDat (pd.DataFrame): see top of file
        historicDat (dict): see top of file
        daysToLookForward (int): number of days to look ahead for forward-looking features
        daysToLookBack (int): number of days to look ahead for backward-looking features
        
    OUT:
        insiderDat (pd.DataFrame): input modified to contain engineered features
    '''
    
    insiderDat['FilingDate'] = pd.to_datetime(insiderDat['FilingDate']).dt.date
    insiderDat['TradeDate'] = pd.to_datetime(insiderDat['TradeDate']).dt.date
    
    # create new features
    insiderDat[['NumTrades','TradeToFileTime','ValueOwned','%VolumeChange','%FuturePriceChange']] = 0

    startDate = min(insiderDat.FilingDate)
    endDate = max(insiderDat.FilingDate)
    delta = endDate - startDate

    for tradeNum, trade in insiderDat.iterrows():
        print(f'Processing trade {tradeNum}', end='\r')
        tick = trade['Ticker']
        tickDat = historicDat[tick]
        tradeDate = trade['TradeDate']
        fileDate = trade['FilingDate']

        # skip the first DAYS_TO_LOOK_BACK days so we have data to look back at
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


        # compute and categorize time gaps between trades and filings
        tradeToFileTime = (fileDate - tradeDate).days
        insiderDat.at[tradeNum, 'TradeToFileTime'] = tradeToFileTime


        # compute and categorize the number of same-ticker trades in the last DAYS_TO_LOOK_BACK days
        recentTrades = insiderDat.apply(lambda x: True if (x['Ticker'] == tick) 
                                                    and (x['FilingDate'] <= fileDate)
                                                    and (x['FilingDate'] 
                                                         >= fileDate-dt.timedelta(days=daysToLookBack))
                                                    else False, axis=1)

        insiderDat.at[tradeNum, 'NumTrades'] = len(recentTrades[recentTrades == True].index)


        # compute and categorize the percentage volume change in the last DAYS_TO_LOOK_BACK days
        # compute the most best closing price percentage change in the next DAYS_TO_LOOK_FORWARD days
        percentChangeVol, percentChangePrice = returnVolumeAndPriceChange(tick, 
                                                                          tickDat, 
                                                                          fileDate,
                                                                          'Close',
                                                                          daysToLookForward, 
                                                                          daysToLookBack)
        
        insiderDat.at[tradeNum, '%VolumeChange'] = percentChangeVol
        insiderDat.at[tradeNum, '%FuturePriceChange'] = percentChangePrice
        
    return insiderDat



############################################################################
########################## Model Preparation ###############################
############################################################################
def prepareForModel(insiderDat):
    '''
    Assigns insider titles to categories and ensures that each feature is the proper dtype.
    '''
    
    def fixTitle(title):
        '''
        Assigns insider titles to categories.
        
        I figure that the Chair of the Board is the most fiscally powerful person in a company, so to break ties for
        people who hold multiple titles, we prioritize COB, then C-suite, then other directors, then anyone else.
        '''

        directorKeywords = ['Dir', 'VP', 'Vice', 'V.P.', 'Pres']
        officerKeywords = ['CEO', 'C.E.O' 'COO', 'C.O.O', 'CHRO', 'C.H.R.O', 
                           'CFO', 'C.F.O', 'CTO', 'C.T.O', 'Chief']
        chairKeywords = ['COB', 'C.O.B.', 'Chair']

        if any([re.search(s, title, re.IGNORECASE) for s in chairKeywords]):
            newTitle = 'Chair'
        elif any([re.search(s, title, re.IGNORECASE) for s in officerKeywords]):
            newTitle = 'Officer'
        elif any([re.search(s, title, re.IGNORECASE) for s in directorKeywords]):
            newTitle = 'Director'
        else:
            newTitle = 'Other'

        return newTitle
    
    if 'Title' in insiderDat.columns:
        insiderDat['TitleCat'] = None
        insiderDat.TitleCat = [fixTitle(t) for t in insiderDat.Title]
        insiderDat = insiderDat.drop('Title', axis=1)

    insiderDat.FilingDate = pd.to_datetime(insiderDat['FilingDate']).dt.date
    insiderDat = insiderDat.astype({'Price': 'float', 
                                    'Qty': 'float', 
                                    'Owned': 'float', 
                                    'DeltaOwn': 'float', 
                                    'Value': 'float', 
                                    'NumTrades': 'int', 
                                    'TradeToFileTime': 'int', 
                                    '%VolumeChange': 'float', 
                                    '%FuturePriceChange': 'float'})
    
    return insiderDat


def returnXandY(insiderDat, startDate, endDate):
    '''
    Drops features not used for training and splits trade data into input and output sets.
    
    IN:
        insiderDat (pd.DataFrame): see top of file
        startDate (str): start date for data
        endDate (str): end date for data
    OUT:
        data_XY (pd.DataFrame): contains input and output data together, along with 
                                'FilingDate' and 'Ticker'
        data_X (pd.DataFrame): input data for model
        data_Y (pd.DataFrame): output data for model
    '''
    dateRange = pd.date_range(start=startDate, end=endDate).date

    insiderDat = insiderDat.drop(columns=['CompanyName', 'TradeDate', 'InsiderName'])
    
    # Get rid of a column that seems to be created when data is re-indexed
    if 'Unnamed: 0' in insiderDat.columns: insiderDat = insiderDat.drop(columns=['Unnamed: 0'])
    
    # Use one-hot encoding for insider title and trade type
    dummies_data = pd.get_dummies(insiderDat, columns=['TitleCat', 'TradeType'], prefix=['Title', None])

    data_XY = dummies_data[dummies_data['FilingDate'].isin(dateRange)]
    data_XY = data_XY.dropna()
    data_X = data_XY.drop(columns=['FilingDate', '%FuturePriceChange', 'Ticker'])
    data_Y = data_XY['%FuturePriceChange']

    assert np.any(np.isnan(data_X)) == False
    assert np.all(np.isfinite(data_X)) == True
    assert np.any(np.isnan(data_Y)) == False
    assert np.all(np.isfinite(data_Y)) == True
    
    return data_XY, data_X, data_Y



###########################################################################
########################## Trade Simulation ###############################
###########################################################################
def runTradeSimulation(data_XY, historicDat, startDate, endDate, buyThresh, sellThresh):
    '''
    Runs the basic investment simulation outlined in strategy_simulation.
    
    IN:
        data_XY (pd.DataFrame): contains input and output data together, along with 
                                'FilingDate' and 'Ticker'
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
            if trade['Prediction'] < buyThresh: continue

            tick = trade['Ticker']

            buyPrice, buyDate = returnDataOnDate(tick, historicDat[tick], currDate, delta=1, dataName='Open')
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
                currPrice = historicDat[tick].loc[currDate]['Close']
                totalProfit += sum([(currPrice-elem['BuyPrice'][idx])/elem['BuyPrice'][idx] 
                                    for idx, val in enumerate(elem['SellPrice']) if val is None])
                myTrades[tick]['SellPrice'] = [currPrice if val is None else val for val in elem['SellPrice']]


    print('\n-----------------------------------------\n')

    '''Determine total profit in the given time period.'''
    print(f'We invested ${totalInvested}. Our portfolio is now worth ${round(totalInvested+totalProfit, 2)}, ' +
          f'giving a return of {round(100*totalProfit/totalInvested, 2)}%.')
