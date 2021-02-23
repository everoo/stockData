import pandas as pd
import os
from datetime import datetime, timedelta
import itertools
from collections import Counter
from functools import reduce

def str_var(s):
    if s == '-':
        return None
    try:
        return float(s[:-1])*mil[s[-1]]
    except:
        return s

def getFiles(closing=True):
    files = sorted([f for f in os.listdir() if len(f) == 16])
    if closing:
        files = [f for f in files if file_dt(f).hour > 13]
    return files

def getAboveAverageSectors(df):
    avgs = df.mean()
    sectAvgs = {s: df[df['Sector'] == s].mean() for s in df['Sector'].drop_duplicates()}
    return {avg: [s for s in sectAvgs if sectAvgs[s][avg] > avgs[avg]] for avg in avgs.index}

def getStocksWithBest(stat, n, df):
    reduce_df = lambda d: d[d >= d.mean()]
    ndf = reduce_df(df[stat])
    while len(ndf) > n:
        ndf = reduce_df(ndf)
    return ndf.keys()#[df.loc[i]['Ticker'] for i in ndf.index]

mil = {'%': 1, 'M': 1e6, 'B': 1e9}
red = lambda x, y: x.combine(y, lambda l, r: l+[r] if isinstance(l, list) else [l, r])
find_duplicates = lambda l: [t for t, c in Counter(itertools.chain.from_iterable(l)).items() if c > 1]
file_dt = lambda f: datetime.strptime(f, '%Y%m%d%H%M.csv')
dt_file = lambda dt: dt.strftime('%Y%m%d%H%M.csv')
scoreSectors = lambda df: Counter(itertools.chain.from_iterable(getAboveAverageSectors(df).values()))
bestStocks = lambda df: Counter(itertools.chain.from_iterable([getStocksWithBest(stat, 20, df) for stat in df.mean().index]))
bestStockOfBestSector = lambda df: bestStocks(df[df['Sector'] == scoreSectors(df).most_common()[0][0]])

def findTrends(dfs, stat, cut, filters):
    list_str = lambda l: ''.join(['G' if l[n]*(1+cut) < l[n+1] else 'L' if l[n+1] < l[n]*(1-cut) else '-' for n in range(len(l)-1)])
    return [t for t, v in reduce(red, [df[stat] for df in dfs]).apply(list_str).items() if filters(v)]

def getGoodPriVolStocks(dfs):
    vols = findTrends(dfs, 'Volume', 0.1, lambda v: 1<v.count('L')<3 and 'G' in v[-2:])
    pris = findTrends(dfs, 'Price', 0.03, lambda v: v.count('L')>=3 and 'G' not in v[-2:])
    return find_duplicates([vols, pris])

files = getFiles()
dfs = [pd.read_csv(f).set_index('Ticker').applymap(str_var) for f in files]

def findTrendTrends(dfs):
    o_values = {'Volume':[0.1, lambda v: 1<v.count('L')<3 and v[-3:].count('G')>=1],
              'Price':[0.03, lambda v: v.count('L')>=3 and 'G' not in v[-2:]]}
    
    values = {'Volume':[0.25, lambda v: v[:-2].count('L')>=2 and v[-3:].count('G')>=1],
              'Price':[0.03, lambda v: v[:-2].count('L')>=1 and v[-3:].count('G')>=2]}
    return find_duplicates([findTrends(dfs, k, v[0], v[1]) for k, v in values.items()])

tot = 1.0
for n in range(len(dfs)-6):
    print(file_dt(files[n+5]))
    total = 1.0
    for ticker in findTrendTrends(dfs[n:n+5]):
        change = dfs[n+6].at[ticker, 'Change']
        total*=1+change/100
        tot*=1+change/100
        print(ticker, change)
    print('TOTAL', total)
print('Omega Total', tot)
