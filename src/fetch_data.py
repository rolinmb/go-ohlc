from av_config import AV_KEY
from sys import argv, exit
import time
import datetime as dt
import pandas as pd
import urllib.request, json

def buildSeriesDataFrame(ticker, data):
    # print(f'JSON Snapshot before building DataFrame:\n{data}\n')
    df = pd.DataFrame(columns=['Ticker','Date','Open','High','Low','Close', 'Volume'])
    for k,v in data.items():
        date = dt.datetime.strptime(k, '%Y-%m-%d')
        df.loc[-1,:] = [ticker, date.date(),
            float(v['1. open']), float(v['2. high']),
            float(v['3. low']), float(v['4. close']),
            float(v['5. volume'])]
        df.index += 1
    df.sort_values('Date')
    return df[::-1] # Now Chronologically sorted: Past(start/top)->Present(end/bottom)

def fetchSeriesData(ticker, logging=True, ts_csv_out=None):
    if (ts_csv_out is None or ts_csv_out.strip() == '' or ts_csv_out.strip() == '.csv' or not ts_csv_out.endswith('.csv')):
        ts_csv_out = 'ohlc_data/%s_tseries.csv'%ticker
        print('fetchSeriesData():\t!! NOTICE: Invalid/no .csv filename entered for AlphaVantage Time Series data; using default option ''%s''!!))'%ts_csv_out)
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s'%(ticker, AV_KEY)
    with urllib.request.urlopen(url) as source:
        try: # Attempt to read JSON, if no JSON then maybe 404 error or other type of error
            data = json.loads(source.read().decode())
        except Exception as e:
            exit(f'fetchSeriesData(): ! ERROR {e}: Couldn''t fetch the AlphaVantage API URL => calling sys.exit()')
        try: # If JSON loaded correctly, isolate the Time Series OHLC data from the AlphaVantage API
            data = data['Time Series (Daily)']
        except KeyError:
            exit(f'fetchSeriesData():! KEY ERROR: {ticker} is not valid to query on AlphaVantage => calling sys.exit()')
        if logging:
            print(f'fetchSeriesData():* urllib.request.urlopen() successfully navigated to\n\t{url}')
        # Call helper function to process 'data' into DataFrame object and write to csv
        df = buildSeriesDataFrame(ticker, data)
        df.to_csv(ts_csv_out, index=False) 
        del(df)
        if logging:
            print('fetchSeriesData():\t(Successfully created/refreshed ''%s'')'%ts_csv_out)

if __name__ == '__main__':
    t_start = time.time()
    try:
        ticker = argv[1].strip().upper()
    except IndexError:
        exit('\t! INDEX ERROR: No 1st argument ''ticker'' entered => calling sys.exit()')
    fetchSeriesData(ticker, logging=True)
    print(f'\nfetch_data.py Total Execution Time: {str(round(time.time() - t_start, 2))} seconds')