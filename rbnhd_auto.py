from re import L
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import shutil
from time import sleep
import asyncio

import datetime
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib import dates
import matplotlib.dates as mdates

from datetime import timedelta
import numpy as np
import plotly.graph_objects as go

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.util import Pt
from pptx.util import Inches

from tqdm import tqdm
import robin_stocks
# from robin_stocks import *
import robin_stocks.robinhood as rs


# Hard code a login file OUTSIDE of directory, then hardcode the path here
pass_pull_file = r'C:\Users\dean.huang\main\reallife\spending\login.txt'
# First line username, second line password

# Working directory
long_dir = r'C:\Users\dean.huang\main\reallife\spending\robinhood_auto'


def rs_login(pass_file):
    pass_list = []
    with open(pass_file) as file:
        for line in file:
            pass_list.append(line.rstrip())
    rs.login(username=pass_list[0],
            password=pass_list[1],
            expiresIn=86400,
            by_sms=True)
    # rs.logout()




def limit_sell(tick, poll_time):
    df = pd.DataFrame(columns=['date', 'price'])
    
    base_price = float(rs.stocks.get_latest_price(tick, includeExtendedHours=True)[0])
    while True:
        price = float(rs.stocks.get_latest_price(tick, includeExtendedHours=True)[0])
        df.loc[len(df)] = [pd.Timestamp.now(), price]
        if price < 0.99*base_price:
            print("Price has dropped 1%, time to sell!")
            break
        elif price > 1.03*base_price:
            print("Price has raised 3%, time to sell!")
            break
        else:
            sleep(poll_time)

    # start_time = df.date.iloc[-1] - pd.Timedelta(minutes=60)
    # df = df.loc[df.date >= start_time] # cuts dataframe to only include last hour of data
    # max_price = df.price.max()
    # min_price = df.price.min()
    # print(df)
    # print(max_price, min_price)

def sell():
    pass



async def main():
    rs_login(pass_pull_file)
    limit_sell('NVDA', 60)



    # Stock Notes: 
    # Start at 10:30, ends at 5:00
    # Should I keep track of gold, FED interest rate, inflation, etc?
    # Keeping track of Stocks is already good enough?








if __name__ == '__main__':
    asyncio.run(main())

