from re import L
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import shutil
import time
from time import sleep
import asyncio
import random
import statistics as stat
import ast

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


async def sell_loop(tick, wait, min, max, rounds):
    start = time.time()
    buyin = initial = 100
    final = -1
    transaction_count = 0
    wl_count = 0
    while transaction_count < rounds:
        rolling_credit = await limit_sell(tick, wait, buyin, min, max)
        print("Sell happened! Buyin: %.2f, Sold: %.2f" % (buyin, rolling_credit))
        if rolling_credit > buyin: wl_count += 1
        buyin = rolling_credit
        sleep(60)
        transaction_count += 1
        
    # Final has rolling credit score,
    ratio = round(buyin/initial, 2)
    wl_ratio = round(wl_count/transaction_count,2)
    print("Finished %i transactions! Buyin: %.2f, Final: %.2f, Return: %.2f, WL: %.2f" 
        % (transaction_count, buyin, final, ratio, wl_ratio))
    print("Trading Time for %s : %.2f" % (tick, round(time.time() - start,2)))

    return ratio, wl_ratio

        
        



async def limit_sell(tick, poll_time, buyin, min, max):
    df = pd.DataFrame(columns=['date', 'price'])
    base_price = float(rs.stocks.get_latest_price(tick, includeExtendedHours=True)[0])
    print("Base price for %s is $%.2f" % (tick, base_price))
    while True:
        price = float(rs.stocks.get_latest_price(tick, includeExtendedHours=True)[0])
        print("Current price for %s is $%.2f" % (tick, price))
        df.loc[len(df)] = [pd.Timestamp.now(), price]
        if price < min*base_price:
            print("Price has dropped 2p to %.2f, return %.2f!" % (price, price*buyin/base_price))
            return price * (buyin/base_price)
        elif price > max*base_price:
            print("Price has raised 1p to %.2f, return %.2f!" % (price, price*buyin/base_price))
            return price * (buyin/base_price)
        else:
            sleep(poll_time)

    # start_time = df.date.iloc[-1] - pd.Timedelta(minutes=60)
    # df = df.loc[df.date >= start_time] # cuts dataframe to only include last hour of data
    # max_price = df.price.max()
    # min_price = df.price.min()
    # print(df)
    # print(max_price, min_price)

def sell():
    # write the sell function here, will sell all 100% of dollar amount available...
    pass



# Input: Min, Confidence, Generations, Samples
# Output 1: Statistics, list [0, 25, 50, 75, 100, avg, std]
# Output 2: Table that shows list of statistics... for higher pullouts, like 1.05, 1.06, 1.07, etc...
# Output 3: Solver. Solve for which minimum pullout will I allow! super important... Basically just return which exact ratio gives me closest to 1 return!

# Goal: Graph Min, Confidence vs. Max

# Have a table that ONLY shows recommended Max pullout ratio, then each one will have the 5, 10, 50, 100, 1000 expected rates
def pullout_solver(min, conf, gen, samp):
    level = 0
    max, step = 1.0,.1
    while level < 4:
        last_gen_med = 0
        samp_list, last_gen = [], []
        while last_gen_med < 1.0: 
            samp_list = pullout_stats(min, max, conf, gen, samp)
            last_gen = [s[-1] for s in samp_list]
            last_gen_med = stat.median(last_gen)
            max+=step
        max = max - 2*step
        step = step /10
        level += 1
    return max

def pullout_stats(min, max, conf, gen, samp):
    samp_list = [generate_sample_list(min, max, conf, gen) for _ in range(samp)]
    return samp_list

def generate_sample_list(min, max, conf, gen):
    n, g_list = 1, [1]
    for _ in range(gen-1):
        n = n*max if random.randint(1,100)>=conf else n*min
        g_list.append(n)
    return g_list


def graph_3D():
    recal = 0

    filename = 'plot_data_30_70.txt'
    conf_list = [b for b in range(35, 70, 5)]
    min_list = [.92 + .0025*(m+1) for m in range(24)]

    # filename = 'plot_data.txt'
    # conf_list = [5*(b+1) for b in range(19)]
    # min_list = [.89 + .005*(m+1) for m in range(21)]

    graph_hub = []
    if recal:
        for c in tqdm(conf_list):
            for m in (min_list):
                temp_list = [m, c, round(pullout_solver(m, c, 1000, 80),3)]
                graph_hub.append(temp_list)
        write_file(str(graph_hub), filename)


    # Read the file in folder
    print(filename)
    data_points = eval(read_file(filename))
    
    gm_list = x = [row[0] for row in data_points]
    gc_list = y = [row[1] for row in data_points]
    gp_list = z = [row[2] for row in data_points] 


    fig,ax=plt.subplots(1,1)
    X, Y = np.meshgrid(min_list, conf_list)
    z_2 = []
    temp_z = []
    counter = 0
    for zs in z:
        temp_z.append(zs)
        counter += 1
        if counter == len(min_list): # reached length
            counter = 0
            z_2.append(temp_z)
            temp_z = []

    
    Z = np.array(z_2)
    # Z = np.transpose(np.array(z_2))

    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)
    ax.grid(visible = True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
    ax.set_xlabel('Min')
    ax.set_ylabel('Conf')

    
    # ax.set_zlabel('Pull')



    plt.title("Pull Ratio vs. (Min Pull and Market Confidence)")
    plt.show()

def write_file(in_message, filename):
    f = open(filename, "w")
    f.write(in_message)
    f.close()

def read_file(filename):
    f = open(filename, "r")
    return f.read()


async def run_simulation(sim):
    # run sell_loop 20 times
    # try: sell loop
    # except" fails
    # at the end, there will be a 20 list of simulation results
    # print this out into an excel sheet



    sim_list = []
    ranges = [[.99, 1.01], [.98,1.01], [.99, 1.02]]
    count = 0
    for s in range(sim):
        try:
            start_time = time.time()
            curr_sim = []
            wait, min, max, rounds = 15, 0.98, 1.01, 20
            min, max = ranges[count % 3][0], ranges[count % 3][1] #reassign minmax
            # f1 = asyncio.create_task(sell_loop("UUUU", wait, min, max, rounds))
            # f2 = asyncio.create_task(sell_loop("BBBY", wait, min, max, rounds))
            # f3 = asyncio.create_task(sell_loop('VAXX', wait, min, max, rounds))
            ratio, wl_ratio = await sell_loop('COSM', wait, min, max, rounds)

            elapsed_time = round(time.time() - start_time, 2)
            curr_sim = [wait, min, max, rounds, ratio, wl_ratio, elapsed_time]

            sim_list.append(curr_sim)
            count += 1
        except:
            print('Simulation Error')
    
    df = pd.DataFrame(sim_list)
    df.to_csv('first_test.csv', index=False, header=False)
    return sim_list

        



async def main():
    rs_login(pass_pull_file)
    slist = await run_simulation(10)





    # min_pull, conf, gen, samp = 0.99, 60, 1000, 100
    # pullout_solver(min_pull, conf, gen, samp)

    # graph_3D()

    






    # Stock Notes: 
    # Start at 10:30, ends at 5:00
    # Should I keep track of volume, inflation, gold, silver (elements), FED interest rate, inflation, etc?
    # Keeping track of Stocks is already good enough?

    # Dean Kitchen Thoughts
    # 1) The Confidence (Favor) of Market changes rapidly. Not so simple
    # as a stagnant percentage of me winning/losing. Maybe Find trend?
    # 2) Use Machine Learning to figure out at certain patterns what 
    # Confidence proved to be true, then use that as my basis
    # 3) Figure out the width, variance of a stock. How likely
    # it is to reach 3%, or 5%, etc... then I could use that as prediction 

    # 221209
    # Related to KT1
    # Create a big simulation
    # Try every 15 minutes...
    # Then have 20 reports of "20 rolls" to see how well I did. Have a time stamp on start and end as well...
    # Print this out into a csv
    # Inputs: # rolls, min, max
    # Outputs: return ratio, W/L ratio, total time
    # Have these trade, like test [0.99, 1.02], [.98, 1.01], [.99, 1.01] have these 3 alternate...






if __name__ == '__main__':
    asyncio.run(main())
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    # loop.close()



















    ''' Plot 3D extra plot stuff
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection='3d')
    ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
    # Creating color map
    my_cmap = plt.get_cmap('hsv')
 
    # Creating plot
    sctt = ax.scatter3D(x, y, z,
                        alpha = 0.8,
                        c = z,
                        cmap = my_cmap,
                        marker ='P')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
    # for m, zlow, zhigh in [('o', 1, 1.05), ('^', 1.051, 5)]:
    #     xs = gm_list
    #     ys = gc_list
    #     zs = gp_list
    #     ax.scatter(xs, ys, zs, marker=m)
    '''

    '''
    # ax.scatter(gm_list, gc_list, gp_list, c=gp_list, cmap='viridis', linewidth=0.5)
    # ax.plot_trisurf(gm_list, gc_list, gp_list, cmap='viridis', edgecolor='none')
    # ax.scatter3D(gm_list, gc_list, gp_list, cmap= "Greens")
    '''