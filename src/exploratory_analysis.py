import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from datetime import datetime
import sys
import warnings
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

def do_analysis():
    df = pd.read_csv("../data/forex.csv")
    print(df.shape)
    df['date'] = pd.to_datetime(df['date'])
    print()
    for x in set(df['slug'].str.split('/', expand=True)[0].unique()):
        print (x)
    Cur1 = 'INR'
    df_IC = df[df['slug'].str.contains(Cur1 + '/')]
    scatter_plot(Cur1, df_IC)

def scatter_plot(Cur1, df_IC):
    for Cur2 in df_IC['currency'].unique():
        plt.figure(figsize=(18,6))
        df_temp = df_IC[df_IC['currency'] == Cur2]
        plt.scatter(df_temp['date'], df_temp['close'])
        plt.title("1 " + Cur1 + " = x " + Cur2, fontsize = 20,fontweight = 'bold', color='blue')
        plt.xlabel("Date", fontsize = 14, fontweight='bold', color='red')
        plt.ylabel('Closing Value', fontsize = 14, fontweight='bold', color='red')
        plt.show()