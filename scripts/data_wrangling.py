######################
### DATA WRANGLING ###
######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

#INPUT_PATH = "C:/Users/u00378/Desktop/tcc_est_unb"
INPUT_PATH = "C:/Users/Igor/Desktop/TCC"
year = 2007
df = pd.read_csv(f"{INPUT_PATH}/dados/datatran{year}.csv", encoding='latin1', on_bad_lines='skip', sep =';',\
                 dtype={'br':'object','km':'object'}, na_values='(null)')
for i in range(1, 17):
        year +=1
        df1 = pd.read_csv(f"{INPUT_PATH}/dados/datatran{year}.csv", encoding='latin1', on_bad_lines='skip', sep =';',\
                          dtype={'br':'object','km':'object'}, na_values='(null)')
        #df1.columns=df.columns
        df = pd.concat([df,df1], ignore_index=True)

df['data_inversa'] = pd.to_datetime(df['data_inversa'], format = 'mixed', dayfirst=True)

df.to_pickle(f'{INPUT_PATH}/dados/tcc_data.pkl')
print("Dados salvos com sucesso!")