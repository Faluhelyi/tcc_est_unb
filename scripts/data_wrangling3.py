########################
### DATA WRANGLING 3 ###
########################

import tqdm.auto
import pandas as pd
import numpy as np
import holidays
import func
import math
import pmdarima as pm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive #Imports the models you will use
from datetime import date
from dateutil.relativedelta import relativedelta
import datetime
from tbats import TBATS
from pmdarima.preprocessing import FourierFeaturizer
from scipy import stats

#INPUT_PATH = "C:/Users/u00378/Desktop/tcc_est_unb"
INPUT_PATH = "C:/Users/Igor/Desktop/TCC"

df = pd.read_pickle(f'{INPUT_PATH}/dados/tcc_data.pkl')


#########################
### Regiões do brasil ###
#########################
#NORTE ok
norte = ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC']
#NORDESTE ok
nordeste = ['MA', 'PI', 'CE', 'RN', 'PE',\
            'PB', 'SE', 'AL', 'BA']
#CENTRO-OESTE + DF ok
centro_oeste = ['MT', 'MS', 'GO', 'DF']
#SUDESTE ok
sudeste = ['SP', 'RJ', 'ES', 'MG']
#SUL ok
sul = ['PR', 'RS', 'SC']


df['weekday'] = list(map(lambda x: x.weekday(), df['data_inversa']))
df = pd.get_dummies(df, columns=['weekday'], drop_first=True)

isnt_na = lambda x: False if bool(__import__('pandas').isna(x)) else True
filtro1 = np.array(list(map(isnt_na, df['uf'])))
filtro2 = np.array((list(map(isnt_na, df['br']))))

df_tcc = df[(filtro1) & (filtro2)]


def effect_long_holiday(date, uf):
    import holidays
    hholidays = holidays.country_holidays('BR', subdiv=uf) 
    date_after1 = pd.to_datetime(date) + np.timedelta64(1,'D')
    date_after2 = pd.to_datetime(date) + np.timedelta64(2,'D')
    date_after3 = pd.to_datetime(date) + np.timedelta64(3,'D')
    date_after4 = pd.to_datetime(date) + np.timedelta64(4,'D')

    if (date_after1.day_name() == 'Friday') and ((hholidays.get(date_after1)!= None) or (hholidays.get(date_after2)== 'Carnaval')):
        return 1
    elif (date_after3.day_name() == 'Monday') and ((hholidays.get(date_after3)!= None) or (hholidays.get(date_after4)== 'Carnaval')):
        return 1
    
    else:
        return 0
    

def is_long_holiday(date, uf):
    import holidays
    hholidays = holidays.country_holidays('BR', subdiv=uf)
    date = pd.to_datetime(date)
    date_after1 = pd.to_datetime(date) + np.timedelta64(1,'D')
    if (date.day_name() == 'Friday') and ((hholidays.get(date)!= None) or (hholidays.get(date_after1)=='Carnaval')):
        return 1
    else:
        return 0
    

train_uf_df = pd.DataFrame()
test_uf_df = pd.DataFrame()
for uf in np.unique(df_tcc['uf'].dropna()):
    train = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] <= '2023-01-31')].groupby('data_inversa')['id'].agg('count').resample('D').sum()
    test = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] >= '2023-02-01')&\
                  (df_tcc['data_inversa'] <= '2023-02-28')].groupby('data_inversa')['id'].agg('count').resample('D').sum()
    
    index_train = pd.date_range(start = '2007-01-01', end = '2023-01-31', freq = 'd')
    train = train.reindex(index=index_train, fill_value=0)

    index_test = pd.date_range(start = '2023-02-01', end = '2023-02-28', freq = 'd')
    test = test.reindex(index=index_test, fill_value=0)

    holidays_train = list(map(lambda x: 1 if holidays.country_holidays('BR', subdiv=uf).get(x) != None else 0,index_train))
    effect_long_holiday_train = list(map(lambda x: effect_long_holiday(x, uf), index_train))
    long_holiday_train = list(map(lambda x: is_long_holiday(x, uf), index_train))
    

    holidays_test_test = list(map(lambda x: 1 if holidays.country_holidays('BR', subdiv=uf).get(x) != None else 0, index_test))
    effect_long_holiday_test = list(map(lambda x: effect_long_holiday(x, uf), index_test))
    long_holiday_test = list(map(lambda x: is_long_holiday(x, uf), index_test))
    

    train_uf_df = pd.concat([train_uf_df, pd.DataFrame({'unique_id': [uf]*len(index_train), 'ds':index_train, 'y':train, 'is_holiday':holidays_train,\
                                                  'effect_long_holiday':effect_long_holiday_train, 'is_long_holiday':long_holiday_train})], axis = 0)
    
    test_uf_df = pd.concat([test_uf_df, pd.DataFrame({'unique_id': [uf]*len(index_test), 'ds':index_test, 'y':test, 'is_holiday':holidays_test_test,\
                                                  'effect_long_holiday':effect_long_holiday_test, 'is_long_holiday':long_holiday_test})], axis = 0)



train_uf_df.to_excel(f'{INPUT_PATH}/git_repo/results/train_uf_df.xlsx', index = False)
test_uf_df.to_excel(f'{INPUT_PATH}/git_repo/results/test_uf_df.xlsx', index = False)