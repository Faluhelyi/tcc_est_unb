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


# TIME TO RUN 1100m 46.9s


train_uf_df = pd.read_excel(f'{INPUT_PATH}/git_repo/results/train_uf_df.xlsx')
test_uf_df = pd.read_excel(f'{INPUT_PATH}/git_repo/results/test_uf_df.xlsx')
#train_df = train_df.iloc[:,1:]
#test_df = test_df.iloc[:,1:]

cols = list(map(lambda x: f'F{x}', range(1, 29)))
submission_file = pd.DataFrame(np.zeros(shape= (208,28)), index = np.unique(df_tcc['br']), columns = cols)
submission_file.index.names = ['br']

cv_df = pd.DataFrame(np.zeros(shape= (27,5)), index = np.unique(df_tcc['uf']), columns = ['TBATS', 'DHR', 'RAE', 'AutoARIMA', 'sNAIVE'])
cv_df.index.names = ['uf']


cols = list(map(lambda x: f'F{x}', range(1, 29)))
uf_forecasts = pd.DataFrame(np.zeros(shape= (27,28)), index = np.unique(df_tcc['uf']), columns = cols)

cols = list(map(lambda x: f'Train{x}', range(1, 5876)))
uf_true_train = pd.DataFrame(np.zeros(shape= (27,5875)), index = np.unique(df_tcc['uf']), columns = cols)

cols = list(map(lambda x: f'Test{x}', range(1, 29)))
uf_true_test = pd.DataFrame(np.zeros(shape= (27,28)), index = np.unique(df_tcc['uf']), columns = cols)

for uf in np.unique(df_tcc['uf'].dropna()):
    mae_tbats = []
    mae_dhr = []
    mae_rae = []
    mae_arima = []
    mae_snaive = []
    final_date = pd.to_datetime('2007-01-01')
    for i in range(1, 5):
        train_df_c = train_uf_df.copy()
        final_date = final_date +  np.timedelta64(4,'Y')

        test_df_c = train_df_c[(train_df_c['unique_id'] == uf) & (train_df_c['ds'] >= str((final_date + np.timedelta64(1,'D')).date()))\
                              & (train_df_c['ds'] <= str((final_date + np.timedelta64(28,'D')).date()))]
        train_df_c = train_df_c[(train_df_c['unique_id'] == uf) & (train_df_c['ds'] <= str(final_date.date()))]

        #lambda_boxcox = stats.boxcox(train_df_c['y'])[1]

        horizon = len(test_df_c) # Predict the lenght of the test df

        sf1 = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
        sf1.fit(df=train_df_c[train_df_c['unique_id']==uf])
        Y_hat_df1 = sf1.predict(h=horizon, X_df=test_df_c[test_df_c['unique_id']==uf][['unique_id', 'ds', 'is_holiday', 'effect_long_holiday', 'is_long_holiday']])

        sf2 = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
        sf2.fit(df=train_df_c[train_df_c['unique_id']==uf][['unique_id', 'ds', 'y']])
        Y_hat_df2 = sf2.predict(h=horizon)

        testee = 1000
        for k in range(1,4,1):
            sf3 = StatsForecast(models=[AutoARIMA(season_length=7, seasonal = False)], freq='D', n_jobs=-1)
            trans = FourierFeaturizer(7, k)
            y_prime, x_f = trans.fit_transform(train_df_c[train_df_c['unique_id']==uf]['y'].values)
            sf3.fit(df=pd.concat([train_df_c[train_df_c['unique_id']==uf].reset_index(drop=True), x_f], axis = 1))
            Y_hat_df3 = sf3.predict(h=horizon, X_df=pd.concat([test_df_c[test_df_c['unique_id']==uf].reset_index(drop=True), pd.DataFrame(x_f[-horizon:])\
                                                               .reset_index(drop=True)], axis = 1).drop('y', axis = 1))
            if func.mae(Y_hat_df3['AutoARIMA'].values, test_df_c['y'].values) < testee:
                dhr_preds = Y_hat_df3['AutoARIMA'].values

                testee = func.mae(Y_hat_df3['AutoARIMA'].values, test_df_c['y'].values)

            
            
        #result_auto_arima = pm.auto_arima(train.values, d = 1, start_p=0, start_q=0, max_p=3, max_q=3,\
        #                          seasonal = True, m = 7, D=1, start_P=0, start_Q=0, max_P=1, max_Q=1,\
        #                            information_criterion='aic', error_action='ignore', stepwise=True)

        result_auto_arima = pm.auto_arima(train_df_c['y'], seasonal = True, m = 7, start_p=0, start_q=0, max_p=1, max_q=1,\
                                          start_P=0, start_Q=0, max_P=1, max_Q=1)
        auto_arima_forecast= result_auto_arima.predict(28)

        result_tbats = TBATS(seasonal_periods=[7, 365.25/12, 365.25], n_jobs=1, \
                              use_box_cox= False, use_arma_errors=False)
        
        fitted_tbats = result_tbats.fit(train_df_c['y'])
        tbats_forecast_valid = fitted_tbats.forecast(steps = horizon)
        
        y_true = test_df_c['y'].values
        tbats_preds = tbats_forecast_valid
        #dhr_preds = Y_hat_df3['AutoARIMA'].values
        rae_preds = Y_hat_df1['AutoARIMA'].values
        arima_preds = auto_arima_forecast
        snaive_preds = Y_hat_df2['SeasonalNaive'].values

        mae_tbats.append(func.mae(tbats_preds, y_true))
        mae_dhr.append(func.mae(dhr_preds, y_true))
        mae_rae.append(func.mae(rae_preds, y_true))
        mae_arima.append(func.mae(arima_preds, y_true))
        mae_snaive.append(func.mae(snaive_preds, y_true))

    cv = {'TBATS': np.mean(mae_tbats), 'DHR':np.mean(mae_dhr), 'RAE': np.mean(mae_rae), 'AutoARIMA': np.mean(mae_arima), 'sNAIVE': np.mean(mae_snaive)}
    cv_df.loc[uf] = list(cv.values())

    best_model = min(cv, key=cv.get)

    train_df_c = train_uf_df.copy()
    test_df_c = test_uf_df.copy()

    train_df_c = train_df_c[(train_df_c['unique_id'] == uf) & (train_df_c['ds'] <= '2023-01-31')]
    test_df_c = test_df_c[(test_df_c['unique_id'] == uf) & (test_df_c['ds'] >= '2023-02-01')\
                              & (test_df_c['ds'] <= '2023-02-28')]
    

    uf_true_train.loc[uf] = train_df_c[train_df_c['unique_id']==uf]['y'].values
    uf_true_test.loc[uf] = test_df_c[test_df_c['unique_id']==uf]['y'].values

    #lambda_boxcox = stats.boxcox(train_df_c[train_df_c['unique_id']==uf]['y'].values)[1]
    if best_model == 'TBATS':
        result_tbats = TBATS(seasonal_periods=[7, 365.25/12, 365.25], n_jobs=1, \
                              use_box_cox= False, use_arma_errors=False) 
        fitted_tbats = result_tbats.fit(train_df_c[train_df_c['unique_id']==uf]['y'].values)
        tbats_forecast = fitted_tbats.forecast(steps = 28)

        uf_forecasts.loc[uf] = tbats_forecast

        rot = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] >= '2022-01-01')& (df_tcc['data_inversa'] <= '2023-01-31')].groupby(['br'])['id']\
            .agg(['nunique'])
        rot['proportion'] = (rot['nunique']/396)/(sum(rot['nunique'])/396) #Top-down approaches: Proportions of the historical averages

        for br in rot.index.values:
            submission_file.loc[br] = submission_file.loc[br] +  (tbats_forecast*rot.loc[br]['proportion'])

    elif best_model == 'DHR':

        testee = 1000
        for k in range(1,4,1):
            sf1 = StatsForecast(models=[AutoARIMA(season_length=7, seasonal = False)], freq='D', n_jobs=-1)
            trans = FourierFeaturizer(7, k)
            y_prime, x_f = trans.fit_transform(train_df_c[train_df_c['unique_id']==uf]['y'].values)
            sf1.fit(df=pd.concat([train_df_c[train_df_c['unique_id']==uf].reset_index(drop=True), x_f], axis = 1))
            Y_hat_df1 = sf1.predict(h=28, X_df=pd.concat([test_df_c[test_df_c['unique_id']==uf].reset_index(drop=True), pd.DataFrame(x_f[-28:])\
                                                          .reset_index(drop=True)], axis = 1).drop('y', axis = 1))
            if func.mae(Y_hat_df3['AutoARIMA'].values, test_df_c['y'].values) < testee:
                dhr_forecast = Y_hat_df1['AutoARIMA'].values

                testee = func.mae(Y_hat_df3['AutoARIMA'].values, test_df_c['y'].values)

        uf_forecasts.loc[uf] = dhr_forecast
        
        rot = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] >= '2022-01-01')& (df_tcc['data_inversa'] <= '2023-01-31')].groupby(['br'])['id']\
            .agg(['nunique'])
        rot['proportion'] = (rot['nunique']/396)/(sum(rot['nunique'])/396) #Top-down approaches: Proportions of the historical averages

        for br in rot.index.values:
            submission_file.loc[br] = submission_file.loc[br] +  (dhr_forecast*rot.loc[br]['proportion'])


    elif best_model == 'RAE':
        sf1 = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
        sf1.fit(df=train_df_c[train_df_c['unique_id']==uf])
        Y_hat_df1 = sf1.predict(h=28, X_df=test_df_c[test_df_c['unique_id']==uf][['unique_id', 'ds', 'is_holiday', 'effect_long_holiday', 'is_long_holiday']])
        rae_forecast = Y_hat_df1['AutoARIMA'].values

        uf_forecasts.loc[uf] = rae_forecast

        rot = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] >= '2022-01-01')& (df_tcc['data_inversa'] <= '2023-01-31')].groupby(['br'])['id']\
            .agg(['nunique'])
        rot['proportion'] = (rot['nunique']/396)/(sum(rot['nunique'])/396) #Top-down approaches: Proportions of the historical averages
        for br in rot.index.values:
            submission_file.loc[br] = submission_file.loc[br] +  (rae_forecast*rot.loc[br]['proportion'])
            
    elif best_model == 'AutoARIMA':
        sf1 = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
        sf1.fit(df=train_df_c[train_df_c['unique_id']==uf][['unique_id', 'ds', 'y']])
        Y_hat_df1 = sf1.predict(h=28)
        autoarima_forecast = Y_hat_df1['AutoARIMA'].values
        uf_forecasts.loc[uf] = autoarima_forecast

        rot = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] >= '2022-01-01')& (df_tcc['data_inversa'] <= '2023-01-31')].groupby(['br'])['id']\
            .agg(['nunique'])
        rot['proportion'] = (rot['nunique']/396)/(sum(rot['nunique'])/396) #Top-down approaches: Proportions of the historical averages
        for br in rot.index.values:
            submission_file.loc[br] = submission_file.loc[br] +  (autoarima_forecast*rot.loc[br]['proportion'])

    elif best_model == 'sNAIVE':
        sf1 = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
        sf1.fit(df=train_df_c[train_df_c['unique_id']==uf][['unique_id', 'ds', 'y']])
        Y_hat_df1 = sf1.predict(h=28)
        snaive_forecast = Y_hat_df1['SeasonalNaive'].values

        uf_forecasts.loc[uf] = snaive_forecast

        rot = df_tcc[(df_tcc['uf'] == uf) & (df_tcc['data_inversa'] >= '2022-01-01')& (df_tcc['data_inversa'] <= '2023-01-31')].groupby(['br'])['id']\
            .agg(['nunique'])
        rot['proportion'] = (rot['nunique']/396)/(sum(rot['nunique'])/396) #Top-down approaches: Proportions of the historical averages
        for br in rot.index.values:
            submission_file.loc[br] = submission_file.loc[br] +  (snaive_forecast*rot.loc[br]['proportion'])



submission_file.to_excel(f'{INPUT_PATH}/git_repo/results/submission_file_br.xlsx')
cv_df.to_excel(f'{INPUT_PATH}/git_repo/results/cv_summary.xlsx')
uf_forecasts.to_excel(f'{INPUT_PATH}/git_repo/results/uf_forecasts.xlsx')
uf_true_train.to_excel(f'{INPUT_PATH}/git_repo/results/uf_true_train.xlsx')
uf_true_test.to_excel(f'{INPUT_PATH}/git_repo/results/uf_true_test.xlsx')