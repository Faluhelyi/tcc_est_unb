import pandas as pd
import numpy as np
import glob


INPUT_PATH = "C:/Users/Igor/Desktop/TCC"

path_df = np.array([])
for i in glob.glob(f"{INPUT_PATH}/dados/zips/*"):
    path_df = np.append(path_df, i)


for i in range(len(path_df)):
    if i == 0:
        df = pd.read_csv(path_df[i], sep = ',', encoding='latin', engine = 'pyarrow',\
                         dtype = {'km':'string[pyarrow]'},\
                            na_values= '(null)', dtype_backend = 'pyarrow')

    else:
        try:
            supp = pd.read_csv(path_df[i], sep = ',', encoding='latin', engine = 'pyarrow',\
                               dtype = {'km':'string[pyarrow]'},\
                                na_values= '(null)', dtype_backend = 'pyarrow')
            if len(supp.columns) > 1:
                df = pd.concat([df, supp], ignore_index=True)
            else:
                supp = pd.read_csv(path_df[i], sep = ';', encoding='latin', engine = 'pyarrow',\
                                   dtype = {'km':'string[pyarrow]'},\
                                    na_values= '(null)', dtype_backend = 'pyarrow')
                df = pd.concat([df, supp], ignore_index=True)

        except:
            supp = pd.read_csv(path_df[i], sep = ';', encoding='latin', engine = 'pyarrow',\
                               dtype = {'km':'string[pyarrow]'},\
                                na_values= '(null)', dtype_backend = 'pyarrow')
            df = pd.concat([df, supp], ignore_index=True)

#######################
#### Data wrangling ###
#######################

### ARQUIVO PRINCIPAL DO TCC
df['data_inversa'] = pd.to_datetime(df['data_inversa'], format = 'mixed', dayfirst=True)
df.to_pickle(f"{INPUT_PATH}/git_repo/dados_tcc.pkl") #large file format