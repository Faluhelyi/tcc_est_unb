##############################################################################################
### Arquivo para alcançar, via web scraping, os dados, da PRF, que serao utilizados no TCC ###
##############################################################################################
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import wget
import zipfile
import warnings
import patoolib
import shutil
import os
import glob
import matplotlib.pyplot as plt
import urllib.request
import pyarrow

##########################
### Dowload .ZIP files ###
##########################
warnings.filterwarnings('ignore')

# Request
r1 = requests.get('https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes')
r1.status_code
 
# We'll save in coverpage the cover page content
coverpage = r1.content

# Soup creation
soup1 = BeautifulSoup(coverpage, 'html.parser')

# useful identification
coverpage_useful = soup1.find_all('a', class_='external-link')

#coverpage_useful = np.delete(coverpage_useful, 6)

# links to dowload
links = np.array([])
for i in range(len(coverpage_useful[5:])):
    links = np.append(links, coverpage_useful[5:][i]['href'])


ids = [i[32:65] for i in links]

urls = [f'https://drive.google.com/u/0/uc?id={i}&export=download' for i in ids]

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'), ('Accept','text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'), ('Accept-Encoding','gzip, deflate, br'),\
    ('Accept-Language','en-US,en;q=0.5' ), ("Connection", "keep-alive"), ("Upgrade-Insecure-Requests",'1')]

urllib.request.install_opener(opener)

PATH = 'C:/Users/Igor/Desktop/TCC/git_repo'

if os.path.isfile(f'{PATH}/zips') == False:
    shutil.rmtree(f'{PATH}/zips')
    os.makedirs(f'{PATH}/zips')

else:
    os.makedirs(f'{PATH}/zips')

errors_url = []
for i in range(len(urls)):
    url = urls[i]
    try:
        name = coverpage_useful[5:][i].get_text()
        urllib.request.urlretrieve(url, f'{PATH}/zips/{name}.zip')

    except:
        print(f'Erro no download do link {url}')
        errors_url.append(url)

###############################################
### Extracting .ZIP Archives & reading them ###
###############################################
if os.path.isfile('C:/Users/Igor/Desktop/TCC/dados') == False:
    shutil.rmtree('C:/Users/Igor/Desktop/TCC/dados')
    os.makedirs('C:/Users/Igor/Desktop/TCC/dados')

else:
    os.makedirs('C:/Users/Igor/Desktop/TCC/dados')

# , outdir='dados'
for i in range(len(urls)):
    name = coverpage_useful[5:][i].get_text()
    try:
        patoolib.extract_archive(f'{PATH}/zips/{name}.zip', outdir='C:/Users/Igor/Desktop/TCC/dados')
    except:
        print('O arquivo não foi encontrado.')

path_df = np.array([])
for i in glob.glob('C:/Users/Igor/Desktop/TCC/dados/*'):
    path_df = np.append(path_df, i)


for i in range(len(path_df)):
    if i == 0:
        df = pd.read_csv(path_df[i], sep = ',', encoding='latin')

    else:
        try:
            supp = pd.read_csv(path_df[i], sep = ',', encoding='latin')
            if len(supp.columns) > 1:
                df = pd.concat([df, supp], ignore_index=True)
            else:
                supp = pd.read_csv(path_df[i], sep = ';', encoding='latin')
                df = pd.concat([df, supp], ignore_index=True)

        except:
            supp = pd.read_csv(path_df[i], sep = ';', encoding='latin')
            df = pd.concat([df, supp], ignore_index=True)

#######################
#### Data wrangling ###
#######################

### ARQUIVO PRINCIPAL DO TCC
df['data_inversa'] = pd.to_datetime(df['data_inversa'], format = 'mixed', dayfirst=True)
df.to_pickle(f'{PATH}/dados_tcc.pkl') #large file format
df.to_pickle('C:/Users/Igor/Desktop/TCC/dados/dados_tcc.pkl') #large file format
