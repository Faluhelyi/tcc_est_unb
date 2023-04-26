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
INPUT_PATH = "C:/Users/Igor/Desktop/TCC"

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

for i in range(len(urls)):
    url = urls[i]
    #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    #cookies = {'session_id': 'seu_valor_de_sessao'}
    response = requests.get(url)
    name = coverpage_useful[5:][i].get_text()
    if response.status_code == 200:
        with open(f'{INPUT_PATH}/dados/zips/{name}.zip', 'wb') as file:
            file.write(response.content)
        print(f'Arquivo baixado com sucesso: {url}')
    else:
        print(f'Falha ao baixar arquivo. Código de resposta: {response.status_code}')
        print(f'Erro no download do link {url}')

print('END OF THE DOWNLOADS')
###############################################
### Extracting .ZIP Archives & reading them ###
###############################################

# , outdir='dados'
for i in range(len(urls)):
    name = coverpage_useful[5:][i].get_text()
    try:
        patoolib.extract_archive(f'{INPUT_PATH}/dados/zips/{name}.zip', outdir=f'{INPUT_PATH}/dados')
    except:
        print('O arquivo não foi encontrado.')

