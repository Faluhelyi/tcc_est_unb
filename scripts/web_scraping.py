##########################################################################################
### .py para alcançar, via web scraping, os dados, da PRF, que serao utilizados no TCC ###
##########################################################################################
from bs4 import BeautifulSoup
import requests
from pyunpack import Archive
import zipfile

##############################
### Get links for download ###
##############################
INPUT_PATH = "C:/Users/u00378/Desktop/tcc_est_unb"
#INPUT_PATH = "C:/Users/Igor/Desktop/TCC"
url = 'https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes'

agent = "Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
# Making a GET request
# , headers={"User-Agent": agent} in r = requests.get(url)
r = requests.get(url, headers={"User-Agent": agent})
 
# check status code for response received
# success code - 200
print(f"Acesso ao site liberado para web scraping" if r.status_code == 200 else f"Acesso negado ao site para web scraping")

# Parsing the HTML
soup = BeautifulSoup(r.content, 'html.parser') #[2484:2569]
s = soup.find_all('a', class_= 'external-link')

links = []
for i in range(len(s)):
    links.append(s[i]['href'])

links = links[4:22]
links.remove('https://arquivos.prf.gov.br/arquivos/index.php/s/n1T3lymvIdDOzzb')

ids = [i[32:65] for i in links]
urls = [f'https://drive.google.com/u/0/uc?id={i}&export=download' for i in ids]

###########################
### Download .ZIP files ###
###########################
print('BEGINNING OF DOWNLOADS...')
name = 2023
for i in range(len(urls)):
    url = urls[i]
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f'{INPUT_PATH}/dados/zips/{name}.zip', 'wb') as file:
            file.write(response.content)
        print(f'Arquivo baixado com sucesso: {name}: {url}')
    else:
        print(f'Falha ao baixar arquivo. Código de resposta: {response.status_code}')
        print(f'Erro no download de: {name} {url}')
        print("DOWNLOAD FAILED")
        break
    name = name -1

print('END OF DOWNLOADS')

################################
### Extracting .ZIP Archives ###
################################
print('BEGINNING OF EXTRACTION...')
name = 2023
for i in range(len(urls)):
    try:
        #with zipfile.ZipFile(f'{INPUT_PATH}/dados/zips/{name}.zip', 'r') as zip_ref:
        #    zip_ref.extractall(f'{INPUT_PATH}/dados')
        Archive(f'{INPUT_PATH}/dados/zips/{name}.zip').extractall(f'{INPUT_PATH}/dados')
        print(f'O arquivo {name} foi extraído com sucesso.')
    except:
        print(f'O arquivo {name} não foi encontrado.')
    name = name -1

print('END OF EXTRACTION')