#########################
### Working directory ###
#########################
library(glue)
tryCatch(stop(setwd("C:/Users/igor/Desktop/TCC_EST_UNB")),
         error=function(e) print('Voce esta na brb seguros'),
         finally=setwd("C:/Users/u00378/Desktop/TCC_EST_UNB"))

########################
### Import .pkl file ###
########################
#install.packages('reticulate')
require("reticulate")

PATH = glue("{getwd()}")
source_python(glue("{PATH}/scripts/pickle_reader.py"))
pickle_data <- read_pickle_file(glue("{PATH}/dados_tcc.pkl"))

###########
### EDA ###
###########

# DEPRECATED FOR LARGE MEMORY USAGE!!!
#
