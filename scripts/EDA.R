#### import packages ---- 

#getwd()
#setwd("C:/Users/u00378/Desktop/TCC_EST.UnB")
#setwd("C:/Users/Igor/Desktop/TCC")
library(magrittr)
library(readxl)
library(Hmisc)
library(forecast)
library(ggplot2)
library(zoo)
library(seasonal)

cores <- c('#00843d', '#003a70', '#CC9900')


theme <- function(...) {
  theme <- ggplot2::theme_bw() +
    ggplot2::theme(
      axis.title.y = ggplot2::element_text(colour = "black", size = 12),
      axis.title.x = ggplot2::element_text(colour = "black", size = 12),
      axis.text = ggplot2::element_text(colour = "black", size = 9.5),
      panel.border = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(colour = "black"),
      legend.position = "top",
      ...
    )
  
  return(
    list(
      theme,
      scale_fill_manual(values = cores),
      scale_colour_manual(values = cores)
    )
  )
}

#### Read datas from web_scraping.ipynb ----

aci_agg <- read_excel('scripts/dados/aci_agg.xlsx')
aci_esta <- read_excel('scripts/dados/aci_esta.xlsx')
aci_rodo <- read_excel('scripts/dados/aci_rodo.xlsx')


#### EDA - número de acidentes diários ----

# boxplot
ggplot(aci_agg) +
  aes(x=factor(""), y=acidentes) +
  geom_boxplot(fill=c("#00843d"), width = 0.5) +
  guides(fill='none') +
  stat_summary(fun="mean", geom="point", shape=23, size=3, fill="white")+
  labs(x="", y="Número de acidentes diários")+
  theme()
ggsave("imagens/aci_agg1.pdf", width = 158, height = 93, units = "mm")


summary(aci_agg$acidentes)
sd(aci_agg$acidentes)

## calculate moving avarage

#Make zoo object of data
temp_zoo<-zoo(aci_agg$acidentes,aci_agg$data_inversa)

#Calculate moving average with window 30d
# and make first and last value as NA (to ensure identical length of vectors)

m_av<-rollmean(temp_zoo, 30,fill = list(NA, NULL, NA))

#Add calculated moving averages to existing data frame

aci_agg$moving_avarage <- coredata(m_av)

# visual série agrupada 1
autoplot(ts(aci_agg[,c("acidentes", "moving_avarage")],
            start = 2007, frequency = 365.25), size = 0.2) +
  ggtitle("") +
  xlab("Year") +
  ylab("# acidentes")+
  scale_colour_manual(values = c('#00843d', '#003a70'))
ggsave("imagens/aci_agg2.pdf", width = 158, height = 93, units = "mm", dpi = 200)


# seasonal plot

ggseasonplot(ts(aci_agg[,"moving_avarage"],
                start = 2007, frequency = 365.25), year.labels=TRUE,
             year.labels.left=TRUE) +
  ylab("moving avarage for # acidentes (30d)") +
  ggtitle("")+
  scale_x_continuous(breaks=1:8,
                     labels=c("31/12", "16/02", "03/04", "19/05",
                              "04/07", "19/08", "04/10", "19/11"))

ggsave("imagens/aci_agg3.pdf", width = 158, height = 93, units = "mm", dpi = 200)


# DECOMPOSIÇÃO

ts(aci_agg[,"moving_avarage"],
   start = 2007, frequency = 365.25)%>% decompose(
     type="multiplicative") %>%
  autoplot(size = 0.2) + xlab("Year") +
  ggtitle("")

ggsave("imagens/aci_agg4.pdf", width = 158, height = 93, units = "mm", dpi = 200)


#### EDA - número de acidentes diários por Estado ----

autoplot(ts(aci_esta[,c("uf_MG", "uf_SC", "uf_PR", "uf_RJ", "uf_RS")],
            start = 2007, frequency = 4), size = 0.8) +
  ggtitle("") +
  xlab("Year") +
  ylab("# acidentes")+
  scale_colour_manual(values = c('#00843d', '#003a70', '#CC9900', '#A11d21', '#170c00'))
ggsave("imagens/aci_esta1.pdf", width = 158, height = 93, units = "mm", dpi = 200)

#### EDA - número de acidentes diários por rodovia ----

autoplot(ts(aci_rodo[,c("br_101.0", "br_116.0", "br_381.0", "br_40.0", "br_153.0")],
            start = 2007, frequency = 4), size = 0.8) +
  ggtitle("") +
  xlab("Year") +
  ylab("# acidentes")+
  scale_colour_manual(values = c('#00843d', '#003a70', '#CC9900', '#A11d21', '#170c00'))
ggsave("imagens/aci_rodo1.pdf", width = 158, height = 93, units = "mm", dpi = 200)