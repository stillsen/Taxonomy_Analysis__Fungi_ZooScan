# Title     : TODO
# Objective : TODO
# Created by: stillsen
# Created on: 14.05.20


library('plotrix')
library('ggplot2')
library('iRF')
library(dplyr)

path <- '/home/stillsen/Documents/Data/Results/ComparisionPlot'
filename <- 'comparisonPlot.csv'
abs_filename <- paste(path, '/', filename, sep='')

# read csv, which is actually tsv ^^
df <- read.table(abs_filename, sep=',', header=TRUE, stringsAsFactors = TRUE)
# forget about zero filled first column
# df <- df[,-1]
# attach to search path for easier variable access
attach(df)

df %>%
  arrange(Rank) %>%
  mutate(Rank = factor(Rank, levels=c("phylum", "class", "order", "family", "genus", "species")))%>%
  arrange(Rank)%>%
ggplot()+
  geom_point(aes_string(y="Score", x="Rank", colour="Model", shape="Dataset", group="Model"), size = 2.5, position =position_dodge(width = .25))

file_name <- paste(path,'/',"comparisonPlot.pdf", sep="")
ggsave(file_name)