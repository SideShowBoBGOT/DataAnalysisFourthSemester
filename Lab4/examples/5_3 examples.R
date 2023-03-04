################################################
##             ПРОСТОРОВИЙ РОЗПОДІЛ           ##
##                ОБ'ЄКТІВ, ч.2               ##
################################################
# 
# Побудова картограм
# 
# library(maptools)
# 
# Читання shape-файлів
# readShapePoly("Fra_adm1.shp")
# 
# Функція побудови карт
# spplot()
# 
# Функція побудови карт ggplot2
# ggplot() + geom_map() 
# 
################################################

library(dplyr)
library(ggplot2)
library(sp)           
library(maptools)    
library(rgeos)       

setwd("~/data/Maps/FRA_adm_shp")

library(maptools)   

# Shape file:         опрацьовуються довго!!!!!!!!!!!
Regions <- readShapePoly("FRA_adm1.shp")

slotNames(Regions)    # характеристики, як поля в класах
Regions@data          # сюди будем додавати свої дані для відображення як нові стовпчики, це датафрейм
str(Regions@polygons) # це полігони, сюди не ліземо
Regions@data$NAME_1    

# відобразити карту
spplot(Regions,
       "NAME_1", 
       scales = list(draw = T), 
       col.regions = rainbow(n = 22) ) 

library(RColorBrewer) # інша палітра

spplot(Regions,
       "NAME_1", 
       scales = list(draw = T), col.regions = brewer.pal(22, "Set3"),
       par.settings = list(axis.line = list(col = NA)))

# хочемо додати кілька даних
# порядок важливий
Regions@data$Value = rnorm(22)                              # додали нову колонку з випадковими числами
mypalette <- colorRampPalette(c("seagreen", "whitesmoke"))  # задаємо палітру
mypalette  # mypalette - is a function

spplot(Regions, "Value",
       col.regions = mypalette(20),  
       col = "transparent", # without borders
       par.settings = list(axis.line = list(col = NA)))

#==============================================
# теж саме в ggplot2 

counties <- fortify(Regions, region = "NAME_1")       
str(counties)
ggplot() + geom_map(data = counties,
                    aes(map_id = id), 
                    map = counties) + 
  expand_limits(x = counties$long, y = counties$lat) +
  coord_map("polyconic")

# ggplot2 працює тільки з датафреймами, тому робимо перетворення і вказуємо що буде відображатись
fake_data <- as.data.frame(Regions@data)              
ggplot() + geom_map(data = fake_data, aes(map_id = NAME_1, fill = Value),     # звязок з картою по  map_id = NAME_1, фарбування по fill = Value              
                    map = counties) + expand_limits(x = counties$long, y = counties$lat) + coord_map("polyconic")

# в інших кольорах
library(scales) 
ggplot() + geom_map(data = fake_data,
                    aes(map_id = NAME_1, fill = Value),
                    colour = "gray",
                    map = counties) + 
  expand_limits(x = counties$long, y = counties$lat) +
  scale_fill_gradient2(low = muted("blue"), 
                       midpoint = 0,
                       mid = "white",
                       high = muted("red"),
                       limits = c(min(fake_data$Value),
                                  max(fake_data$Value))) +
  coord_map("polyconic")

#==============================================
#теж саме в  highcharter

library(highcharter)

mapdata <- get_data_from_map(download_map_data("countries/fr/fr-all-all"))
glimpse(mapdata)
data_fake <- mapdata %>%                         # подложка
  select(code = `hc-a2`) %>%                     # реріони та карти варто склеювати по коду
  mutate(value = rnorm(nrow(.),1000,500))        # дані, що відображаємо

hcmap("countries/fr/fr-all-all", data = data_fake, value = "value",      # подложка
      joinBy = c("hc-a2", "code"), name = "Fake data",                   # реріони та карти варто склеювати по коду - так помилок менше
      dataLabels = list(enabled = TRUE, format = '{point.name}'),        # point.name - вспливає на об'єктах
      borderColor = "#FAFAFA", borderWidth = 0.1,
      tooltip = list(valueDecimals = 2, valuePrefix = "", valueSuffix = " mln")) %>%  # як відображати дані
  hc_mapNavigation(enabled = TRUE)               # інструмети для масштабування

