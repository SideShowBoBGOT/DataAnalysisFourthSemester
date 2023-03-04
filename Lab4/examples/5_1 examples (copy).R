################################################
##    ВИКОРИСТАННЯ КАРТ ДЛЯ ВІЗУАЛІЗАЦІЇ      ##
################################################
# 
install.packages("highcharter", dep=T) 
install.packages('geosphere',dep=T)

################################################
# 
# 1. Особливості
# 2. Генерація досліджуваних об'єктів
# 3. Області застосування
#    
# Читання картинки  (https://gadm.org/download_country.html)       працюємо з растровим зображенням
# image <- readJPEG("France.jpg")
# gadm<-readRDS("FRA_adm0.rds")
# france <- fortify(gadm)
# 
# Виведення картинки
# ggplot() + geom_map()
# 
# або                                                             rds файл - полігон, що описує територію (по контуру)
# 
# plot()
# rasterImage(image, pars)
# 
# library(highcharter)                                            для зображень з великою кількістю деталей
# hcmap()
# http://jkunst.com/highcharter/highmaps.html 
# https://code.highcharts.com/mapdata/
#   
# Встановлення вектору координат
# locator()
# 
# Відстань між точками 
# dist(df)
# 
# library(geosphere)
# its in meters
# distHaversine(c(0, 45.0), c(-2, 47.2)) 
# distVincentyEllipsoid(c(0, 45.0), c(-2, 47.2)) 
# 
################################################

# Робота з растровим зображенням
#install.packages('jpeg',dep=T)
library(jpeg)   

setwd("~/data/Maps")
image <- readJPEG("France.jpg")

Xras = 800; Yras = 731                                             # подивитись у властивостях зображення
Reg <- c("Paris","Orlean","Lion","Tulusa")                         # що будемо опрацьовувати

par(mar = c(0, 0, 0, 0))                                           # параметри візуалізації, прибрати відступи
?par                                                               # параметри візуалізації
# міняєм тільки це
plot(1,xlim = c(0, Xras), ylim = c(0, Yras), xlab = "", ylab = "") # підготували область візуалізації
lim <- par()                                                       # зберегли характеристики візуалізації - запамятали їх
lim$usr                                                            # це координати, куди розмістити зображення - подивились їх

rasterImage(image, lim$usr[1], lim$usr[3], lim$usr[2], lim$usr[4])

# отримати координати інтерактивно
xy <- locator(4)     # чекає, поки вказану кількість разів тикнете в картинку і запамятає координати
xy                   # отримані координати

# хочемо відобразити на карті ці числа, привязані до координат
Estimates <- c(89,54,64,32)

# нормування чисел, що будуть відображаться, min=2, max=10*(max(Estimates) - min(Estimates))/max(Estimates) + 2
mycex = 10*(Estimates - min(Estimates))/max(Estimates) + 2 

colpts = rgb(0.2, 0.5, 0.4, alpha = 0.6)                           # задати колір та прозорість маркерів
points(xy$x, xy$y,cex = mycex, col = 1, pch = 21, bg = colpts)     # відобразити маркери

# відстань між точками в пікселях
df <- data.frame(xy)
dist(df)


#==============================================
# робота з картинками, що задані контуром (полігоном)
library(sp)
library(maptools)
library(ggplot2)
library(rgeos)

setwd("~/data/Maps")
gadm<-readRDS("FRA_adm0.rds")                                      # source https://gadm.org/download_country.html
# adm0 вся країна
# adm1 області
# adm2 райони
# .rds - зручно для R
# .shp - зручно для всіх

france <- fortify(gadm)                                            # перетворити полігон на датафрейм
str(france)

m <- ggplot() + geom_map(data = france,                            # подложка
                         aes(map_id = id), # групування точок по id, щоб розуміло що це одна країна
                         map = france,
                         fill = "white", color = "black") +        # фон білий, лінії чорні
  expand_limits(x = france$long, y = france$lat) +                 # 
  coord_map("mercator") +                                          # стандарт координат для цієї карти/країни
  xlab("Lon") + ylab("Lat") + theme_bw()                           # підписи осей, вибір теми
m

# додати на карту точки 
mp <- m + geom_point(data = data.frame(Lat = c(45.0, 47.2,45.5, 48.5),  # в датафреймі вказана широта та довгота
                                       Lon = c(0, 3.0, 7.0, -2.0)),
                     aes(Lon, Lat), color = I("red"), size = 3)    # вказали, що вважати широтою та довготою
mp                                                                 # якщо в size задати вектор, то точки будуть різного розміру


#==============================================

# приклади використання з кусками коду
# http://jkunst.com/highcharter/highmaps.html 

# перелік готових подложек-карт, 
# https://code.highcharts.com/mapdata/


library(highcharter)        # картинка інтерактивна, по кліку може мінятись

hcmap("countries/fr/custom/fr-all-mainland", showInLegend = FALSE) %>%   # подложка-карта
  hc_add_series(data = data.frame(name = c(1,2,3,4),                     # додати точки
                                  lat = c(45.0, 47.2,45.5, 48.5),        # їх координати
                                  lon = c(0, 3.0, 7.0, -2.0),           
                                  z=c(1,1,1,1)), type = "mapbubble", name = "Data",maxSize="1%") %>%   # розмір та вид точок
  hc_mapNavigation(enabled = TRUE)                                       # можливість міняти картинку



library(geosphere)          # функції для перетворення координат та розрахунку відстаней в різних системах координат

# відстань в метрах
distHaversine(c(0, 45.0), c(-2, 47.2))              # відстань по сфері (земна куля як правильна сфера)
distVincentyEllipsoid(c(0, 45.0), c(-2, 47.2))      # відстань по еліпсу (враховує кривизну поверхні землі) 

# задали координати
coords <- cbind(c(0, 3.0, 7.0, -2.0),
                c(45.0, 47.2,45.5, 48.5))
coords   # подивились координати

# порахувати відстань між 4 точками попарно
Dist <- apply(coords, 1, 
              FUN = function(eachPoint) distHaversine(eachPoint, coords)/1000)
Dist

Dist <- Dist[lower.tri(Dist)]
Dist

mean(Dist) # середня відстань

