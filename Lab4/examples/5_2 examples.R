################################################
##             ПРОСТОРОВИЙ РОЗПОДІЛ           ##
##                ОБ'ЄКТІВ, ч.1               ##
################################################
# 
# library(spatstat) # для роботи з просторовим розподілом об'єктів (польові дослідження з мітками на карті)
# 
# Функція створення моделі 
# ppp()
# 
# Функція розбиття на квадрати
# quadrats()
# 
# Гіпотеза про значимість імовірності влучення у квадрат 
# quadrat.test()
# 
# Побудова карт щільностей
# density.ppp()
# 
# Функція для перегляду критичних областей
# relrisk()
# 
################################################

#install.packages("spatstat", dep=T)
library(spatstat)  # координати можуть бути географічні або абстрактні

setwd("~/data/Maps")
plotBl <- read.table(file = "bloss.txt", header = T, sep = "\t") # набір про квіти, кількість знайдених квітів та їх можливість розмножитись
str(plotBl)

# встановлення розмірів 
xmin <- floor(min(plotBl$x))     # округлити вниз
xmax <- ceiling(max(plotBl$x))   # округлити вгору
ymin <- floor(min(plotBl$y))
ymax <- ceiling(max(plotBl$y))

# створення ppp (point pattern): 
ppp_object <- ppp(x = plotBl$x, y = plotBl$y,                         # координати
                  marks = data.frame(age = plotBl$age,
                                     blossoms = plotBl$blossoms),     # marks     дані, що описує кожна точка
                  window = owin(c(xmin, xmax), c(ymin, ymax),         # window    форма області (owin прямокутна) та її координати
                                unitname = c("metre","metres")))      # одиниці вимірювання, ні на що не впливає, тільки для інформування дослідника

# площа області
A <- area.owin(ppp_object$window)
A
# кількість точок в області
N <- ppp_object$n
N 
# розмір однієї клітинки (квадрантів) в області
L <- sqrt(2*A/N) # правило "великого пальця" - оптимальне розбиття на квадранти за площею та кількістю точок
L

# побудувати об'єкти-квадранти області
quadnum <- round(10/L)
quadnum
ppp_quadrats <- quadrats(ppp_object, nx = quadnum, ny = quadnum)

# кількість точок в області
quadcount <- quadratcount(ppp_object, tess = ppp_quadrats)
quadcount <- round(quadcount, 1)

# показати кількість точок в області

# координати центру області
xgrid <- quadrats(ppp_object, nx = quadnum, ny = quadnum)$xgrid  # отримали координати сітки для розбиття
ygrid <- quadrats(ppp_object, nx = quadnum, ny = quadnum)$ygrid
image(xgrid, ygrid,
      t(quadcount[order(1:quadnum, decreasing = T), ]),          # беремо квадранти у зворотньому порядку, щоб відображення на картинці відповідало звичній системі кооринат, на картинці початок координат у лівому верхньому кутку
      col = colorRampPalette(c("white", "green"))(15),           # колір - градація від білого до зеленого, 15 рівнів
      axes=F, asp=1, xlab="", ylab="",                           # осі відсутні
      main="Number of points")
plot(quadcount, add=T, cex=0.7)                                  # додали дані та сітку, add=T - додати на останню картинку

# відобразити точки
plot(ppp_object, which.marks = "age",                            # місце точки - по значенню координат (age  as.factor!)
     chars = c(19, 24), cex = 0.7, add = T)                      # вигляд маркера - по значенню параметра age, два типи символів chars = c(19, 24), розмір cex = 0.7

# CSR test для імовірностей - чи пуассонівський розподіл в квадрантах, тоді в наступному році ми знайдемо рослини на тих же місцях
quadrat_test_result <- quadrat.test(ppp_object, nx = 9, ny = 9)
quadrat_test_result             # мало даних в кількох квадрантах, + закон не пуассонівський  p-value = 0.001064

# карта щільності для рослин першого типу age=="gene"
gene_ppp <- ppp_object[ppp_object$marks$age=="gene"]

# bandwidth by Silverman: метод розрахунку щільності розподілу по Сілверману 
sigma<-(sd(gene_ppp$x) + sd(gene_ppp$y))/2                      # розмір віконця для розрахунку щільності
iqr<-(IQR(gene_ppp$x) + IQR(gene_ppp$y))/2 
bandwidth <- 0.9*min(sigma, iqr)*gene_ppp$n^(-1/5)
bandwidth

gene_intensity <- density.ppp(gene_ppp, sigma = bandwidth)          # порахували щільність
plot(gene_intensity,   main = "Gene Intensity")
points(gene_ppp, pch = 19, cex = 0.6)

# визначити області з однаковими властивостями (де кількість квітів, що не можуть розмножуватись, критично велике)

for_relrisk_example <- ppp_object
marks(for_relrisk_example) <- ppp_object$marks$age             # цікавить тільки вік рослин

# імовірність age pre:
sigma<-0.5                                                     # критичний порог
p <- relrisk(for_relrisk_example,sigma) 
plot(p, main = "Probability of pre", col = colorRampPalette(
  c("antiquewhite", "aquamarine3","navyblue"))(100))           # кольорова гамма з 100 відтінками

# додати контурні лінії
contour(p, nlevels = 5, lwd = seq(from = 0.1, to = 3, length.out = 5), add = T)
# кількість рівнів # товщина контурних ліній

