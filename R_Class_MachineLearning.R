#logistic regression with glm() function from stats default package
#support vector machines (SVM) with e1071 package
#random forest with randomForest package

#We will begin with Logistic Regression
mydata <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv") 
dim(mydata)

#glm stands for generalized linear model.
#Note that currently the column admit in mydata is of integer type. We encode admission with 1 and rejection with 0.
class(mydata$admit) #"integer"
mydata$admit <- factor(mydata$admit, levels = c(0, 1), labels = c("admitted", "rejected"))

# We also need to convert the rank column to a factor format.
mydata$rank <- factor(mydata$rank) 
head(mydata)

#We will now fit the model
fit.logit <- glm(admit ~ gre + gpa + rank, data = mydata, family = "binomial")

summary(fit.logit) #Model summary
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.6268  -0.8662  -0.6388   1.1490   2.0790  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -3.989979   1.139951  -3.500 0.000465 ***
#   gre          0.002264   0.001094   2.070 0.038465 *  
#   gpa          0.804038   0.331819   2.423 0.015388 *  
#   rank2       -0.675443   0.316490  -2.134 0.032829 *  
#   rank3       -1.340204   0.345306  -3.881 0.000104 ***
#   rank4       -1.551464   0.417832  -3.713 0.000205 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 499.98  on 399  degrees of freedom
# Residual deviance: 458.52  on 394  degrees of freedom
# AIC: 470.52
# 
# Number of Fisher Scoring iterations: 4

coef(fit.logit)
# (Intercept)          gre          gpa        rank2        rank3        rank4 
#-3.989979073  0.002264426  0.804037549 -0.675442928 -1.340203916 -1.551463677 

#You can get the confidence intervals for the coefficients with the confint() fuinction
confint(fit.logit)
# 2.5 %       97.5 %
#   (Intercept) -6.2716202334 -1.792547080
# gre          0.0001375921  0.004435874
# gpa          0.1602959439  1.464142727
# rank2       -1.3008888002 -0.056745722
# rank3       -2.0276713127 -0.670372346
# rank4       -2.4000265384 -0.753542605

#Support Vector Machine Example -- we will use a simple simulation
set.seed(1)
x <- matrix(rnorm(40*2), ncol=2)
y <- c(rep(-1,20), rep(1,20))
x[y == 1, ] <- x[y == 1, ] + 2
dat <- data.frame(x = x, y=as.factor(y)) 
head(dat)
plot(x[, 2], x[, 1], col=(3-y))

#We will use the e1071 library to run an SVM model
library(e1071)
# Set scale to be FALSE otherwise by default x is scaled to zero mean and unit variance 
svmfit <- svm(y ~ ., data=dat, kernel="linear", cost = 10, scale=FALSE)
summary(svmfit)

svmfit$index
##[1] 2 51415

svmfit <- svm(y~., data=dat, kernel="linear", cost=0.05, scale=FALSE) 
svmfit$index
## [1] 1 2 3 4 5 6 7 8 910121314151617181920
summary(svmfit)
plot(svmfit, dat)
# 
# Call:
#   svm(formula = y ~ ., data = dat, kernel = "linear", cost = 0.05, scale = FALSE)
# 
# 
# Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  linear 
# cost:  0.05 
# gamma:  0.5 
# 
# Number of Support Vectors:  23
# 
# ( 12 11 )
# 
# 
# Number of Classes:  2 
# 
# Levels: 
#   -1 1

plot(svmfit, dat)

set.seed(1)
tune.out <- tune(svm,y ~ ., data=dat, kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.05, 0.1, 1,5,10,20)))
summary(tune.out)

# Parameter tuning of ‘svm’:
#   
#   - sampling method: 10-fold cross validation 
# 
# - best parameters:
#   cost
# 0.05
# 
# - best performance: 0.1 
# 
# - Detailed performance results:
#   cost error dispersion
# 1 1e-03 0.625  0.2946278
# 2 1e-02 0.450  0.2581989
# 3 5e-02 0.100  0.1290994
# 4 1e-01 0.125  0.1767767
# 5 1e+00 0.150  0.1748015
# 6 5e+00 0.125  0.1317616
# 7 1e+01 0.125  0.1317616
# 8 2e+01 0.125  0.1317616

bestmod <- tune.out$best.model 
plot(bestmod, dat)

#Create a test dataset, as we did for the train dataset
set.seed(1)
xtest <- matrix(rnorm(30*2), ncol=2)
ytest <- sample(c(-1,1), 30, rep=TRUE)
xtest[ytest == 1, ] <- xtest[ytest == 1,] + 2
testdat <- data.frame(x = xtest, y = as.factor(ytest)) 
plot(xtest[, 2], xtest[, 1], col=(3-ytest))

#The results of the tuned model
ypred <- predict(bestmod, testdat) 
table(predict = ypred, truth = testdat$y)
## truth 
## predict -1 1 
## -1 13 3 
## 1 1 3

#And here is the non-tuned model results
svmfit <- svm(y~., data=dat, kernel = "linear", cost=10, scale = FALSE) 
ypred <- predict(svmfit, testdat)
table(predict = ypred, truth = testdat$y)
## truth 
## predict -1 1 
## -1 10 2 
## 1 4 4

#Now we will use a kernel SVM

set.seed(1)
# Generate 200 points
x <- matrix(rnorm(400*2), ncol=2) x[1:100,] <- x[1:100,] + 2
x[101:200,] <- x[101:200, ] - 2
y <- c(rep(1,200), rep(2,200))
dat <- data.frame(x = x, y = as.factor(y)) 
# Let a random half be a training set 
trainIdx <- sample(400, 200)
plot(x[trainIdx, 2], x[trainIdx, 1], col=y[trainIdx])

#We will use the SVM with a radial kernel. Note that here we can additionally specify the gamma parameter for the radial function
svmfit <- svm(y~., data=dat[trainIdx,], kernel = "radial", gamma=1, cost=1) 
table(true = dat[-trainIdx,"y"], pred=predict(svmfit, newdata = dat[-trainIdx, ]))
## pred 
##true 1 2 
## 1 81 16 
## 2 11 92

plot(svmfit, dat[trainIdx,])

#Tune gamma and cost parameters
tune.out <- tune(svm, y~., data=dat[trainIdx,], kernel="radial", ranges=list(cost=seq(0.01, 15, length.out = 10), gamma=seq(0.01, 5, length.out = 5)))
table(true = dat[-trainIdx,"y"], pred = predict(tune.out$best.model, newdata = dat[-trainIdx,]))

plot(tune.out$best.model, dat[trainIdx,])

#Random forest -- Can be used to perform classification AND regression
#Random Forest: Averaging over a collection of decision trees makes the predictions more stable.
#Introduction of randomness into the candidate splitting variables, reduces correlation between the generated trees.
#At each splitting level, the pool of the potential splitting candidate variables does not contain all variables, but only a random subset of them.

#The following in important in the randomForest package
#mtry – Number of variables randomly sampled as candidates at each split.
#ntree– Number of trees generated (to be averaged over).
library(randomForest) 
data(iris)
head(iris)

set.seed(71)
iris.rf <- randomForest(Species ~ ., data=iris, importance=TRUE,proximity=TRUE)
print(iris.rf)

# Call:
#   randomForest(formula = Species ~ ., data = iris, importance = TRUE,      proximity = TRUE) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 2
# 
# OOB estimate of  error rate: 5.33%
# Confusion matrix:
#   setosa versicolor virginica class.error
# setosa         50          0         0        0.00
# versicolor      0         46         4        0.08
# virginica       0          4        46        0.08
## Look at variable importance: 

importance(iris.rf)
# setosa versicolor virginica MeanDecreaseAccuracy MeanDecreaseGini
# Sepal.Length  6.043093   7.852142  7.929416            11.511553         8.773160
# Sepal.Width   4.401772   1.028243  5.437693             5.395909         2.191725
# Petal.Length 21.755707  31.332148 29.641035            32.944251        42.538907
# Petal.Width  22.839777  32.672927 31.679054            34.498646        45.774956

varImpPlot(iris.rf)

#MOVE ONTO UNSUPERVISED LEARNING

#A task of inferring the patterns and the latent (hidden) structure of the data on its own.
#The goal is to understand the relationships between features or among observations.
#There are no response or output variables such as labels. You are not interested in predciting any specific quantity.
#Only X is available but no Y .

#Approaches/techniques:
      #clustering e.g. k-means, hierarchical, gaussian mixture models
      #dimensional reduction, manifold learning e.g. PCA, MDS, SVD, Isomaps anomaly detection

#Example: PCA of the US crime rates data
head(USArrests)
states <- row.names(USArrests)
apply(USArrests, 2, mean)
apply(USArrests, 2, var)

#In R you can use the function prcomp() to do PCA. Check the help page ?prcomp for details.
#prcomp() is a faster and preferred method over princomp(). It computes a singular value decomposition of the matrix.

pca.res <- prcomp(USArrests, scale=TRUE)
names(pca.res)
pca.res$center
pca.res$scale
pca.res$rotation
head(pca.res$x)

biplot(pca.res, scale=0, cex = 0.8)

#Scree Plot
#Choose the smallest number of PCs that explain a sizable amount of the variation in the data.
#Look for a point at which the proportion of variance explained by each subsequent PCs drops off. This is often referred to as an elbow in the
#scree plot.

# compute the eigenvalues
pr.var <- pca.res$sdev^2
# proportion of variance explained 
pve <- pr.var/sum(pr.var)

#k-means clustering
library(jpeg)
library(ggplot2)
setwd("/Users/Alex")
url <- "http://www.infohostels.com/immagini/news/2179.jpg"
# Download the file and save it as "Image.jpg" in the directory
dFile <- download.file(url, "test.jpg")
img <- readJPEG("test.jpg") # Read the image 
(imgDm <- dim(img))

#Convert the 3D array to a data frame.
#Each row of the data frame should correspond a single pixel.
#The columns should include x and y – the pixel location, and R, G, B – the pixel intensity in red, green, and blue.

imgRGB <- data.frame( x = rep(1:imgDm[2], each = imgDm[1]), y = rep(imgDm[1]:1, imgDm[2]), R = as.vector(img[,,1]), G = as.vector(img[,,2]), B = as.vector(img[,,3]))

#Each pixel is a datapoint in 3D specifying the intensity in each of the three “R”, “G”, “B” channels, which thetermin the pixel’s color.
#We use k-means to cluster the pixels into color groups. k will be the number of color clusters the algorithm uses.
#k-means can be performed in R with kmeans() built-in function.
set.seed(1)
kClusters_2 <- 2
kMeans_2 <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters_2) 
names(kMeans_2)

kMeans_2$centers

rgb(kMeans_2$centers)

kMeans_2$cluster[1:20]
kColours_2 <- rgb(kMeans_2$centers[kMeans_2$cluster,]) 
head(kColours_2)

ggplot2(data = imgRGB, aes(x = x, y = y)) + geom_point(colour = kColours_2) + labs(title = paste("k-Means Clustering of", kClusters_2, "Colours")) + xlab("x") + ylab("y") + theme_bw()

#Now add more colors. Increase the number of clusters to 6:
set.seed(1)
kClusters_6 <- 6
kMeans_6 <- kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters_6) 
kColours_6 <- rgb(kMeans_6$centers[kMeans_6$cluster,])

ggplot2(data = imgRGB, aes(x = x, y = y)) +
  geom_point(colour = kColours_6) +
  labs(title = paste("k-Means Clustering of", kClusters_6, "Colours")) + xlab("x") + ylab("y") + theme_bw()
