library(readr)
library(caret)
library(e1071)
library(C50)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(lattice)
library(mlbench)
library(gridExtra)
library(grid)
library(gapminder)
#------------------> LOAD  FILE ----------------------------------------
NewData <- read.csv("file:///C:/Users/User/Desktop/Ubiqum/Task6.2/UJIndoorLoc/trainingData.csv")
ncol(NewData)



View(NewData)
#----------- > Preprocess Data --------------------
NewData$FLOOR = as.factor(NewData$FLOOR)
NewData$LATITUDE = as.numeric(NewData$LATITUDE)
NewData$BUILDINGID =as.factor(NewData$BUILDINGID)
NewData$SPACEID=as.factor(NewData$SPACEID)
NewData$RELATIVEPOSITION=as.factor(NewData$RELATIVEPOSITION)
NewData$USERID=as.factor(NewData$USERID)
NewData$PHONEID=as.factor(NewData$PHONEID)

MyNewData<-NewData  
#+++++++++++++++++++++++++++++++My sample set++++++++++++++++++++++++++

for(i in 1:520)
{
  MyNewData[which(MyNewData[,i]< -95), i]=-110
  MyNewData[which(MyNewData[,i]==100), i]=-110

}
View(MyNewData)

MyNewData2<-function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

Sdata1<-as.data.frame(lapply(MyNewData[1:520],MyNewData2))
View(Sdata1)

Sdata1$LONGITUDE<-as.numeric(NewData$LONGITUDE)
Sdata1$LATITUDE <-as.numeric(NewData$LATITUDE)
Sdata1$FLOOR <- as.factor(NewData$FLOOR)
Sdata1$BUILDINGID<-as.factor(NewData$BUILDINGID)
Sdata1$SPACEID=as.factor(NewData$SPACEID)
Sdata1$RELATIVEPOSITION=as.factor(NewData$RELATIVEPOSITION)
Sdata1$USERID=as.factor(NewData$USERID)
Sdata1$PHONEID=as.factor(NewData$PHONEID)
Sdata1$TIMESTAMP<-NewData$TIMESTAMP
#TIMESTAMP= -0  
#PHONEID = -1
#USERID = -2
#RELATIVEPOSITION = -3
#SPACEID = -4
#BUILDINGID = -5
#FLOOR = -6
#LATITUDE = -7
#LONGITUDE = -8
#WAP520 = -9
#removing NaN
Sdata1<-data.frame(Sdata1)
Sdata1[is.na(Sdata1)]<-0

sampleIndex<- sample(1:nrow(Sdata1), 1000)
Sdata1<-Sdata1[sampleIndex,]
#Sdata1<- Sdata[,!apply(Sdata==-110, 2, all)]

MyNewData2<-Sdata1

View(MyNewData2)
#Remove all culumns with only 100
#Sdata1<- Sdata1[, !apply(Sdata1==-110, 2, all)]
#+++++++++++++++++++++++++Creating my TRaining/Testing Set+++++++++++++++++++++
set.seed(123)
sTraining<- createDataPartition(Sdata1$BUILDINGID, p=0.75, list=FALSE)
Training<-Sdata1[sTraining,]
Testing<- Sdata1[-sTraining,]

View(Sdata1)

#10 folds cross validations.
control_M<-trainControl(method = "repeatedcv", number = 10, repeats = 3)

#+++++++++++++++++++=+++++++++++BLD Models+++++++++++++++++++++++++++++++++++++
fitknn2<-train(BUILDINGID~., data=Training[,c(1:(ncol(Sdata1)-9), (ncol(Sdata1)-5))], method="knn", trControl=control_M, preProcess=c("center", "scale"))

fitsmv2<- train(BUILDINGID~., data=Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-5))], method="svmLinear", preProcess = c("center","scale"))

fitc5.2<-train(BUILDINGID~., data=Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-5))], method="C5.0", trContrel=control_M, preProcess=c("center","scale"))

fitrf<-train(BUILDINGID~., data=Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-5))], method="rf", trContrel=control_M, preProcess=c("center","scale"))
#++++++++++++++++++++++++++++++++++++++++++UMMARY++++++++++++++++++++++++++++++++++++++++++++++
resultBLD<-resamples(list(RF=fitrf,SMV=fitsmv2,c5.0=fitc5.2))
summary(resultBLD)

dotplot(resultBLD, metric = "ROC")
bwplot(resultBLD, layout(c(3,1)))
#+++++++++++++++++++++++++++++++++LATITUDE++++++++++++++++++++++++++++++++++++++++++++++

knnLat2<- train(LATITUDE~., data=Training[,c(1:(ncol(Sdata1)-9), (ncol(Sdata1)-7),(ncol(Sdata1)-5))], method="knn",trControl=control_M, preProcess = c("center","scale"))

smvLat2<- train(LATITUDE~., data = Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-7),(ncol(Sdata1)-5))], method="svmLinear", trControl=control_M, preProcess = c("center","scale"))

RFlat2 <- train(LATITUDE ~ ., data = Training[, c(1:(ncol(Sdata1)-9), (ncol(Sdata1)-5), (ncol(Sdata1)-7))],  method = "rf", trControl=control_M, preProcess = c("center","scale"))

resultLAT<-resamples(list(knn=knnLat2,smv=smvLat2,RF=RFlat2))
summary(resultLAT)
#+++++++++++++++++++++++++++++++++LONGITUDE++++++++++++++++++++++++++++++++++++++++++++

knnLong2<- train(LONGITUDE~., data = Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-8),(ncol(Sdata1)-5))], method="knn", preProcess = c("center","scale"))

smvLong2<- train(LONGITUDE~., data=Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-8),(ncol(Sdata1)-5))], method="svmLinear",trControl=control_M, preProcess = c("center","scale"))

RFlong2 <- train(LONGITUDE ~ ., data = Training[, c(1:(ncol(Sdata1)-9), (ncol(Sdata1)-5), (ncol(Sdata1)-8))],  method = "rf", trControl=control_M, preProcess = c("center","scale"))

resultLONG<-resamples(list(smv=smvLong2,RF=RFlong2))
summary(resultLONG)
knnLong2
#+++++++++++++++++++++++++++++++++++++FLOOR++++++++++++++++++++++++++++++++++++++++++++++
knnFloor2<- train(FLOOR~., method="knn", data=Training[,c(1:(ncol(Sdata1)-5),(ncol(Sdata1)-6), (ncol(Sdata1)-9))], trControl=control_M, preProcess = c("center","scale"))

smvFloor2<- train(FLOOR~., data=Training[,c(1:(ncol(Sdata1)-9),(ncol(Sdata1)-6), (ncol(Sdata1)-5))], method="svmLinear",trControl=control_M,  preProcess = c("center","scale"))

rfFloor2<-train(FLOOR~., data=Training[,c(1:(ncol(Sdata1)-9), (ncol(Sdata1)-6),(ncol(Sdata1)-5))], method="rf", trControl=control_M, preProcess=c("center","scale") )

C5Floor2<- train(FLOOR~., data=Training[,c(1:(ncol(Sdata1)-9), (ncol(Sdata1)-6),(ncol(Sdata1)-5))], method="C5.0", trControl=control_M, preProcess=c("center", "scale"))

resultFLOOR<-resamples(list(knn=knnFloor2,sm=smvFloor2, rf=rfFloor2,c50=C5Floor2))
summary(resultFLOOR)


############################################################PREDICTION############################################

BuildingPred2<-predict(fitknn2, Testing)
BuildingPred2
postResample(BuildingPred2, Testing$BUILDINGID)

table(BuildingPred2, Testing$BUILDINGID)
PreBLD<-BuildingPred2==Testing$BUILDINGID
table(PreBLD)
prop.table(table(PreBLD))

Buildsmv2<-predict(fitsmv2, Testing)
Buildsmv2
postResample(Buildsmv2, Testing$BUILDINGID)

table(Buildsmv2, Testing$BUILDINGID)
PreBLDsmv<-Buildsmv2==Testing$BUILDINGID
table(PreBLDsmv)
prop.table(table(PreBLDsmv))

BuldC5.02<-predict(fitc5.2, Testing)
BuldC5.02
postResample(BuldC5.02, Testing$BUILDINGID)

table(BuldC5.02, Testing$BUILDINGID)
PredBLDc5.0<-BuldC5.02==Testing$BUILDINGID
table(PredBLDc5.0)
prop.table(table(BuldC5.02))
################
BuldRf<-predict(fitrf, Testing)
BuldRf
postResample(BuldRf, Testing$BUILDINGID)

table(BuldRf, Testing$BUILDINGID)
PredBLDrf<-BuldRf==Testing$BUILDINGID
table(PredBLDrf)
prop.table(table(PredBLDrf))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++LATITUDE++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LATPred2<-predict(RFlat2,Testing)
LATPred2
postResample(LATPred2, Testing$LATITUDE)


LATPredi2<-predict(knnLat2, Testing)
LATPredi2
postResample(LATPredi2, Testing$LATITUDE)

LATsmvLat2<-predict(smvLat2, Testing)
LATsmvLat2
postResample(LATsmvLat2, Testing$LATITUDE)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++LONGITUDE+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LONGPredRf2<-predict(RFlong2, Testing)
LONGPredRf2
postResample(LONGPredRf2, Testing$LONGITUDE)

LONPredknn2<-predict(knnLong2, Testing)
LONPredknn2
postResample(LONPredknn2, Testing$LONGITUDE)

LongPredsmv2<-predict(smvLong2, Testing)
LongPredsmv2
postResample(LongPredsmv2, Testing$LONGITUDE)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++FLOOR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FloorPredsmv2<-predict(smvFloor2, Testing)
FloorPredsmv2
postResample(FloorPredsmv2, Testing$FLOOR)


table(FloorPredsmv2, Testing$FLOOR)
predsmvFLR<-FloorPredsmv2==Testing$FLOOR
table(predsmvFLR)
prop.table(table(predsmvFLR))

PredFloorc5.2<-predict(C5Floor2,Testing)
PredFloorc5.2
postResample(PredFloorc5.2, Testing$FLOOR)

table(PredFloorc5.2, Testing$FLOOR)
Predc5.Flr<-PredFloorc5.2==Testing$FLOOR
table(Predc5.Flr)
prop.table(table(Predc5.Flr))

preFloor2rf<-predict(rfFloor2, Testing)
preFloor2rf
postResample(preFloor2rf, Testing$FLOOR)

table(preFloor2rf, Testing$FLOOR)
predFlrRf<-preFloor2rf==Testing$FLOOR
table(predFlrRf)
prop.table(table(predFlrRf))

PredKnnFl2<-predict(knnFloor2, Testing) #NOT WORKING
PredKnnFl2
postResample(PredKnnFl2, Testing$FLOOR)
############################################Calculating the diff between LAT/LONG#########################
check2<-as.data.frame(Testing$LATITUDE)
check2$LONGITUDE<-as.data.frame(Testing$LONGITUDE)
LatPredictions<-as.data.frame(Testing$LATITUDE)
LongPredictions<-as.data.frame(Testing$LONGITUDE)

#========================LATITUDE PREDICTIONS===============================
LatPredictionknn<-predict(knnLat2, Testing)
LatPredictions$knn<-Testing$LATITUDE - (predict(knnLat2, Testing))

LatPredictionSvm<-predict(smvLat2, Testing)
LatPredictions$svm<-Testing$LATITUDE -(predict(smvLat2, Testing))

LatPredictionRf<-predict(RFlat2, Testing)
LatPredictions$RF<-Testing$LATITUDE -(predict(RFlat2, Testing))
#----------------------------------LONGITUDE Predictions------------------------------------
LongPredictknn<-predict(knnLong2, Testing)
LongPredictions$knn<-Testing$LONGITUDE -(predict(knnLong2, Testing))

LongPredictSvm<-predict(smvLong2,Testing)
LongPredictions$Svm<-Testing$LONGITUDE -(predict(smvLong2, Testing))

LongPredictRF<-predict(RFlong2, Testing)
LongPredictions$RF<-Testing$LONGITUDE -(predict(RFlong2, Testing))

#++++++++++++++++++++++++++++Calculate LAT/LONG Error++++++++++++++++++++++++++++++++++++++++
check2$knnError<-as.data.frame(sqrt((LongPredictions$knn)^2 +(LatPredictions$knn)^2))
check2$SvmError<-as.data.frame(sqrt((LongPredictions$Svm)^2 +(LatPredictions$svm)^2))
check2$RfError<-as.data.frame(sqrt((LongPredictions$RF)^2 +(LatPredictions$RF)^2))

colnames(check2)<-c("LAT", "LONG", "KnnError", "SvmError", "RfError")
summary(check2)
View(check2)


plot(check2$RfError)

####################################################PLOTTING DATA############################################
knnPred2=0
knnPred2<-as.data.frame(knnPred2)
knnPred2$Long<-as.data.frame(LongPredictknn)
knnPred2$Lat<-as.data.frame(LatPredictionknn)
View(knnPred2)

svmPred2="0"
svmPred2<-as.data.frame(svmPred2)
svmPred2$Long<-as.data.frame(LongPredictSvm)
svmPred2$Lat<-as.data.frame(LatPredictionSvm)
View(svmPred2)

rfPred2=0
rfPred2<-as.data.frame(rfPred2)
rfPred2$Long<-as.data.frame(LongPredictRF)
rfPred2$Lat<-as.data.frame(LatPredictionRf)
View(rfPred2)
#+++++++++++++++++++++++++++++++++++++++++++PLOTTING++++++++++++++++++++++++++++++++++++
knn2= ggplot()+
  geom_point(data=Testing, aes(LATITUDE,LONGITUDE),color="black")+
  geom_point(data=knnPred2, aes(LatPredictionknn, LongPredictknn), color="green")+
  xlab("Latitude")+ylab("Longitude")+labs(title="KNN vs TestingDataSet")

SVM2= ggplot()+
  geom_point(data=Testing, aes(LATITUDE, LONGITUDE), color="black")+
  geom_point(data=svmPred2, aes(LatPredictionSvm,LongPredictSvm),color="green")+
  xlab("Latitude")+ylab("Longitude")+labs(title="SVM vs TestingSet")

rf2=ggplot()+
  geom_point(data = Testing, aes(LATITUDE, LONGITUDE), color="black")+
  geom_point(data=rfPred2, aes(LatPredictionRf,LongPredictRF),color="green")+
  xlab("Latitude")+ylab("Longitude")+labs(title="RF vs TestingSet")

########################################################################################################
no_100<-MyNewData
no_100$TIMESTAMP<-NULL
no_100$PHONEID <-NULL
no_100$USERID <-NULL
no_100$RELATIVEPOSITION <-NULL
no_100$SPACEID <-NULL
#no_100$BUILDINGID <-NULL
no_100$FLOOR <-NULL
no_100$LATITUDE <-NULL
no_100$LONGITUDE <-NULL
View(MyNewData)

ggplot()+geom_bar(aes(x=apply(no_100,1, function(x)length(which(x!=-110)))))+xlab("Frequency")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++Waps++++++++++++++++++++++++++++
Waps<-gather(no_100, "wapss", "Signal", 1:520)
#colnames(Waps)<-c("waps", "freq")
View(Waps)

no_100s<-filter(Waps, Waps$Signal!=-110)

Signal<-as.numeric(no_100s$Signal)

hist(Signal)

























