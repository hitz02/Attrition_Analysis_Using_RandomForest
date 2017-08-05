library(data.table)
library(caret)
hr = fread(file = file.choose(),header=T,sep = ",")

str(hr)

summary(hr)

head(hr)

hr$Attrition = as.factor(hr$Attrition)
levels(hr$Attrition)

unique(hr$Over18) #As all the records are of age > 18, we can remove this field

unique(hr$EmployeeCount) #As all the records have count = 1, we can remove this field

unique(hr$StandardHours) #As all the records have hours = 80, we can remove this field

hr1 = subset(hr,select = -c(Over18,EmployeeCount,StandardHours))


#Checking missing values

apply(hr1,2,function(x) {sum(is.na(x))})

#No Missing values

summary(hr1)

class(hr1)

str(hr1)

hr1$Attrition = as.factor(hr1$Attrition)

## Let us find the variables Information Value
#install.packages("devtools")
library(devtools)
#install_github("riv","tomasgreif")
library(woe)

hr1 = data.frame(hr1)
iv.plot.summary(iv.mult(hr1[,!names(hr1) %in% c("EmployeeNumber")],
                        "Attrition",TRUE))

iv <- iv.mult(hr1[,!names(hr1) %in% c("EmployeeNumber")],
              "Attrition",TRUE)

iv

str(hr1)

#After looking at the information value summary, variables - PercentSalaryHike,PerformanceRating,Gender
#Education shows very weak strength, hence removing them from data set

hr2 = subset(hr1, select = -c(PercentSalaryHike,PerformanceRating,Gender,
                              Education))

hr2$BusinessTravel  = as.factor(hr2$BusinessTravel)
hr2$Department      = as.factor(hr2$Department)
hr2$EducationField  = as.factor(hr2$EducationField)
hr2$JobRole         = as.factor(hr2$JobRole)
hr2$MaritalStatus   = as.factor(hr2$MaritalStatus)
hr2$OverTime         = as.factor(hr2$OverTime)

levels(hr2$Attrition) = c(0,1)

str(hr2)

source("D:\\Class_Machine_Learning\\Rajesh Jakhotia/Visualization.R")
output_folder = "D:\\Class_Machine_Learning\\Rajesh Jakhotia\\VISUALIZATIONS1/"
Target_var_name = "Attrition"

col_list = colnames(hr2)[
  lapply(hr2, class) %in% c("numeric", "integer")
  ]
col_list
for (i in 1 : length(col_list)) {
  fn_biz_viz(df = hr2, target = Target_var_name, var = col_list[i])
}




table(hr2$Attrition)
# No  Yes 
# 2466  474 

#As we can see  there is class imbalance in the dataset , hence we need to oversample the data
library(ROSE)
hr2_over = ovun.sample(Attrition~.,
                            data = hr2, N = 4200,seed = 1, method = 'over')$data

prop.table(table(hr2_over$Attrition))
# 0         1 
# 0.5871429 0.4128571 

#Partitioning data for dev and holdout
library(caret)
set.seed(123)
ind = createDataPartition(hr2_over$Attrition,times = 1,p=0.7,list = F)

hr_1 = hr2_over[ind,]
hr_test = hr2_over[-ind,]

ind1 = createDataPartition(hr_1$Attrition,times = 1,p=0.6,list = F)

hr_train = hr_1[ind1,]
hr_val = hr_1[-ind1,]

table(hr_train$Attrition)
# 0     1 
# 1037  729 

table(hr_val$Attrition)
# 0   1 
# 690 485 

table(hr_test$Attrition)
# 0   1 
# 739 520

str(hr_train)

## Calling syntax to build the Random Forest
library(randomForest)

#taking mtry as approx sqrt(no. of var) and nodesize aprrox sqrt(population)
#EMployee no not taken for model
ARF <- randomForest(Attrition ~ ., data = hr_train[,-8], 
                   ntree=500, mtry = 5, nodesize = 100,
                   importance=TRUE)


#To check the optimum no. of trees to be built based on the error rate for OOB,class 0 and class 1
plot(ARF, main="")
legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest HR Attr")

ARF$err.rate

#From the plot, it seems 60 trees is the optimum

## List the importance of the variables.
impVar <- round(randomForest::importance(ARF), 2)
impVar[order(impVar[,3], decreasing=TRUE),]

tARF <- tuneRF(x = hr_train[,-c(2,8)], 
               y=hr_train$Attrition,
               mtryStart = 5, 
               ntreeTry=60, 
               stepFactor = 2, 
               improve = 0.0001, 
               trace=FALSE, 
               plot = FALSE,
               doBest = TRUE,
               nodesize = 100, 
               importance=TRUE
)


# 0.03107345 1e-04 
# -0.06997085 1e-04 
# -0.05830904 1e-04 

tARF$importance

## Scoring on dev
hr_train$predict.class <- predict(tARF, hr_train, type="class")
hr_train$predict.score <- predict(tARF, hr_train, type="prob")
head(hr_train)
class(hr_train$predict.score)

#Creating deciles based on prediction score for dev
hr_train$deciles = decile(hr_train$predict.score[,2])

#Creating rank ordering and computing ks for each decile for dev
library(data.table)
tmp_DT = data.table(hr_train)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition==1), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);


library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

#from the rank ordering, 1o deciles  are created and also max ks is 0.67 on dev

#rank_ordering(hr_train[,-c(2,8)],hr_train$Attrition,hr_train$predict.score[,2])

## Scoring on validation sample
hr_val$predict.class <- predict(tARF, hr_val, type="class")
hr_val$predict.score <- predict(tARF, hr_val, type="prob")
head(hr_val)
class(hr_val$predict.score)

#Creating deciles based on prediction score for val
hr_val$deciles = decile(hr_val$predict.score[,2])

#Creating rank ordering and computing ks for each decile for val
#levels(hr_test$Attrition) = c(0,1)

library(data.table)
tmp_DT = data.table(hr_val)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition==1), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);


library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

#########

## Scoring on validation sample
hr_test$predict.class <- predict(tARF, hr_test, type="class")
hr_test$predict.score <- predict(tARF, hr_test, type="prob")
head(hr_test)
class(hr_test$predict.score)

#Creating deciles based on prediction score for test
hr_test$deciles = decile(hr_test$predict.score[,2])

#Creating rank ordering and computing ks for each decile for test
#levels(hr_test$Attrition) = c(0,1)

library(data.table)
tmp_DT = data.table(hr_test)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition==1), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);


library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

################

#from the rank ordering, 1o deciles  are created and also max ks is 0.70 on dev

#Checking Accuracy with various parameter on dev
library(ROCR)
pred <- prediction(hr_train$predict.score[,2], hr_train$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS  #0.6929295

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc #0.9156174

## Gini Coefficient
install.packages('ineq')
library(ineq)
gini = ineq(hr_train$predict.score[,2], type="Gini")
gini #0.4243521

## Classification Error
with(hr_train, table(Attrition, predict.class))
#             predict.class
# Attrition   0   1
#          0 977  60
#          1 201 528

###########
#Checking Accuracy with various parameter on dev
library(ROCR)
pred <- prediction(hr_val$predict.score[,2], hr_val$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS #0.6310922

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc #0.8847363

## Gini Coefficient
#install.packages('ineq')
library(ineq)
gini = ineq(hr_val$predict.score[,2], type="Gini")
gini #0.4071038

## Classification Error
with(hr_val, table(Attrition, predict.class))
#             predict.class
# Attrition   0   1
#         0 646  44
#         1 170 315

########################
#Checking Accuracy with various parameter on dev
library(ROCR)
pred <- prediction(hr_test$predict.score[,2], hr_test$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS #0.6176017

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc #0.8809709

## Gini Coefficient
#install.packages('ineq')
library(ineq)
gini = ineq(hr_test$predict.score[,2], type="Gini")
gini #0.4098616

## Classification Error
with(hr_test, table(Attrition, predict.class))
# predict.class
# Attrition   0   1
# 0 671  68
# 1 174 346


