## Data Science Project Regression Modeling : California Housing Price Prediction Overview <br>
* Build a machine learning regression model to estimate and predict housing price located in california region.
* Engineered some feaures to better specify how many rooms, bedrooms per rooms, bedrooms per household and population per household.
* Exploratory data with seaborn and matplotlib help giving better understanding of which specific area having the heighest or lowest house price and which are the most densely populated.
* One hot encoding and standard scaler are used to pre process dataaset before implemented it to machine learning model.
* Linear regression, Support Vector regression and Tree based and Gradient boosting regressor model are used alongside gridsearch cv to find and optimize the best best model.

### Code and Resource Used
* Packages : pandas, numpy, matplotlib, seaborn, sci-kit learn

### Exploratory Data Analysis
* Numerical features distribution 

* Income Stratified Distribution

* Population

### Data Pre Processing Before Building and Implement Model
* First to split between train and test dataset with a proportion of 80 % of train and 20 % of test.
* Define which features are numerical and categorical ones.
* Standardizing numerical features with standardscaler, while onehot encoder is used to process the categorical features of both train and test set using custom pieline transformation.

### Model Building
5 different models are used in which :
* **Linear Regression**
* **Decision Tree**
* **Random Forest**
* **Gradient BOosting Regressor**
* **Support Vector Regressor**

### Model Performance
> **Linear Regression** RMSE : 68777.12503644277
> **Linear Regression** Validation RMSE : 68917.9102919516

> **Decision Tree** RMSE : 0.0
> **Decision Tree** Validation RMSE : 67516.34954441145

> **Random Forest** RMSE : 18141.62284949507
> **Random Forest** Validation RMSE : 68917.9102919516

> **Support Vector** RMSE : 118096.46334907328
> **Support Vector** Validation RMSE : 118104.00396212477

> **Gradient Boosting** RMSE : 118096.46334907328
> **Gradient Boosting** Validation RMSE : 55200.46007555312

Linear regression, deciison tree and random forest do not perform very well for this particular dataset all of them exibit a certain degree of overfitting, especially decision tree. <br>
Gradient boosting regressor far outperformed the other 4 models.

### Hyperparamneter Tuning
* Best Model after gridsearch hyperparameter tuning
> GradientBoostingRegressor(max_depth=5, n_estimators=1000, random_state=50)

### Model Performance on Test Set
>Final Model RMSE : 47682.741973582626 <br>
 Final Model MAE : 31395.550834475882










