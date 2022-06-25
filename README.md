## Data Science Project Regression Modeling : California Housing Price Prediction Overview <br>
* Build a machine learning regression model to estimate and predict housing price located in california region.
* Engineered some feaures to better specify how many rooms, bedrooms per rooms, bedrooms per household and population per household.
* Exploratory data with seaborn and matplotlib help giving better understanding of which specific area having the heighest or lowest house price and which are the most densely populated.
* One hot encoding and standard scaler are used to pre process dataaset before implemented it to machine learning model.
* Linear regression, Support Vector regression and Tree based and Gradient boosting regressor model are used alongside gridsearch cv to find and optimize the best best model.

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/intro.png) 

Property valuation is an imprecise science. Individual appraisers and valuers bring their own experience, metrics and skills to a job. Consistency is difficult, with UK and Australian-based studies suggesting valuations between two professionals can differ by up to 40%. Perhaps a well-trained machine could perform this task in place of a human, with greater consistency and accuracy.
Let’s prototype this idea and train some ML models using data about a house’s features, costs and neighbourhood profile to predict its value. Our target variable the houses's  price is numerical, hence the ML task is regression. (For a categorical target, the task becomes classification.).
We’ll use a dataset from california housing data which simulates a portfolio of 20.640 properties, there are 26 columns.
But first we may need to define the business question.<br>
 
 ### Business Questions:
 * Which factor(s) contributes when it comes to housing prices? 
 * If there are indeed factors contributing to houses price, how well is it?
 * Which decision step to be taken moving forward?

Now we move on the data modeling / data processing steps: 

### The Steps for tihs modeling includes:
 * **EDA & Pre processing**: Data exploring. visualizing and cleaning
 * **Model training**: we’ll train and tune some tried-and-true classification algorithms, such as ridge and lasso regression.
 * **Performance evaluation**: we’ll look at common regression task metrics like the R²-score and mean squared average.
 * **Business Action**: Decision driven data needed to be taken to action to improve business.

### Data Exploration and Pre Processing
Exploratory data analysis (EDA) helps us understand the data and provides ideas and insights for data cleaning and feature engineering. Data cleaning prepares the data for our algorithms while feature engineering is the magic sauce that will really help our algorithms draw out the underlying patterns from the dataset.

### Code and Resource Used
* Packages : pandas, numpy, matplotlib, seaborn, sci-kit learn

### Exploratory Data Analysis
* Numerical features distribution <br>

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) 

* Income Stratified Distribution <br>
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) 

* Population <br>
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index2.png)

### Data Pre Processing Before Building and Implement Model
* First to split between train and test dataset with a proportion of 80 % of train and 20 % of test.
* Define which features are numerical and categorical ones.
* Standardizing numerical features with standardscaler, while onehot encoder is used to process the categorical features of both train and test set using custom pieline transformation.

### Model Building
5 different models are used in which :
* **Linear Regression**
* **Decision Tree**
* **Random Forest**
* **Gradient Boosting Regressor**
* **Support Vector Regressor**

### Model Performance
> **Linear Regression** RMSE : 68777.12503644277 <br>
> **Linear Regression** Validation RMSE : 68917.9102919516

> **Decision Tree** RMSE : 0.0 <br>
> **Decision Tree** Validation RMSE : 67516.34954441145

> **Random Forest** RMSE : 18141.62284949507 <br>
> **Random Forest** Validation RMSE : 68917.9102919516

> **Support Vector** RMSE : 118096.46334907328 <br>
> **Support Vector** Validation RMSE : 118104.00396212477

> **Gradient Boosting** RMSE : 118096.46334907328 <br>
> **Gradient Boosting** Validation RMSE : 55200.46007555312

Linear regression, deciison tree and random forest do not perform very well for this particular dataset all of them exibit a certain degree of overfitting, especially decision tree. <br>
Gradient boosting regressor far outperformed the other 4 models.

### Hyperparamneter Tuning
* Best Model after gridsearch hyperparameter tuning
> GradientBoostingRegressor(max_depth=5, n_estimators=1000, random_state=50)

### Model Performance on Test Set
We got the models performance increased after using gridsearch hyperparameter tuning. <br>
From 55200 to 47682
>Final Model RMSE : 47682.741973582626 <br>
 Final Model MAE : 31395.550834475882










