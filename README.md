## Data Science Project Regression Modeling : California Housing Price Prediction Overview <br>
* Build a machine learning regression model to estimate and predict housing price located in california region.
* Engineered some feaures to better specify how many rooms, bedrooms per rooms, bedrooms per household and population per household.
* Exploratory data with seaborn and matplotlib help giving better understanding of which specific area having the heighest or lowest house price and which are the most densely populated.
* One hot encoding and Robust scaler are used to pre process dataaset before implemented it to machine learning model.
* Several machine learning models are used alongside gridsearch cv to find and optimize the best best model.

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/intro.png) 

Property valuation is an imprecise science. Individual appraisers and valuers bring their own experience, metrics and skills to a job. Consistency is difficult, with UK and Australian-based studies suggesting valuations between two professionals can differ by up to 40%. Perhaps a well-trained machine could perform this task in place of a human, with greater consistency and accuracy.
Let’s prototype this idea and train some ML models using data about a house’s features, costs and neighbourhood profile to predict its value. Our target variable the houses's  price is numerical, hence the ML task is regression. (For a categorical target, the task becomes classification.).
We’ll use a dataset from california housing data which simulates a portfolio of 20.640 properties, there are 10 columns.
But first we may need to define the business question.<br>
 
 ### Business Questions:
 * Which factor(s) contributes when it comes to housing prices? 
 * If there are indeed factors contributing to houses price, how well is it?
 * Which decision step to be taken moving forward?

### Data Processing / Modeling Includes:
 * **EDA & Pre processing**: Data exploring. visualizing and cleaning
 * **Model training**: we’ll train and tune some tried-and-true classification algorithms.
 * **Performance evaluation**: we’ll look at common regression performance metrics like the RMSE and MAPE.
 * **Business Action**: Decision driven data needed to be taken to action to improve business.

### Code and Resource Used 
* Packages : pandas, numpy, matplotlib, seaborn, sci-kit learn

### Dataset Profiling

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) <br>
 * Dataset includes 20.640 observations and 10 columns with a total dimesion of 206.400
 
### Features Types

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) <br>
 * Ocean proximity seems the only categorical / non numeric ones in dataset. It refers to how close approximately the house from the sea / ocean.
 
 ### Missing Value CHecking and Handling
 
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) <br>
 * Total bedrooms feature has 204 missing values. To process this, we can either drop or replace the them with a median value. Here i prefer to replace it with median because we can still retain the precious data which otherwise we could not do if we droped the missing values. If done, then we corss check to see if ther'sstill missing values. <br>

### Descriptive Statistics 
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) <br>
 * Several features have a gap between its median and mean, possibly they willbe either right or left skedwed distribution. 

### Features DIstribution
* Numerical features distribution <br>
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) <br>
* Majority of the features are right skewed distribtuion, in the exception of housing age, latitude and longitude. This means, majority of them are in closest to their lower bound values, meanwhile some are in the distance or close to the higher bounds counterpart.

* Boxplot plot / Outilers Checking
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>
 * This bloxplot proves that the right skewed features have some outliers to some extent. Evidence why it is right skewed.

### Data Pre Processing and Model Implementation With Pycaret
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle. It is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only which makes experiments exponentially fast and efficient. Pycaret Setup interface.<br>
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>

### Implementing and COmparing Several Models Performances.
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>
 * An easier method to see and evalute which models fits our dataset best is to check their Root MEan Squared Errors (RMSE) and Mean Average Percentage Errors (MAPE) values. In short, RMSE represents how far / deviates the predicted error value relative to the real ones, meanwhile MAPE represent it in percentage. SO its best if we want to look out for the lowest of both RMSE and MAPE. Here we can see top 3 models have relatively good RMSE and MAPE. We will use these 3 models as base for now. 3 Models includes, Light Gradient Boosting Machine, Random Forest and Extra Trees Regeressor.

### Models Features Importances
Feature Importance refers to techniques that calculate a score for all the input features for a given model — the scores simply represent the “importance” of each feature. A higher score means that the specific feature will have a larger effect on the model that is being used to predict a certain variable.

 * LGBM <br>
 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>
 
 * Random Forest <br>
  ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>
  
 * Extra Trees <br>
 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>
 
  
### Models Features Interpretability
Interpretability is the degree to which a human can understand the cause of a decision. Another one is: Interpretability is the degree to which a human can consistently predict the model’s result . The higher the interpretability of a machine learning model, the easier it is for someone to comprehend why certain decisions or predictions have been made <br>
