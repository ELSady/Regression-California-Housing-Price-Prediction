## Data Science Project Regression Modeling : California Housing Price Prediction Overview <br>
* Build a machine learning regression model to estimate and predict housing price located in california region.
* Engineered some feaures to better specify how many rooms, bedrooms per rooms, bedrooms per household and population per household.
* Features categorical encoding and Robust scaler are used to pre process dataaset before implemented it to machine learning model.
* Several machine learning models are used alongside gridsearch cv to find and optimize the best best model.

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/287558.jpg) 

Property valuation is an imprecise science. Individual appraisers and valuers bring their own experience, metrics and skills to a job. Consistency is difficult, with UK and Australian-based studies suggesting valuations between two professionals can differ by up to 40%. Perhaps a well-trained machine could perform this task in place of a human, with greater consistency and accuracy.
Let’s prototype this idea and train some ML models using data about a house’s features, costs and neighbourhood profile to predict its value. Our target variable the houses's  price is numerical, hence the ML task is regression. (For a categorical target, the task becomes classification.).
We’ll use a dataset from california housing data which simulates a portfolio of 20.640 properties, there are 10 columns.
But first we may need to define the business question.<br>
 
 ### Business Questions:
 * Which factor(s) contributes when it comes to housing prices? 
 * If there are indeed factors contributing to houses price, how well is it?
 * Which course of action to be taken moving forward?

### Data Processing / Modeling Includes:
 * **EDA & Pre processing**: Data exploring. visualizing and cleaning
 * **Model training**: we’ll train and tune some tried-and-true classification algorithms.
 * **Performance evaluation**: we’ll look at common regression performance metrics like the RMSE and MAPE.
 * **Business Action**: Decision driven data needed to be taken to action to improve business.

### Code and Resource Used 
* Packages : pandas, numpy, matplotlib, seaborn, sci-kit learn

### Dataset Profiling
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2006-40-28%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>

* Dataset includes 20.640 observations and 10 columns with a total dimesion of 206.400 <br>

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2007-18-30%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png)
 
### Features Types
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2007-18-49%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>
* Ocean proximity seems the only categorical / non numeric ones in dataset. It refers to how close approximately the house from the sea / ocean. <br>
 
 ### Missing Value Checking and Handling
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2007-19-10%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>

* Total bedrooms feature has 204 missing values. To process this, we can either drop or replace the them with a median value. Here i prefer to replace it with median because we can still retain the precious data which otherwise we could not do if we droped the missing values. If done, then we corss check to see if ther'sstill missing values. <br>

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2007-53-30%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>

* Data has been cleaned. <br>

### Descriptive Statistics 
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2009-38-56%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>
* Several features have a gap between its median and mean, possibly they will be either right or left skewed distribution.

### Features DIstribution
* Numerical features distribution <br>
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index.png) <br>

Insight we can get following the distribution plots include:
* Majority of the features are right skewed distribtuion, in the exception of housing age, latitude and longitude. This means, majority of them are in closest to their lower bound values, meanwhile some are in the distance or close to the higher bounds counterpart. <br>
* Several features have a gap between its median and mean, possibly they willbe either right or left skedwed distribution.
* Total population density predominantly are in the range 1000 to 2000.
* For households, predominantly there are approximately 300 to 500 in dataset.
* Meanwhile the income predominantly around at the range of 3k per year.

* Boxplot plot / Outilers Checking <br>
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index1.png) <br>

* This bloxplot proves that the right skewed features have some outliers to some extent. Evidence why it is right skewed. <br>

### Data Pre Processing and Model Implementation With Pycaret
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle. It is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only which makes experiments exponentially fast and efficient. Pycaret Setup interface.<br>

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2008-05-17%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>

### Implementing and Comparing Several Models Performances.
![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/Screenshot%202022-06-25%20at%2008-06-26%20California%20Housing%20Price%20Prediction%20Model%20Performances%20and%20Evaluation%20-%20Jupyter%20Notebook.png) <br>

* An easier method to see and evalute which models fits our dataset best is to check their Root MEan Squared Errors (RMSE) and Mean Average Percentage Errors (MAPE) values. In short, RMSE represents how far / deviates the predicted error value relative to the real ones, meanwhile MAPE represent it in percentage. SO its best if we want to look out for the lowest of both RMSE and MAPE. Here we can see top 3 models have relatively good RMSE and MAPE. We will use these 3 models as base for now. 3 Models includes, Light Gradient Boosting Machine, Random Forest and Extra Trees Regeressor. <br>

### Models Features Importances
Feature Importance refers to techniques that calculate a score for all the input features for a given model — the scores simply represent the “importance” of each feature. A higher score means that the specific feature will have a larger effect on the model that is being used to predict a certain variable. <br>

* LGBM <br>

 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index3.png) <br>
 
* Random Forest <br>

  ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index4.png) <br>
  
* Extra Trees <br>

 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index5.png) <br>
 
* Here we can conclude base on the 3 plot models above, factors to watch out for are The amount of Income households have earned, House located in the land, and longitude as well as latitude. However, we are yet to know how the values affect those factors are to the target features - houseprice-. TO address it, we use SHAP model interpreation model <br>

### SHAP Model Interpretability
Interpretability is the degree to which a human can understand the cause of a decision. Another one is: Interpretability is the degree to which a human can consistently predict the model’s result . The higher the interpretability of a machine learning model, the easier it is for someone to comprehend why certain decisions or predictions have been made <br>

SHAP stands for “SHapley Additive exPlanations.” Shapley values are a widely used approach from cooperative game theory. The essence of Shapley value is to measure the contributions to the final outcome from each player separately among the coalition, while preserving the sum of contributions being equal to the final outcome.
When using SHAP values in model explanation, we can measure the input features’ contribution to individual predictions.

* LGBM <br>

 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index6.png) <br>
 
* Random Forest <br>

 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index7.png) <br>
  
* Extra Trees <br>

 ![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index8.png) <br>

Insights we can get from those 3 plots above:
 * Higher value of housholds income led to a higher predicted house prices, meanwhile it is true for the opposite. This is very reasonable.
 * Higher value of both langitud and latitude led to a lower predicted house prices, the opposite is true as well. This means the closer it gets to ocean / sea level, the higher the huose prices will be.
 * THe last feature is weteher or not the house located inland. If is it true then it led to a lower predicted houseprices, and if it False led to a higher predicted house prices. This is synergious and inline with our previous latitude and longitude insight.

Now we proceed to visualize the data checking to see wehter to model prediction is right. <br>

### Visualization 
* `Total HouseHolds Income` in respect to `Houseprices` <br>

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index10.png) <br>
 
* As we can see from the plot a higher huseholds income is, the higher the houseprices, while the opposite is also true <br>
  
* `Ocean Prozimity` In Respect to `Houseprices` <br>

![alt text](https://github.com/ELSady/Regression-California-Housing-Price-Prediction/blob/main/index9.png) <br>
 
* Houses located in less than an hour from the ocean have the highest total house prices value, mean while ones located in inland is the second highest followed by located in Near ocean and Near bay. This sort of do not go in line with our model presumes, this as well confirms our third model insight is not totally correct, it did to some extent but not all. This lead to a conclusion that the factors influencing house prices significantly is the households income themselves and not where its located. <br>

### Conclusion
* WH=hen it comes to predicting house prices its best to look out for how much households income obtained in a year on average. Because this significantly affects how much the house will be worth in the near future.
