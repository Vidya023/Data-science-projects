# Data-science-projects

1. [Customer Segmentation using Unsupervised Machine Learning Techniques](Customer_segmentation.ipynb) 

`Project Summary`

* **Customer Segmentation**: Applied K-means, Hierarchical, and DBSCAN clustering techniques to group customers based on similar behaviors and preferences.

* **RFM Modelling**: Used Recency, Frequency, and Monetary (RFM) analysis to evaluate and score customers, aiding in the segmentation process.

* **Correlation Analysis**: Analyzed correlations between different customer metrics to understand their relationships and influence on segmentation.

* **Clustering Algorithm Implementation**: Implemented and compared various clustering algorithms including K-means, Hierarchical, and DBSCAN to identify customer segments.

* **Cluster Validation**: Despite the absence of clearly separated clusters in visual plots, the clusters formed were validated through statistical analysis and algorithmic outputs.

2. [Market Basket Analysis and Association Rule Mining with FPGrowth in PySpark](Market_Basket_Analysis_and_Association_Rule_Mining_with_FPGrowth_PySpark.ipynb)

`Project Summary`

* **Understanding of FPGrowth Algorithm**: Learn how to apply the FPGrowth algorithm for mining frequent itemsets and generating association rules.

* **Data Preparation and Transformation**: Gain experience in transforming and preparing data for analysis, including indexing categorical variables and aggregating transaction data.

* **Feature Analysis**: Learn how to perform association rule mining for different subsets of data (in this case, by country) and compare results across these subsets.

* **Metrics Interpretation**: Understand key metrics like support, confidence, and lift, and how they are used to evaluate the strength of association rules.

* **Results Aggregation and Analysis**: Learn how to aggregate and summarize association rules across different groups, and how to identify the most significant rules.

* **PySpark Data Handling**: Gain practical knowledge of handling large datasets with PySpark, including using SQL queries and DataFrame operations.

3. [Predictive Modeling of Shipment Pricing Using Machine Learning Regression Algorithms](Predictive_Modeling_of_Shipment_Pricing_using_ML_Regression.ipynb)

`Project Summary`

* **Data Analysis**: Conducted univariate and multivariate analysis for numerical and categorical features to understand the dataset and its structure.

* **Preprocessing**: Applied preprocessing techniques such as feature encoding, scaling, and handling outliers. Utilized PowerTransformer, Log Transformation, and various scaling methods to prepare data for modeling.

* **Correlation and Multicollinearity Check**: Performed correlation analysis and checked for multicollinearity using Variance Inflation Factor (VIF) to ensure the integrity of the features.

* **Model Selection**: Implemented and compared several regression algorithms, including Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, SVR, XGBoost, and CatBoost.

* **Hyperparameter Tuning**: Used RandomizedSearchCV for hyperparameter tuning to optimize the performance of the top models.

* **Model Evaluation**: Identified the best model, CatBoost Regressor, which achieved a 97.04% score.


4. [Sentiment Analysis for Predicting Consumer Disputes Using ML Classification Algorithms](Sentiment_Analysis_for_Consumer_Dispute_Prediction.ipynb)

`Project Summary`

* **Data Analysis**: Performed univariate and multivariate analysis on target columns, including handling two date features and calculating the duration for complaints.

* **Text Processing and Vectorization**: Conducted comprehensive text preprocessing, including punctuation removal, stop word removal, lowercasing, tokenization, and lemmatization. Used TF-IDF vectorization to convert text data into numerical features.

* **Feature Engineering**: Created functions for text tokenization and lemmatization, transformed data using column transformers, and prepared the dataset for modeling.

* **Handling Imbalanced Data**: Applied SMOTETomek to address imbalance in the dataset

* **Model Selection and Training**: Implemented and compared various classification algorithms, including Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Logistic Regression, K-Nearest Neighbors, XGBoost, and CatBoost.

* **Hyperparameter Tuning**: Used Hyperopt for hyperparameter tuning of XGBoost and CatBoost models, selecting XGBoost as the best model with an accuracy of 83.33%.



5. [Predictive Time Series Modeling for Bitcoin Closing Price Using Machine Learning and Deep Learning Techniques](Predictive_Time_series_modeling.ipynb)

`Project Summary`

* **Data Retrival**: Collected data from Yahoo Finance for Bitcoin (BTC) and other cryptocurrencies from January 1, 2018, to September 1, 2019.

* **Data Visualization and Normalization**: Visualized the performance of selected cryptocurrencies and calculated p-values for correlations. 

* **Correlation Analysis**: Implemented correlation analysis, finding that Ethereum (ETH) had the highest correlation.

* **Model Selection and Training**: Scaled data using MinMaxScaler and trained various models including LSTM, Moving Average, ARIMA, and ANN to predict Bitcoin's closing price.

* **Model Evaluation**: Evaluated the performance of each model, achieving the following accuracies:
LSTM: 93.68% \\
Moving Average: 83.05% \\
ARIMA: 72.18% \\
ANN: 91.87% \\

* **Best Model**: Determined that LSTM model provided highest accuracy, demonstrating its effectiveness for time series forecasting.