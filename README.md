# CHL5230-Datathon1

## Team members

- Pourya Momtaz
- Xiaoxuan (Rose) Han
- Yiran Wang

# Public Health Data Analytics Datathon

This GitHub repository contains the code and resources for the CHL5230 Datathon1.

## Introduction

This datathon is dedicated to addressing the pressing challenge of obesity and its profound implications for public health. Additionally, we aim to explore the correlations between obesity and lung cancer risk factors to develop innovative solutions for these health concerns. Through meticulous data analysis, our primary goal is to uncover meaningful patterns and insights related to obesity within the provided datasets. These insights will serve as valuable information for decision-makers to craft effective surveillance and intervention strategies. Furthermore, we incorporate data on lung cancer risk factors from a research paper on Ethiopia to identify underlying patterns and risk factors. This approach allows us to investigate the intersections between obesity and lung cancer while developing predictive models for public health planning. Our central question revolves around understanding how various factors, including obesity and others, influence the severity of lung cancer within a specific population. Thus, our ultimate aim is to create machine learning models that can accurately assess the severity of both lung cancer and obesity based on input data.

We are provided with two datasets for this task. The first dataset contains lung cancer risk factor data from Tikur Ambesa Hospital, enabling us to explore the influence of risk factors on lung cancer severity. The second dataset includes various public health factors that may affect BMI in Canada. Body Mass Index (BMI) is a globally recognized standard for assessing obesity in individuals and is crucial for evaluating their nutritional and physical health. We leverage these health factors to predict BMI levels since elevated BMI levels are often associated with certain cancerous conditions.


# Methodology

## Data Exploration and Preprocessing

Our methodology begins with data exploration and preprocessing to gain insights into the provided datasets and prepare them for further analysis. We initially load the datasets, handle missing values, and clean the data. In this process, we ensure that the data is in a format suitable for analysis.

### Gender-Based Distribution Analysis

We perform a gender-based distribution analysis to understand the distribution of various health-related features within the dataset. This analysis helps us identify potential differences in health attributes between males and females. We create histograms for features such as Age, Calorie Intake, Sugar Intake, Physical Activity, Fast Food Frequency, Screen Time, Height, Weight, and BMI, stratified by gender.

### BMI Categorization

To facilitate classification tasks, we categorize BMI values into five levels: Underweight, Healthy, Overweight, Obese, and Severely Obese. This categorization simplifies the BMI prediction problem.

## Predictive Modeling for BMI

Our first predictive modeling task involves predicting BMI levels using K-Nearest Neighbors (KNN) algorithms. We perform this task with and without the inclusion of Height and Weight features to investigate the impact of these variables on BMI prediction.

### KNN with Height and Weight

We split the data into training and testing sets and scale the features for better model performance. We employ the KNN algorithm with k=20, which represents the number of neighbors used for classification. The results include a confusion matrix and classification report, providing insights into the model's accuracy, precision, recall, and F1-score for each BMI category. We visualize the confusion matrix to illustrate the model's performance.

### KNN without Height and Weight

Similar to the previous task, we predict BMI levels using KNN, but this time, we exclude Height and Weight features. We again provide a confusion matrix and classification report to assess the model's performance.

### Hyperparameter Tuning

To optimize the KNN algorithm's hyperparameter (k), we perform a grid search to identify the best k value for our dataset. We visualize the relationship between k values and model accuracy to determine the optimal k for future use.

## Predictive Modeling for Lung Cancer Severity

In the second part of our analysis, we focus on predicting lung cancer severity based on various risk factors. We use the KNN algorithm for this task, aiming to classify the severity of lung cancer accurately.

### KNN for Lung Cancer Severity

We preprocess the data, including scaling the features and splitting the dataset into training and testing sets. We apply the KNN algorithm with k=3 to predict lung cancer severity. We present a confusion matrix and classification report to evaluate the model's performance.

### Hyperparameter Tuning for Lung Cancer Severity

Similar to the BMI prediction, we conduct hyperparameter tuning to find the optimal k value for the KNN algorithm in predicting lung cancer severity. We visualize the relationship between k values and model accuracy to determine the best k.

## Additional Analysis: Feature Clustering

In addition to predictive modeling, we explore feature clustering to uncover patterns and relationships within the data. We apply K-Means clustering to different combinations of features in both datasets.

### K-Means Clustering for Dataset 1

We perform K-Means clustering on various combinations of features from Dataset 1, visualizing the clusters in scatter plots. This analysis helps us identify natural groupings or patterns within the data.

### K-Means Clustering for Dataset 2

Similarly, we apply K-Means clustering to different feature combinations in Dataset 2 and visualize the resulting clusters. This analysis aids in understanding how risk factors for lung cancer correlate and form distinct groups.


## Data Engineering Process

The BMI dataset consists of 25,355 entries, encompassing 14,896 females and 8,639 males. Data cleaning involves handling duplicates and removing null values. To enhance machine learning performance, we encode the 'Gender' variable into a dummy variable ('Male' = 0 and 'Female' = 1). Following international BMI standards, we categorize BMI values into five levels: underweight (BMI < 18.5), healthy (18.5 - 24.9), overweight (25 - 29.9), obese (30 - 39.9), and severely obese (BMI >= 40), encoding these levels from 0 to 4.

The second dataset contains lung cancer risk factor data from Tikur Ambesa Hospital, with 1000 patients and 11 crucial risk factors.

For both datasets, we initially exclude all missing values. Subsequently, we split the data into training and testing datasets and scale the features before applying KNN to each dataset.

# Analysis

Our analysis comprises two parts: exploratory analysis and machine learning modeling analysis. We begin with exploratory data analysis to visualize patterns among explanatory variables, such as multicollinearities among different lung cancer risk factors or factors affecting obesity and BMI scores.

**Exploratory Analysis:**

Our exploratory analysis involved generating various graphs to gain insights. These graphs include histograms, scatter plots, and correlation heatmaps.

**Machine Learning Analysis:**

We applied the K-Nearest Neighbors (KNN) algorithm to both datasets, achieving high-accuracy predictions due to the availability of labeled data for BMI scores and lung cancer severity. In our first approach, we employed KNN with k=20 to predict BMI levels, splitting the dataset into 70% training data and 30% testing data.

Initially, we excluded 'Height' and 'Weight' variables to avoid multicollinearity in the model, as these factors directly contribute to calculating BMI. However, due to the weak relationship observed between BMI and other variables, we incorporated 'Height' and 'Weight' components to improve model performance.

In the lung cancer dataset, we applied KNN to all 10 features with k=3, using the severity of lung cancer as the labeled data. We also experimented with a reduced number of features to compare prediction accuracies, focusing on Gender, Age, Air Pollution, Alcohol Usage, Genetic Risk, Obesity, and Smoking to mitigate multicollinearities. In both KNN models, we employed Euclidean distance as all the data are quantitative. Our results indicated that k=3 yielded the highest accuracies.

# Findings

**Exploratory Analysis:**


- Upon analyzing our datasets, we observed potential gender bias in both the BMI and lung cancer datasets. In the BMI dataset, we found that there were more female participants than male participants. This gender imbalance could introduce bias into our analysis, as certain health-related factors may differ between genders.

- Further investigation into the distribution of health-related features by gender revealed interesting findings. Despite the gender imbalance in the dataset, we found that health-related features, such as age, calorie intake, sugar intake, physical activity, fast food frequency, screen time, height, weight, and BMI, followed nearly identical distributions for both genders. This suggests that, in this particular dataset, the health-related factors we are analyzing are consistent across genders.

- The correlation heatmap of the BMI dataset revealed that only 'Height' and 'Weight' demonstrated reasonably high correlations with BMI, while other parameters exhibited minimal correlation.

- In the lung cancer dataset, the severity of lung cancer displayed correlations with specific risk factors, including genetic risk, chest pain, alcohol usage, obesity, smoking, and coughing of blood. Additionally, some risk factors exhibited correlations with obesity.

**Machine Learning Analysis:

- To optimize the performance of our K-Nearest Neighbors (KNN) model for predicting BMI levels, we conducted an analysis to find the optimal value of K, the number of neighbors., We tested different K values ranging from 1 to 99 and evaluated their impact on the model's accuracy. The results showed an interesting trend: the accuracy of the model increased initially, reaching a local maximum of approximately 0.33. After this point, the accuracy started to decrease as K increased further. To select the optimal K, we considered both the elbow method and the maximum accuracy point. Based on these criteria, we determined the best K value for our KNN model. After selecting the optimal K value, we trained our KNN model on the training data and evaluated its performance on the test data. The results provided insights into the model's accuracy, precision, recall, and F1-score: Accuracy: 0.3223, Precision: 0.2417, Recall: 0.3223, F1-Score: 0.2407

It's important to note that the relatively low evaluation metrics might be attributed to the nature of predicting BMI. BMI may not have significant correlations with factors other than height and weight, making it challenging to accurately predict using health-related features alone.


- After including height and weight in our feature set, we evaluated the performance of the K-Nearest Neighbors (KNN) model. The results demonstrated a significant improvement in model accuracy and predictive power: Accuracy: 0.7706, Precision: 0.7916, Recall: 0.7706, F1-Score: 0.7593. These improved metrics reflect the enhanced capability of the model to predict BMI levels when considering height and weight as essential factors. Notably, height and weight are core elements in BMI calculations, and although their inclusion should theoretically lead to precise BMI predictions, our analysis suggests that the incorporation of additional factors may have contributed to the model's complexity and affected its accuracy.

- According to the classification report, "underweight" has a precision of 0.99, while "severely obese" has a precision of 1.00. The precision with other levels is less precise, followed by healthy (0.74), overweight (0.73), and obese (0.80). This demonstrates the model correctly identifying individuals as underweight and severely obese. Recall measures indicate whether a model can recognize every instance of a class. In our data set, the recall of a severely obese case is only 0.08, which may be due to an extremely uncommon severely obese instance. The model performs a wonderful job of recognizing most of the instances of that class, with the exception of the seriously obese, whose recall is in the range of 0.52 to 0.93. 

- From the first KNN model that included all features with K setting as 3, we got relatively high precisions with an average of 0.93. In another KNN model that only included important features, we observed slightly lower accuracies with an average of 0.91, thus we will proceed with the KNN model with 10 features, even though potential overfitting might occur. 

- The linear regression analysis for the BMI dataset yielded the following results: Mean Squared Error (MSE): 1.0022, R-squared (R2) Score: 0.9792. These results indicate that the linear regression model performed exceptionally well in predicting BMI values. The low MSE and high R-squared score of approximately 0.9792 suggest that the model's predictions closely align with the actual BMI values.

It's important to note that linear regression is typically used for continuous data, and in this case, we applied it specifically to the BMI dataset, which contains continuous BMI values. The strong performance of the model highlights its effectiveness in capturing the relationships between the input features and BMI.

- When we conducted the linear regression analysis after removing height and weight from the dataset, we obtained the following results: Mean Squared Error (MSE): 48.1323, R-squared (R2) Score: -0.0011
These results present a stark contrast to the previous linear regression analysis that included height and weight as features. In this case, the high MSE of approximately 48.1323 and the negative R-squared score of approximately -0.0011 indicate a poor performance of the model. The negative R-squared score suggests that the model did not capture the underlying relationships between the input features and the BMI values, resulting in predictions that did not align with the actual data

- We rigorously tested different values of k for K-Means clustering, aiming to identify distinct clusters within the data. Unfortunately, none of these attempts resulted in clearly defined clusters. This outcome suggests that the BMI dataset, whether including or excluding height and weight as features, does not naturally form discrete clusters based on the available attributes. The lack of well-defined clusters could be due to the complexity of BMI as a continuous variable influenced by multiple factors, including height and weight.


# Conclusion

In conclusion, our analyses using various machine learning techniques and statistical methods have provided valuable insights into the prediction of BMI (Body Mass Index) and its associated factors.

First, we explored the distribution of BMI by gender, revealing that our dataset had a larger representation of females. While this gender imbalance could potentially introduce bias, we subsequently examined the distribution of health-related features by gender. Interestingly, we found that these features followed similar distributions for both males and females, indicating that health-related factors did not significantly differ between genders. This insight suggests that gender may not be a major confounding factor in our analysis.

We proceeded to analyze the performance of a K-Nearest Neighbors (KNN) model for BMI prediction. Initially, we tested various values of k and observed an almost ascending graph of accuracy with a local maximum of around 0.33. Beyond this point, accuracy started to decrease, suggesting an optimal k value. The confusion matrix for this optimal k value revealed an accuracy of approximately 0.3223, precision of 0.2417, recall of 0.3223, and an F1-score of 0.2407. These relatively low evaluation metrics may be attributed to the limited correlation between BMI and factors other than height and weight.

To address this limitation, we incorporated height and weight into our feature set, significantly enhancing the KNN model's performance. With height and weight as essential factors in BMI calculation, the KNN model achieved a notable accuracy of approximately 0.7706, precision of 0.7916, recall of 0.7706, and an F1-score of 0.7593. These improved metrics reflect the model's enhanced capability to predict BMI levels when considering height and weight as crucial factors.

Furthermore, our linear regression analysis produced promising results. When height and weight were included as features, the linear regression model demonstrated a low Mean Squared Error (MSE) of approximately 1.0023 and a high R-squared (R2) score of approximately 0.9792. This outcome suggests that the model effectively predicts BMI when height and weight are considered. However, upon removing these factors, the linear regression model's performance declined significantly, with an elevated MSE of approximately 48.1323 and a negative R-squared score of approximately -0.0011. This drastic contrast underscores the strong correlation between BMI and height/weight, highlighting their fundamental role in BMI prediction.

In summary, our findings emphasize that BMI is highly correlated with height and weight, while other features in the dataset have limited influence on BMI prediction. Height and weight are the primary drivers of accurate BMI predictions, and their exclusion from the analysis can significantly compromise model performance. These insights underscore the importance of considering height and weight when assessing body mass index and provide valuable guidance for future analyses in this domain.

# Assumptions and Future Directions

While our analysis provides valuable insights, it is essential to acknowledge certain assumptions and potential areas for future exploration:

1. **Data Source Variability**: Our analysis focused on specific datasets related to obesity and lung cancer risk factors. Future research could incorporate data from diverse sources and populations to enhance the generalizability of findings.

2. **Feature Engineering**: Further feature engineering and the incorporation of additional variables may improve BMI prediction and clustering outcomes.

3. **Bias Considerations**: Future analyses should carefully consider and address potential biases, including gender imbalances, to ensure robust and equitable results.

4. **Additional Predictive Models**: Exploring alternative machine learning models and techniques may offer new perspectives on BMI prediction and feature clustering.



# Individual Contributions
- Pourya Momtaz: Led the group in terms of the overall structure, inluding coding; conducted the overall explanatory analysis and the modelling parts.
- Xiaoxuan (Rose) Han: Focused on analyzing the lung cancer dataset; structured and worked on the written report based on analysis outputs from the coding.
- Yiran Wang: Focused on analyzing the BMI dataset; worked on incorporating the key information from the written report to the presentation slides. 

Each group member has an equal contribution to this project in both the coding analysis and report writing process, where constant discussions have happened throughout the week.
