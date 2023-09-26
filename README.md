# CHL5230-Datathon1

## Team members

- Pourya Momtaz
- Xiaoxuan (Rose) Han
- Yiran Wang

## Public Health Data Analytics Datathon

This GitHub repository contains the code and resources for the CHL5230 Datathon1.

## Introduction

This datathon is dedicated to addressing the pressing challenge of obesity and its profound implications for public health. Additionally, we aim to explore the correlations between obesity and lung cancer risk factors to develop innovative solutions for these health concerns. Through meticulous data analysis, our primary goal is to uncover meaningful patterns and insights related to obesity within the provided datasets. These insights will serve as valuable information for decision-makers to craft effective surveillance and intervention strategies. Furthermore, we incorporate data on lung cancer risk factors from a research paper on Ethiopia to identify underlying patterns and risk factors. This approach allows us to investigate the intersections between obesity and lung cancer while developing predictive models for public health planning. Our central question revolves around understanding how various factors, including obesity and others, influence the severity of lung cancer within a specific population. Specifically, we explore gender-based differences in the influence of smoking and alcohol consumption on BMI. Thus, our ultimate aim is to create machine learning models that can accurately assess the severity of both lung cancer and obesity based on input data.

We are provided with two datasets for this task. The first dataset contains lung cancer risk factor data from Tikur Ambesa Hospital, enabling us to explore the influence of risk factors on lung cancer severity. The second dataset includes various public health factors that may affect BMI in Canada. Body Mass Index (BMI) is a globally recognized standard for assessing obesity in individuals and is crucial for evaluating their nutritional and physical health. We leverage these health factors to predict BMI levels since elevated BMI levels are often associated with certain cancerous conditions.

## Methodology

To simplify encoding, we first categorize BMI levels within the first dataset using predefined thresholds. We employ clustering techniques to address the questions and, for the initial question, train a classifier to establish a relationship.

## Data Engineering Process

The BMI dataset consists of 25,355 entries, encompassing 14,896 females and 8,639 males. Data cleaning involves handling duplicates and removing null values. To enhance machine learning performance, we encode the 'Gender' variable into a dummy variable ('Male' = 0 and 'Female' = 1). Following international BMI standards, we categorize BMI values into five levels: underweight (BMI < 18.5), healthy (18.5 - 24.9), overweight (25 - 29.9), obese (30 - 39.9), and severely obese (BMI >= 40), encoding these levels from 0 to 4.

The second dataset contains lung cancer risk factor data from Tikur Ambesa Hospital, with 1000 patients and 11 crucial risk factors.

For both datasets, we initially exclude all missing values. Subsequently, we split the data into training and testing datasets and scale the features before applying KNN to each dataset.

## Analysis

Our analysis comprises two parts: exploratory analysis and machine learning modeling analysis. We begin with exploratory data analysis to visualize patterns among explanatory variables, such as multicollinearities among different lung cancer risk factors or factors affecting obesity and BMI scores.

**Exploratory Analysis:**

Our exploratory analysis involved generating various graphs to gain insights. These graphs include histograms, scatter plots, and correlation heatmaps.

**Machine Learning Analysis:**

We applied the K-Nearest Neighbors (KNN) algorithm to both datasets, achieving high-accuracy predictions due to the availability of labeled data for BMI scores and lung cancer severity. In our first approach, we employed KNN with k=18 to predict BMI levels, splitting the dataset into 70% training data and 30% testing data.

Initially, we excluded 'Height' and 'Weight' variables to avoid multicollinearity in the model, as these factors directly contribute to calculating BMI. However, due to the weak relationship observed between BMI and other variables, we incorporated 'Height' and 'Weight' components to improve model performance.

In the lung cancer dataset, we applied KNN to all 10 features with k=3, using the severity of lung cancer as the labeled data. We also experimented with a reduced number of features to compare prediction accuracies, focusing on Gender, Age, Air Pollution, Alcohol Usage, Genetic Risk, Obesity, and Smoking to mitigate multicollinearities. In both KNN models, we employed Euclidean distance as all the data are quantitative. Our results indicated that k=3 yielded the highest accuracies.

## Findings

**Exploratory Analysis:**

- The correlation heatmap of the BMI dataset revealed that only 'Height' and 'Weight' demonstrated reasonably high correlations with BMI, while other parameters exhibited minimal correlation.

- In the lung cancer dataset, the severity of lung cancer displayed correlations with specific risk factors, including genetic risk, chest pain, alcohol usage, obesity, smoking, and coughing of blood. Additionally, some risk factors exhibited correlations with obesity.

**Machine Learning Analysis:

- After adding the height and weight factors into the model, the model achieves an overall accuracy of 0.77 with k=20. This means the model correctly classified 77% BMI level based on health factors. 

- According to the classification report, "underweight" has a precision of 0.99, while "severely obese" has a precision of 1.00. The precision with other levels is less precise, followed by healthy (0.74), overweight (0.73), and obese (0.80). This demonstrates the model correctly identifying individuals as underweight and severely obese. Recall measures indicate whether a model can recognize every instance of a class. In our data set, the recall of a severely obese case is only 0.08, which may be due to an extremely uncommon severely obese instance. The model performs a wonderful job of recognizing most of the instances of that class, with the exception of the seriously obese, whose recall is in the range of 0.52 to 0.93. 

- From the first KNN model that included all features with K setting as 3, we got relatively high precisions with an average of 0.93. In another KNN model that only included important features, we observed slightly lower accuracies with an average of 0.91, thus we will proceed with the KNN model with 10 features, even though potential overfitting might occur. 