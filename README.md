# CHL5230-Datathon1

## Team members

- Pourya Momtaz
- Xiaoxuan (Rose) Han
- Yiran Wang

## Public Health Data Analytics Datathon

This GitHub repository hosts code and resources for the CHL5230 Datathon1. This datathon focuses on addressing the challenge of obesity and its implications for public health. Additionally, we explore correlations between obesity and lung cancer risk factors to develop innovative solutions for these pressing health issues.

## Objective

The main objectives of this datathon are:

1. **Thorough Data Analysis**: Analyze the provided datasets to uncover meaningful patterns and insights related to obesity. These findings will inform decision-makers in crafting effective surveillance and intervention strategies.

2. **Correlation and Integration**: Integrate data on lung cancer risk factors from a research paper on Ethiopia to identify underlying patterns and risk factors. This enables exploration of the intersections between obesity and lung cancer and the development of models for public health planning.


# Questions

How do lifestyle factors such as daily caloric intake, physical activity, and sleep hours correlate with BMI in the Canadian population?

Are there gender-based differences in the influence of smoking and alcohol consumption on BMI?

Can we predict BMI categories using machine learning techniques based on the given public health factors?

# Methodology

We define some thresholds on BMI on the first dataset to categorize them, so it would be easier to encode. Then we're going to use clustering methods to answer the questions and for the first question we'll train a classifier to learn a relation.

# Introduction
The Body Mass Index(BMI) is a crucial international standard frequently used to gauge the extent of obesity in humans and can be used to assess a person's nutritional and physical health. We use several health factors to forecast the level of BMI because a higher BMI may be linked to some cancerous conditions.

# Data engineering process
The BMI dataset contains 25,355 entries, comprising 14,896 females and 8,639 males. We use the duplication and deleting the null value for data cleaning. For better performance in machine learning, we optimize the gender variable into the dummy variable(‘Male’=0 and ‘Female’=1). Based on the international standard for BMI, we categorized BMI values into five levels: underweight(BMI<18.5), healthy(18.5-24.9), overweight(25-29.9), obese (30-39.9), and severely obese(BMI>=40), and encoding these five levels to 0-4.

# Analysis
The K-Nearest Neighbours algorithm with k=18 is used in our analysis to forecast BMI levels. Due to the KNN method's superior performance, we divided the dataset into 70% training data and 30% testing data. 

Our first idea is to exclude the 'Height' and 'Weight' variables from the analysis since BMI is already calculated from these two features. The reason for excluding them is to avoid multicollinearity in the model. Multicollinearity occurs when predictor variables in a regression model are highly correlated, making it difficult to distinguish the individual effects of each predictor. In this case, 'Height' and 'Weight' are directly used to calculate BMI, so including them would introduce redundancy and could lead to instability in the model's coefficients.

However, when we look at the dataset one correlation heatmap. Only height and weight show a reasonably high correlation with BMI; other parameters appear to have no correlation with BMI. As a result, the situation is very extreme.

We begin by using the KNN estimator without height and weight. Only 30% of the time is accurate, so there is a weak relationship between BMI and other variables. Therefore, we include the height and weight components in our model for improved performance.

After adding the height and weight factors into the model, the model achieves an overall accuracy of 0.77 with k=20. This means the model correctly classified 77% BMI level based on health factors. 
According to the classification report, "underweight" has a precision of 0.99, while "severely obese" has a precision of 1.00. The precision with other levels is less precise, followed by healthy (0.74), overweight (0.73), and obese (0.80). This demonstrates the model correctly identifying individuals as underweight and severely obese. Recall measures indicate whether a model can recognize every instance of a class. In our data set, the recall of a severely obese case is only 0.08, which may be due to an extremely uncommon severely obese instance. The model performs a wonderful job of recognizing most of the instances of that class, with the exception of the seriously obese, whose recall is in the range of 0.52 to 0.93. 

