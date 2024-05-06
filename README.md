# Customer Churn Prediction Project

This project aims to predict customer churn for a telecommunications company using machine learning techniques. The model developed in this project can help identify customers who are likely to churn, allowing the company to take proactive measures to retain them.

## Overview

Customer churn refers to the phenomenon where customers stop doing business with a company. 

It is a critical metric for businesses, especially subscription-based services like telecommunications companies, as retaining existing customers is often more cost-effective than acquiring new ones.

## Dataset

The dataset used for this project contains various customer attributes such as monthly charges, total charges, tenure, contract type, internet service, and more. It also includes a target variable indicating whether a customer churned or not.

## Methodology

* Data Preprocessing: The dataset underwent preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features.
  
* Expalanatory Data Analysis: The data was visualized to gain deeper insights into the factors influencing customer churn and to uncover patterns within the dataset.
    
* Model Training: Several machine learning algorithms were trained and evaluated using performance metrics such as accuracy, precision, recall, and ROC-AUC score.
  
* Model Deployment: The best-performing model, Logistic Regression, was deployed as a web service using Flask. The deployed model allows users to input customer information and receive churn predictions in real-time.

## Files and Folders

WA_Fn-UseC_-Telco-Customer-Churn.csv: Raw dataset containing customer information and churn labels.

telco.ipynb: Jupyter Notebook containing the data preprocessing, model training, and evaluation code.

model.pkl: File with the model to be used in predicting customer churn

app.py: Flask application file responsible for handling HTTP requests and serving churn predictions.

templates/: Directory containing HTML file for the web interface.


## Usage

Clone the repository: git clone https://github.com/Risper8/Churn-Prediction/

Access the web interface at https://risper8.github.io/Churn-Prediction/ in your browser.

## Technologies Used

* Python
  
* Pandas, NumPy, Scikit-learn
  
* Flask
  
* HTML, CSS

