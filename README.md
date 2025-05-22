# Heart Attack Risk Prediction (Group 9 - WQD7006/WQD7012)
A machine learning project developed for the course WQD7006/WQD7012 - Machine Learning for Data Science, aiming to predict the risk of heart attack based on health and lifestyle indicators. The system is deployed using Streamlit for interactive access and demonstration.

# Project Structure

**WQD7006-7012-Group9/**
- **data/** 
  - v2heartattack_finaldataset.csv
- **pages/**
  - 1_EDA_and_Dataset.py
  - 2_Model_Predict.py
  - 3_GitHub_Links.py
- app.py
- model_params.pkl
- requirements.txt
- README.md

# Model Overview

- **Input Features**:  
  Body Mass Index (BMI), sleep hours, general health status, chronic disease history, age, and other demographic and health-related indicators.

- **Models Used for Evaluation**:  
  Logistic Regression, Random Forest, and XGBoost were trained and compared.

- **Deployed Model**:  
  We selected **Logistic Regression** as the final deployed model.

- **Imbalanced Data Handling**:  
  A hybrid approach combining **SMOTE** and **ENN** was used to address class imbalance.

- **Evaluation Metrics**:  
  Accuracy, Precision, Recall, F1-Score, and ROC-AUC.


# How to Run Locally
#### Step 1: Clone the repository
git clone https://github.com/Zoe-Xian/WQD7006-7012-Group9.git

cd WQD7006-7012-Group9

#### Step 2: Install dependencies
pip install -r requirements.txt

#### Step 3: Launch the app
streamlit run app.py

# Online Demo
You can try the deployed version here: https://wqd7006-7012-occ3-group9.streamlit.app/ 

# Team & Course Info
Course: WQD7006/WQD7012 – Machine Learning for Data Science

Supervisor: Dr. Riyaz Ahamed

Group 9 Members: 
- Mah Seau Sher
- Gayathri Danappal
- Huang Lili
- Xian Zhiyi
