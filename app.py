import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(
    page_title="Heart Attack Risk Prediction Project",
    page_icon="❤️",
    layout="wide"
)

# 标题和项目概览
st.title("❤️Heart Attack Risk Prediction Project")

st.markdown("Welcome to our group project for **WQD7006/WQD7012 - Machine Learning for Data Science**.")

# 项目导航说明
st.markdown("### Use the sidebar to navigate between pages:")

col1, col2 = st.columns([1, 9])

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 Data & EDA
    - Dataset description
    - Exploratory data analysis
    - Data preprocessing steps
    - Visual insights
    """)

with col2:
    st.markdown("""
    #### 🔮 Model Predict
    - Try predictions via form
    - Upload CSV for batch prediction
    - View prediction results
    - Get health recommendations
    """)

with col3:
    st.markdown("""
    #### 📁 GitHub Links
    - View source code
    - Project documentation
    - References and resources
    - Contact information
    """)

# 项目简介
st.markdown("---")
st.markdown("## Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### SDG 3 | Predictive Modeling for Detection of Heart Attacks

    Heart attacks, or myocardial infarctions, continue to be a critical global health burden, contributing substantially to morbidity and mortality rates. According to the World Health Organization, cardiovascular diseases (CVDs) are the leading cause of death globally, claiming approximately 17.9 million lives in 2019, or 32% of all global deaths.

    This project supports United Nations Sustainable Development Goal 3 (SDG 3) – Good Health and Well-being – by applying Machine Learning to build predictive tools that assess heart attack risk. Using data from the Behavioral Risk Factor Surveillance System (BRFSS), we've developed a model that can identify individuals at elevated risk of heart attack based on key risk factors.
    """)

with col2:
    st.image("https://www.heart.org/-/media/Images/Logos/Global-Do-No-Edit/Header/AHA_icon.svg?h=90&w=70&sc_lang=en&hash=77591B08CFB60097E13610EC8BB886B6",
             caption="Heart Attack Risk Factors (Source: heart.org)")

# 模型架构
st.markdown("## Machine Learning Pipeline")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    Our machine learning pipeline follows these key steps:

    1. **Data Preprocessing**: The BRFSS dataset is cleaned, with missing values handled and features transformed for machine learning compatibility.

    2. **Feature Engineering**: We created new features such as CumulativeHealthConditions to better capture health risk profiles.

    3. **Class Imbalance Handling**: The dataset has significant class imbalance (only 5.4% positive cases), which we addressed using SMOTEENN technique.

    4. **Model Training & Optimization**: We evaluated multiple classifiers including Logistic Regression, Random Forest, and XGBoost.

    5. **Threshold Optimization**: Model thresholds were optimized to balance precision and recall, particularly important for medical applications.

    6. **Deployment**: The final logistic regression model is deployed in this web application for real-time risk assessment.
    """)

# with col2:
#     st.image("https://miro.medium.com/v2/resize:fit:1400/1*VSQ0XEywxSgZBwW05gm3pQ.png",
#              caption="Machine Learning Pipeline")

# 显示主要发现
st.markdown("## Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Key Risk Factors:

    Our analysis identified several key risk factors for heart attacks:

    * **History of coronary heart disease**: Strong predictor across all models
    * **Height**: Significant association with heart attack risk
    * **Sleep duration**: Important lifestyle factor influencing risk
    * **Cumulative health conditions**: Multiple chronic conditions significantly increase risk
    * **Age**: Consistent predictor of heart attack risk
    """)

# 团队信息
st.markdown("---")
st.markdown("## Project Team - Group 9")

team_data = {
    "Name": ["Mah Seau Sher", "Gayathri Danappal", "Huang Lili", "Xian Zhiyi"],
    "Matric Number": ["22115483", "17116052", "23107324", "23122622"]
}

st.table(pd.DataFrame(team_data))

# 页脚
st.markdown("---")
st.markdown("WQD7006/WQD7012 Project | University of Malaya")