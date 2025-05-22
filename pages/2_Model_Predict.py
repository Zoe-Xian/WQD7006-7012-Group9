import streamlit as st
import pandas as pd
import numpy as np
import pickle


# 定义ThresholdModel类 - 确保这个定义与队友的一致
# Define model wrapper class
class ThresholdModel:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        # Make predictions using the optimal threshold
        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        # Return probability predictions
        return self.model.predict_proba(X)


# Feature normalization function
def normalize_features(input_df):
    """
    Normalize specific numeric features according to the same method used during training.
    This uses MinMaxScaler approach, but applies it directly to the features.
    """
    # Define min and max values for each feature based on the training dataset
    # These values should be calculated from your training data
    normalization_params = {
        'Body Mass Index(BMI)': {'min': 10.0, 'max': 50.0},
        'Height(m)': {'min': 1.0, 'max': 2.5},
        'Sleep Hours per night': {'min': 0.0, 'max': 24.0}
    }

    # Create a copy of the input DataFrame to avoid modifying the original
    normalized_df = input_df.copy()

    # Apply normalization only to the specified features
    for feature, params in normalization_params.items():
        if feature in normalized_df.columns:
            # Apply MinMax scaling: (x - min) / (max - min)
            min_val = params['min']
            max_val = params['max']
            normalized_df[feature] = (normalized_df[feature] - min_val) / (max_val - min_val)

    return normalized_df


# 设置页面配置
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# 加载参数并重建模型
try:
    # 加载参数
    with open("model_params.pkl", "rb") as f:
        params = pickle.load(f)

    # 从sklearn导入LogisticRegression
    from sklearn.linear_model import LogisticRegression

    # 重建基础模型
    if params['type'] == 'LogisticRegression':
        base_model = LogisticRegression()
        base_model.coef_ = params['coef']
        base_model.intercept_ = params['intercept']
        base_model.classes_ = params['classes']

        # 包装在ThresholdModel中
        model = ThresholdModel(base_model, threshold=params['threshold'])

        # st.sidebar.success(f"✅ Model loaded successfully with threshold: {params['threshold']:.3f}")
    else:
        # 如果不是LogisticRegression类型，处理其他可能的类型
        st.error(f"Unsupported model type: {params['type']}")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 设定页面标题
st.title("❤️ Heart Attack Risk Prediction")

st.markdown("""
This app predicts the risk of heart attack based on various health factors. 
You can either fill in your information manually or upload a CSV file with multiple records.
""")

# 获取18个输入字段（必须与模型训练用列名顺序一致）
feature_columns = [
    'Coronary Heart Disease History',
    'CumulativeHealthConditions',
    'Depression Diagnosis',
    'General health status',
    'Drank Alcohol Past 30 days',
    'Body Mass Index(BMI)',
    'Arthritis Status',
    'Asthma Status',
    'Sleep Hours per night',
    'Height(m)',
    'Stroke History',
    'Physical activity Past 30 days',
    'Skin Cancer History',
    'AgeNumerical',
    'Diabetes Status',
    'COPD (lung disease) Status',
    'Difficulty Walking/Climbing Stairs',
    'Kidney Disease Status'
]

# 侧边栏选择输入模式
st.sidebar.header("Input Options")
option = st.sidebar.radio("Select input mode:", ["📝 Fill in Form", "📁 Upload CSV"])

# 创建两列布局
if option == "📝 Fill in Form":
    st.subheader("📝 Enter Your Health Information")

    # 创建3列布局以优化表单显示
    col1, col2, col3 = st.columns(3)

    # 初始化用户输入字典
    user_input = {}

    cols = [col1, col2, col3]
    for i, col in enumerate(feature_columns):
        with cols[i % 3]:
            if col == "Body Mass Index(BMI)":
                user_input[col] = st.number_input(col, min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            elif col == "Height(m)":
                user_input[col] = st.number_input(col, min_value=1.0, max_value=2.5, value=1.7, step=0.01)
            elif col == "Sleep Hours per night":
                user_input[col] = st.number_input(col, min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            elif col == "AgeNumerical":
                user_input[col] = st.number_input(col, min_value=18, max_value=100, value=40, step=1)
            elif col == "CumulativeHealthConditions":
                user_input[col] = st.number_input(col, min_value=0, max_value=10, value=0, step=1)
            elif col == "General health status":
                user_input[col] = st.number_input(col, min_value=1, max_value=5, value=3, step=1,
                                                  help="1=Poor, 2=Fair, 3=Good, 4=Very good, 5=Excellent")
            else:
                user_input[col] = st.selectbox(col, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # 添加一个预测按钮
    if st.button("💡 Predict Heart Attack Risk", use_container_width=True):
        input_df = pd.DataFrame([user_input])

        # 重要步骤：应用特征标准化
        normalized_input_df = normalize_features(input_df)

        # 使用模型进行预测
        try:
            prediction = model.predict(normalized_input_df)[0]
            prob = model.predict_proba(normalized_input_df)[0][1]

            # 创建结果展示区
            st.subheader("Prediction Results")

            # 使用米尺样式展示风险概率
            st.markdown("### Risk Assessment")

            # 创建进度条显示风险
            risk_color = "green" if prob < 0.3 else "orange" if prob < 0.7 else "red"
            st.progress(prob, text=f"Risk Probability: {prob:.2%}")

            # 显示结果
            if prediction == 1:
                st.error("⚠️ **High Risk of Heart Attack Detected**")
                st.markdown("Based on the provided information, our model predicts a significant risk of heart attack.")
            else:
                st.success("✅ **Low Risk of Heart Attack Detected**")
                st.markdown("Based on the provided information, our model predicts a low risk of heart attack.")

            # 添加一些建议
            st.subheader("Health Recommendations")
            st.markdown("""
            * Regular cardiovascular check-ups are essential for heart health
            * Maintain a balanced diet low in saturated fats and sodium
            * Regular physical activity (at least 150 minutes per week)
            * Avoid smoking and excessive alcohol consumption
            * Manage stress through relaxation techniques

            **Note:** This prediction is for informational purposes only and should not replace professional medical advice.
            """)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif option == "📁 Upload CSV":
    st.subheader("📁 Upload CSV File for Batch Prediction")

    st.markdown("""
    Please upload a CSV file with the following columns:
    ```
    'Coronary Heart Disease History', 'CumulativeHealthConditions', 'Depression Diagnosis', 
    'General health status', 'Drank Alcohol Past 30 days', 'Body Mass Index(BMI)', 
    'Arthritis Status', 'Asthma Status', 'Sleep Hours per night', 'Height(m)', 
    'Stroke History', 'Physical activity Past 30 days', 'Skin Cancer History', 
    'AgeNumerical', 'Diabetes Status', 'COPD (lung disease) Status', 
    'Difficulty Walking/Climbing Stairs', 'Kidney Disease Status'
    ```

    For binary features, use 0 for No and 1 for Yes.
    """)

    # 文件上传
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # 检查缺失的列
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {missing_cols}")
                st.markdown("Your CSV file must contain all required columns. Please check your file and try again.")
            else:
                st.success("File uploaded successfully!")
                st.dataframe(df.head())

                if st.button("🔍 Run Batch Prediction", use_container_width=True):
                    # 创建进度条
                    progress_bar = st.progress(0)

                    # 应用特征标准化
                    normalized_df = normalize_features(df)

                    # 为每行添加预测
                    preds = model.predict(normalized_df)
                    probs = model.predict_proba(normalized_df)[:, 1]

                    # 更新进度条
                    progress_bar.progress(100)

                    # 添加预测结果到数据框
                    results_df = df.copy()
                    results_df['Prediction'] = preds
                    results_df['Risk_Probability'] = probs
                    results_df['Risk_Category'] = results_df['Risk_Probability'].apply(
                        lambda x: "Low Risk" if x < 0.3 else "Moderate Risk" if x < 0.7 else "High Risk"
                    )

                    # 显示结果
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)

                    # 显示统计信息
                    st.subheader("Summary Statistics")

                    # 统计不同风险类别的数量
                    risk_counts = results_df['Risk_Category'].value_counts()

                    # 以条形图显示
                    st.bar_chart(risk_counts)

                    # 提供下载结果的选项
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="heart_attack_risk_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# 添加页脚
st.markdown("---")
st.markdown("❤️ Heart Attack Risk Prediction Tool - For educational purposes only")