import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Dataset Overview & EDA")

# Dataset Source
st.subheader("Dataset Source")
st.markdown("""
- Source: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2022  
- [Official CDC Website](https://www.cdc.gov/brfss/annual_data/annual_2022.html)
- This dataset is derived from the CDC BRFSS 2022 survey. A total of 28 variables were selected by the team to explore heart attack risk.
""")

# 数据加载（你需要将该CSV放在项目根目录或 data/ 文件夹下）
@st.cache_data
def load_data():
    return pd.read_csv("data/v2heartattack_finaldataset.csv")

df = load_data()

# 显示数据维度
st.write("### Dataset Preview")
st.write(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

# 显示列名（可折叠）
with st.expander("📌 Feature List"):
    for col in df.columns:
        st.markdown(f"- {col}")

# 分类变量 - 分布柱状图（提取 object 类型）
st.write("### 📊 Categorical Feature Distributions")
categorical_cols = df.select_dtypes(include='object').columns

selected_cat = st.selectbox("Choose a categorical column to view distribution:", categorical_cols)
fig, ax = plt.subplots()
df[selected_cat].value_counts(dropna=False).plot(kind="bar", color="green", ax=ax)
ax.set_title(f"Value Counts: {selected_cat}")
ax.set_ylabel("Count")
st.pyplot(fig)

# 数值变量直方图
st.write("### 📈 Numeric Feature Distribution")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

selected_num = st.selectbox("Choose a numeric column:", numeric_cols)
fig2, ax2 = plt.subplots()
df[selected_num].plot(kind='hist', bins=30, color='maroon', edgecolor='black', ax=ax2)
ax2.set_title(f"Histogram: {selected_num}")
ax2.set_xlabel("Value")
st.pyplot(fig2)

# 箱线图（数值变量离群值检测）
st.write("### 📦 Boxplot of Numeric Feature")
fig3, ax3 = plt.subplots(figsize=(6, 2))
sns.boxplot(x=df[selected_num], ax=ax3, color='orange')
ax3.set_title(f"Boxplot: {selected_num}")
st.pyplot(fig3)

# 相关性热力图（数值型）
st.write("### 🔥 Correlation Matrix (Numerical)")
corr = df[numeric_cols].corr()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)
