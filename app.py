import streamlit as st
import pickle
import pandas as pd

st.title("📉 顧客流失預測器")

# 載入模型
with open('model_pipeline_80.pkl', 'rb') as f:
    model_data = pickle.load(f)

pipeline = model_data['pipeline']
features = model_data['selected_features']

# 使用者輸入區
contract = st.selectbox("Contract 合約類型", ['Month-to-month', 'One year', 'Two year'])
internet = st.radio("Internet Service 是否有上網服務", ['Yes', 'No'])
referrals = st.number_input("Number of Referrals 推薦人數", min_value=0, value=1)
dependents = st.number_input("Number of Dependents 受扶養人數", min_value=0, value=0)
married = st.radio("Married 是否已婚", ['Yes', 'No'])
streaming_movies = st.radio("Streaming Movies 是否有看電影服務", ['Yes', 'No'])
streaming_tv = st.radio("Streaming TV 是否有看電視服務", ['Yes', 'No'])
tenure = st.slider("Tenure in Months 使用月數", min_value=0, max_value=100, value=12)
online_security = st.radio("Online Security 是否有網路安全服務", ['Yes', 'No'])
monthly_charge = st.number_input("Monthly Charge 每月費用", min_value=0.0, value=70.0)

# 預測按鈕
if st.button("🔮 預測是否流失"):
    input_dict = {
        'Contract': contract,
        'Internet Service': internet,
        'Number of Referrals': referrals,
        'Number of Dependents': dependents,
        'Married': married,
        'Streaming Movies': streaming_movies,
        'Streaming TV': streaming_tv,
        'Tenure in Months': tenure,
        'Online Security': online_security,
        'Monthly Charge': monthly_charge
    }

    input_df = pd.DataFrame([input_dict])[features]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.subheader("🧾 預測結果")
    st.write(f"顧客流失機率為：**{prob:.2%}**")
    if prob > 0.5:
        st.warning("⚠️ 高風險用戶，建議主動聯繫留客")
    else:
        st.success("✅ 穩定用戶，流失風險低")