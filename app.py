import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

st.title("📉 顧客流失預測器")

# —— 載入模型 Pipeline ——
with open('model_pipeline_80.pkl', 'rb') as f:
    data = pickle.load(f)
    pipeline = data['pipeline']
    features = data['selected_features']

# —— 取得特徵名稱的輔助函數 ——
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    out = transformer.get_feature_names_out(cols)
                except Exception:
                    out = cols
            else:
                out = cols
            feature_names.extend(out)
    if column_transformer.remainder == 'passthrough':
        try:
            all_feats = list(column_transformer.feature_names_in_)
        except AttributeError:
            all_feats = list(column_transformer._feature_names_in)
        for f in all_feats:
            if f not in feature_names:
                feature_names.append(f)
    return feature_names

# —— 使用者輸入 ——
contract = st.selectbox("Contract 合約類型", ['Month-to-month', 'One year', 'Two year'])
internet = st.radio("Internet Service 是否有上網服務", ['Yes', 'No'])
referrals = st.number_input("Number of Referrals 推薦人數", 0, 100, 1)
dependents = st.number_input("Number of Dependents 受扶養人數", 0, 10, 0)
married = st.radio("Married 是否已婚", ['Yes', 'No'])
streaming_movies = st.radio("Streaming Movies 是否有看電影服務", ['Yes', 'No'])
streaming_tv = st.radio("Streaming TV 是否有看電視服務", ['Yes', 'No'])
tenure = st.slider("Tenure in Months 使用月數", 0, 100, 12)
online_security = st.radio("Online Security 是否有網路安全服務", ['Yes', 'No'])
monthly_charge = st.number_input("Monthly Charge 每月費用", 0.0, 500.0, 70.0)

# —— 預測與顯示結果 ——
if st.button("🔮 預測是否流失"):
    # 組成輸入 DataFrame
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

    # 模型機率預測
    prob = pipeline.predict_proba(input_df)[0][1]
    st.subheader("🧾 預測結果")
    st.markdown(
        f"<div style='font-size:24px; font-weight:bold;'>顧客流失機率為：{prob:.2%}</div>",
        unsafe_allow_html=True
    )

    if prob > 0.5:
        st.warning("⚠️ 高風險用戶，建議主動聯繫留客")

        # 前處理後轉 DataFrame
        feature_names = get_feature_names(pipeline.named_steps['preprocess'])
        X_trans = pipeline.named_steps['preprocess'].transform(input_df)
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

        # SHAP 解釋 (原生方式)
        explainer = shap.Explainer(pipeline.named_steps['clf'])
        shap_values = explainer(X_trans_df)

        # 擷取前三正向 SHAP 特徵
        vals = shap_values.values[0]
        pos_vals = np.where(vals > 0, vals, -np.inf)
        top3_idx = np.argsort(pos_vals)[-3:][::-1]
        top3_feats = [feature_names[i] for i in top3_idx]

        # 策略對照表
        strategy_map = {
            "Contract": (
                "合約類型：延長合約優惠\n"
                "續約升級優惠：續約一年升一級享 95 折，續約兩年升兩級享 9 折，"
                "達指定級數後升為終生會員，享永久折扣與專屬禮遇。"
            ),
            "Number of Referrals": (
                "推薦人數：推薦新客戶優惠\n"
                "每推薦 2 位以上新客戶即獲得優惠券一張，推薦不限次數，累積可兌換服務折扣。"
            ),
            "Number of Dependents": (
                "扶養人口：(針對無扶養人口的用戶) 寵物套餐\n"
                "凡申辦寵物加值方案，即享「寵物攝影機」設備優惠價（或免費租用），"
                "搭配主方案使用可遠端觀看。"
            ),
            "Tenure in Months": (
                "累計合約期間：預付費用滿12個月\n"
                "預繳費用（滿12 個月）立即升級 VIP 會員，享專屬客服、加值服務與每月驚喜禮，"
                "並優先參與內部活動。"
            ),
            "Monthly Charge": (
                "月費：大眾方案\n"
                "一次儲值 $350 元，可自當月起享月費 85 折優惠，"
                "優惠持續至儲值金額使用完畢為止，適合長期使用者。"
            )
        }


        # 顯示策略文字
        st.subheader("💡 建議的留客策略")
        for feat in top3_feats:
            txt = strategy_map.get(feat, "🔍 尚未設定此欄位的策略")
            st.markdown(f"**{feat}** ➜ {txt}")

        # 繪製 SHAP waterfall
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.success("✅ 穩定用戶，流失風險低")
