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
with st.expander("合約與帳務 ", expanded=True):
    contract = st.radio("合約類型", ['Month-to-Month', 'One Year', 'Two Year'])  # 注意大小寫
    tenure = st.number_input("使用月數", 0, 100, 10)
    monthly_charge = st.number_input("每月費用", 0.0, 500.0, 70.0)

with st.expander("服務項目 (有使用請勾選)", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        internet = st.checkbox("上網服務")
    with col2:
        online_security = st.checkbox("網路安全服務")
    with col3:
        streaming_movies = st.checkbox("串流電影服務")
    with col4:
        streaming_tv = st.checkbox("串流電視服務")

with st.expander("個人狀況", expanded=True):
    referrals = st.number_input("推薦人數", 0, 100, 1)
    dependents = st.number_input("扶養人數", 0, 10, 0)
    married = st.radio("是否已婚", ['Yes', 'No'])

# —— 建議留客策略文字（加粗字體18px）——
strategy_map = {
    "Contract": (
        "<div style='font-size:18px; font-weight:bold;'>延長合約優惠</div>"
        "［合約等級說明］<br>"
        "- 一等｜月租制<br>"
        "- 二等｜一年約<br>"
        "- 三等｜兩年約<br>"
        "- 四等｜三年以上合約<br><br>"
        "［優惠內容］<br>"
        "凡升級合約等級之用戶，依升級級數享月費折扣：<br>"
        "- 升級一級 → 月費 95 折<br>"
        "- 升級二級 → 月費 9 折<br>"
        "- 升級三級（即由月租升至三年以上）→ 成為終生會員，<br>"
        "  在服務內容不變的情況下，享永久固定資費續約"
    ),
    "Number of Referrals": (
        "<div style='font-size:18px; font-weight:bold;'>推薦新客戶獎勵</div>"
        "［適用對象］<br>"
        "所有現有用戶皆可參與推薦計畫<br>"
        "［獎勵內容］<br>"
        "每成功推薦 2 位新用戶，獲得一張 $5 現金抵用券，每多推薦一人，額外加贈一張（例如推薦 3 位 → 獲得 2 張，以此類推）。"
        "抵用券與推薦人數均無上限累積，每人每月限使用 1 張抵用券，抵用卷不可折現金"
    ),
    "Tenure in Months": (
        "<div style='font-size:18px; font-weight:bold;'>預繳補足升級福利</div>"
        "[適用對象] 累積使用合約未滿一年者<br>"
        "[優惠內容]<br>"
        "若合約累積使用時間未滿 12 個月，用戶可預繳剩餘月份，即刻享有年約福利。例如：已使用 8 個月，預繳 4 個月，即享「一年約」等級福利<br><br>"
        "福利包含但不限於：<br>"
        "專屬客服服務、加值服務折扣、限量活動邀約（如新品體驗、VIP日等）"
    ),
    "Number of Dependents": (
        "<div style='font-size:18px; font-weight:bold;'>寵物友善方案</div>"
        "［適用對象］無登記扶養人口但有寵物，且屬高流失風險用戶<br>"
        "［方案內容］<br>"
        "申辦「寵物友善加值流量方案」，可選擇（a)以優惠價購買寵物攝影機，或(b)綁約 30 個月以上享免費租用攝影機，"
        "此方案可搭配原主方案使用，透過遠端觀看功能提升情感連結與留存率"
    ),
    "Monthly Charge": (
        "<div style='font-size:18px; font-weight:bold;'>用戶預儲值優惠</div>"
        "［適用對象］月費偏高、價格敏感、高流失風險之用戶<br>"
        "［優惠內容］<br>"
        "一次性儲值 $350 元（或以上）即日起享 月費 85 折優惠。折扣自儲值當月起生效，持續至金額扣抵完畢為止。"
        "每位會員限參加一次，且儲值金額不可退費"
    )
}

# —— 預測與顯示結果 ——
if st.button("🔮 預測是否流失"):
    # 將布林值轉成模型訓練用的字串
    internet_str = "Yes" if internet else "No"
    online_security_str = "Yes" if online_security else "No"
    streaming_movies_str = "Yes" if streaming_movies else "No"
    streaming_tv_str = "Yes" if streaming_tv else "No"

    # 組成輸入 DataFrame
    input_dict = {
        'Contract': contract,
        'Internet Service': internet_str,
        'Number of Referrals': referrals,
        'Number of Dependents': dependents,
        'Married': married,
        'Streaming Movies': streaming_movies_str,
        'Streaming TV': streaming_tv_str,
        'Tenure in Months': tenure,
        'Online Security': online_security_str,
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

        feature_names = get_feature_names(pipeline.named_steps['preprocess'])
        X_trans = pipeline.named_steps['preprocess'].transform(input_df)
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

        explainer = shap.Explainer(pipeline.named_steps['clf'])
        shap_values = explainer(X_trans_df)

        vals = shap_values.values[0]
        pos_vals = np.where(vals > 0, vals, -np.inf)
        top3_idx = np.argsort(pos_vals)[-3:][::-1]
        top3_feats = [feature_names[i] for i in top3_idx]

        st.subheader("💡 建議的留客策略")

        default_strategy = (
            "<div style='font-size:18px; font-weight:bold;'>用戶預儲值優惠</div>"
            "［適用對象］月費偏高、價格敏感、高流失風險之用戶<br>"
            "［優惠內容］<br>"
            "一次性儲值 $350 元（或以上）即日起享 月費 85 折優惠。折扣自儲值當月起生效，持續至金額扣抵完畢為止。"
            "每位會員限參加一次，且儲值金額不可退費</div>"
        )

        for feat in top3_feats:
            txt = strategy_map.get(feat, default_strategy)
            st.markdown(txt, unsafe_allow_html=True)
            st.markdown('<hr style="border:1px solid #ccc;">', unsafe_allow_html=True)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        st.markdown("""
            **圖例說明：**  
            🔴 紅色：提升流失風險&nbsp; / &nbsp;🔵 藍色：降低流失風險
            """)
    else:
        st.success("✅ 穩定用戶，流失風險低")