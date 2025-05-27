
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Dự đoán Tác nhân và Gợi ý Kháng sinh", layout="wide")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

@st.cache_data
def load_model():
    df = pd.read_csv("AI_13_cleaned.csv")
    df = df[df["Tac nhan"] != "unspecified"]
    X = df.drop(columns=["Tac nhan"])
    y = df["Tac nhan"]

    for col in X.columns:
        if X[col].dropna().isin([0.0, 1.0]).all():
            X[col] = X[col].astype(int)

    X = X.dropna()
    y = y[X.index]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model, label_encoder, X.columns.tolist()

@st.cache_data
def load_ksd():
    return pd.read_csv("Mô hình KSD.csv")

model, encoder, features = load_model()
df_ksd = load_ksd()

st.header("📋 Nhập dữ liệu lâm sàng")
user_input = {}
for col in features:
    if col.lower() in ["sot", "ho", "non0 oi", "tieu chay", "kich thich0 quay khoc",
                       "tho ren0 met0 nhanh", "bo an0 an kem", "chay mui", "dam", "kho tho",
                       "kho khe", "ran phoi", "dong dac phoi phai", "dong dac phoi trai", "1 lom long nguc"]:
        user_input[col] = 1 if st.radio(col, ["Không", "Có"], horizontal=True) == "Có" else 0
    else:
        user_input[col] = st.number_input(col, value=0.0, format="%.2f")

if st.button("🔍 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    pred = model.predict(input_df)[0]
    pathogen = encoder.inverse_transform([pred])[0]
    st.success("✅ Tác nhân được dự đoán: **{}**".format(pathogen))

    st.subheader("💊 Gợi ý kháng sinh")
    if pathogen == "RSV":
        st.info("RSV là virus, không dùng kháng sinh.")
    elif pathogen == "M. pneumonia":
        st.write("Khuyến cáo: Macrolide (Azithromycin), Doxycycline, Levofloxacin")
    else:
        row = df_ksd[df_ksd["Tac nhan"] == pathogen]
        if not row.empty:
            khang_sinh = row.drop(columns=["Tac nhan"]).T
            ks_goi_y = khang_sinh[khang_sinh[row.index[0]] == 0.5].index.tolist()
            if ks_goi_y:
                st.write("Kháng sinh nhạy (gợi ý):")
                for ks in ks_goi_y:
                    st.markdown("- {}".format(ks))
            else:
                st.warning("Không có kháng sinh nào nhạy được ghi nhận.")
        else:
            st.error("Không tìm thấy dữ liệu kháng sinh đồ.")
