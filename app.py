import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px

st.set_page_config(
    page_title="AI Student Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    background-color: #1e3a8a;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

.result-pass {
    background-color: #dcfce7;
    color: #065f46;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid green;
}

.result-fail {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid red;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Hệ thống Dự đoán Kết quả Học tập Sinh viên")

@st.cache_data
def load_data():

    np.random.seed(42)

    data = pd.DataFrame({
        'studytime': np.random.randint(1, 5, 649),
        'failures': np.random.randint(0, 4, 649),
        'absences': np.random.randint(0, 30, 649),
        'goout': np.random.randint(1, 5, 649),
        'internet': np.random.choice([0,1], 649),
        'G1': np.random.randint(0, 20, 649),
        'G2': np.random.randint(0, 20, 649),
        'Medu': np.random.randint(0, 5, 649),
        'age': np.random.randint(15, 22, 649)
    })

    score = (
        data['G1'] * 0.3 +
        data['G2'] * 0.4 +
        data['studytime'] * 1.5 -
        data['failures'] * 2 -
        data['absences'] * 0.1
    )

    data['result'] = np.where(score >= 10, 1, 0)

    return data

@st.cache_resource
def train_model():

    data = load_data()

    X = data.drop("result", axis=1)
    y = data["result"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model()

tab1, tab2, tab3 = st.tabs([
    "📌 Dự đoán",
    "📊 Phân tích",
    "📂 Dự đoán hàng loạt"
])

with tab1:

    st.sidebar.header("📥 Nhập thông tin sinh viên")

    studytime = st.sidebar.slider("📚 Thời gian học", 1, 4, 2)
    failures = st.sidebar.slider("❌ Số lần trượt", 0, 4, 0)
    absences = st.sidebar.slider("📅 Số buổi vắng", 0, 30, 5)
    goout = st.sidebar.slider("🎉 Mức độ đi chơi", 1, 5, 2)
    internet = st.sidebar.selectbox("🌐 Có Internet", [0,1])
    G1 = st.sidebar.slider("📝 Điểm học kỳ 1", 0, 20, 10)
    G2 = st.sidebar.slider("📝 Điểm học kỳ 2", 0, 20, 10)
    Medu = st.sidebar.slider("👩 Học vấn của mẹ", 0, 4, 2)
    age = st.sidebar.slider("🎂 Tuổi", 15, 22, 18)

    input_data = pd.DataFrame({
        'studytime':[studytime],
        'failures':[failures],
        'absences':[absences],
        'goout':[goout],
        'internet':[internet],
        'G1':[G1],
        'G2':[G2],
        'Medu':[Medu],
        'age':[age]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 Kết quả dự đoán")

    if prediction == 1:

        st.markdown(f"""
        <div class="result-pass">
        <h2>✅ PASS</h2>
        <h3>Xác suất đậu: {probability[1]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-fail">
        <h2>❌ FAIL</h2>
        <h3>Nguy cơ rớt: {probability[0]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': importance
    }).sort_values(by="Importance", ascending=False)

    st.subheader("📈 Mức độ ảnh hưởng của các yếu tố")

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader("📊 Đánh giá mô hình")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Dự đoán", y="Thực tế"),
        x=['Fail', 'Pass'],
        y=['Fail', 'Pass']
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    scatter_df = pd.DataFrame({
        "Thực tế": y_test,
        "Dự đoán": y_pred
    })

    fig_scatter = px.scatter(
        scatter_df,
        x="Thực tế",
        y="Dự đoán"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:

    st.subheader("📂 Upload CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file:

        batch_data = pd.read_csv(uploaded_file)

        predictions = model.predict(batch_data)
        probabilities = model.predict_proba(batch_data)

        batch_data["Prediction"] = predictions
        batch_data["Pass_Probability"] = probabilities[:,1]

        st.dataframe(batch_data)

        csv = batch_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Result CSV",
            csv,
            "prediction_result.csv",
            "text/csv"
        )
.result-fail {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid red;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Hệ thống Dự đoán Kết quả Học tập Sinh viên")

@st.cache_data
def load_data():

    np.random.seed(42)

    data = pd.DataFrame({
        'studytime': np.random.randint(1, 5, 649),
        'failures': np.random.randint(0, 4, 649),
        'absences': np.random.randint(0, 30, 649),
        'goout': np.random.randint(1, 5, 649),
        'internet': np.random.choice([0,1], 649),
        'G1': np.random.randint(0, 20, 649),
        'G2': np.random.randint(0, 20, 649),
        'Medu': np.random.randint(0, 5, 649),
        'age': np.random.randint(15, 22, 649)
    })

    score = (
        data['G1'] * 0.3 +
        data['G2'] * 0.4 +
        data['studytime'] * 1.5 -
        data['failures'] * 2 -
        data['absences'] * 0.1
    )

    data['result'] = np.where(score >= 10, 1, 0)

    return data

@st.cache_resource
def train_model():

    data = load_data()

    X = data.drop("result", axis=1)
    y = data["result"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model()

tab1, tab2, tab3 = st.tabs([
    "📌 Dự đoán",
    "📊 Phân tích",
    "📂 Dự đoán hàng loạt"
])

with tab1:

    st.sidebar.header("📥 Nhập thông tin sinh viên")

    studytime = st.sidebar.slider("📚 Thời gian học", 1, 4, 2)
    failures = st.sidebar.slider("❌ Số lần trượt", 0, 4, 0)
    absences = st.sidebar.slider("📅 Số buổi vắng", 0, 30, 5)
    goout = st.sidebar.slider("🎉 Mức độ đi chơi", 1, 5, 2)
    internet = st.sidebar.selectbox("🌐 Có Internet", [0,1])
    G1 = st.sidebar.slider("📝 Điểm học kỳ 1", 0, 20, 10)
    G2 = st.sidebar.slider("📝 Điểm học kỳ 2", 0, 20, 10)
    Medu = st.sidebar.slider("👩 Học vấn của mẹ", 0, 4, 2)
    age = st.sidebar.slider("🎂 Tuổi", 15, 22, 18)

    input_data = pd.DataFrame({
        'studytime':[studytime],
        'failures':[failures],
        'absences':[absences],
        'goout':[goout],
        'internet':[internet],
        'G1':[G1],
        'G2':[G2],
        'Medu':[Medu],
        'age':[age]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 Kết quả dự đoán")

    if prediction == 1:

        st.markdown(f"""
        <div class="result-pass">
        <h2>✅ PASS</h2>
        <h3>Xác suất đậu: {probability[1]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-fail">
        <h2>❌ FAIL</h2>
        <h3>Nguy cơ rớt: {probability[0]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': importance
    }).sort_values(by="Importance", ascending=False)

    st.subheader("📈 Mức độ ảnh hưởng của các yếu tố")

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader("📊 Đánh giá mô hình")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Dự đoán", y="Thực tế"),
        x=['Fail', 'Pass'],
        y=['Fail', 'Pass']
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    scatter_df = pd.DataFrame({
        "Thực tế": y_test,
        "Dự đoán": y_pred
    })

    fig_scatter = px.scatter(
        scatter_df,
        x="Thực tế",
        y="Dự đoán"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:

    st.subheader("📂 Upload CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file:

        batch_data = pd.read_csv(uploaded_file)

        predictions = model.predict(batch_data)
        probabilities = model.predict_proba(batch_data)

        batch_data["Prediction"] = predictions
        batch_data["Pass_Probability"] = probabilities[:,1]

        st.dataframe(batch_data)

        csv = batch_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Result CSV",
            csv,
            "prediction_result.csv",
            "text/csv"
        )
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 25px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

[data-testid="stMarkdownContainer"] {
    color: white;
}

.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.result-pass {
    background: rgba(34,197,94,0.25);
    color: #dcfce7;
    padding: 20px;
    border-radius: 20px;
    border: 2px solid #22c55e;
    text-align: center;
    backdrop-filter: blur(10px);
}

.result-fail {
    background: rgba(239,68,68,0.25);
    color: #fee2e2;
    padding: 20px;
    border-radius: 20px;
    border: 2px solid #ef4444;
    text-align: center;
    backdrop-filter: blur(10px);
}

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.15);
    color: white;
}

.stSlider label {
    color: white !important;
}

[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 15px;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.title("🎓 Hệ thống Dự đoán Kết quả Học tập Sinh viên")

@st.cache_data
def load_data():

    np.random.seed(42)

    data = pd.DataFrame({
        'studytime': np.random.randint(1, 5, 649),
        'failures': np.random.randint(0, 4, 649),
        'absences': np.random.randint(0, 30, 649),
        'goout': np.random.randint(1, 5, 649),
        'internet': np.random.choice([0,1], 649),
        'G1': np.random.randint(0, 20, 649),
        'G2': np.random.randint(0, 20, 649),
        'Medu': np.random.randint(0, 5, 649),
        'age': np.random.randint(15, 22, 649)
    })

    score = (
        data['G1'] * 0.3 +
        data['G2'] * 0.4 +
        data['studytime'] * 1.5 -
        data['failures'] * 2 -
        data['absences'] * 0.1
    )

    data['result'] = np.where(score >= 10, 1, 0)

    return data

@st.cache_resource
def train_model():

    data = load_data()

    X = data.drop("result", axis=1)
    y = data["result"]

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model()

tab1, tab2, tab3 = st.tabs([
    "📌 Dự đoán",
    "📊 Phân tích",
    "📂 Dự đoán hàng loạt"
])

with tab1:

    st.sidebar.header("📥 Nhập thông tin sinh viên")

    studytime = st.sidebar.slider(
        "📚 Thời gian học",
        1, 4, 2
    )

    failures = st.sidebar.slider(
        "❌ Số lần trượt",
        0, 4, 0
    )

    absences = st.sidebar.slider(
        "📅 Số buổi vắng",
        0, 30, 5
    )

    goout = st.sidebar.slider(
        "🎉 Mức độ đi chơi",
        1, 5, 2
    )

    internet = st.sidebar.selectbox(
        "🌐 Có Internet",
        [0,1]
    )

    G1 = st.sidebar.slider(
        "📝 Điểm học kỳ 1",
        0, 20, 10
    )

    G2 = st.sidebar.slider(
        "📝 Điểm học kỳ 2",
        0, 20, 10
    )

    Medu = st.sidebar.slider(
        "👩 Học vấn của mẹ",
        0, 4, 2
    )

    age = st.sidebar.slider(
        "🎂 Tuổi",
        15, 22, 18
    )

    input_data = pd.DataFrame({
        'studytime':[studytime],
        'failures':[failures],
        'absences':[absences],
        'goout':[goout],
        'internet':[internet],
        'G1':[G1],
        'G2':[G2],
        'Medu':[Medu],
        'age':[age]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 Kết quả dự đoán")

    if prediction == 1:

        st.markdown(f"""
        <div class="result-pass">
            <h2>✅ PASS</h2>
            <h3>Xác suất đậu: {probability[1]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-fail">
            <h2>❌ FAIL</h2>
            <h3>Nguy cơ rớt: {probability[0]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Yếu tố': input_data.columns,
        'Mức độ ảnh hưởng': importance
    }).sort_values(
        by="Mức độ ảnh hưởng",
        ascending=False
    )

    st.subheader("📈 Mức độ ảnh hưởng của các yếu tố")

    fig = px.bar(
        importance_df,
        x='Mức độ ảnh hưởng',
        y='Yếu tố',
        orientation='h'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader("📊 Đánh giá mô hình")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Dự đoán",
            y="Thực tế"
        ),
        x=['Fail', 'Pass'],
        y=['Fail', 'Pass']
    )

    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

with tab3:

    st.subheader("📂 Dự đoán bằng file CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file:

        batch_data = pd.read_csv(uploaded_file)

        predictions = model.predict(batch_data)

        probabilities = model.predict_proba(batch_data)

        batch_data["Prediction"] = predictions

        batch_data["Pass_Probability"] = probabilities[:,1]

        st.dataframe(batch_data)

        csv = batch_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Result CSV",
            csv,
            "prediction_result.csv",
            "text/csv"
        )    border-radius: 20px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    background-color: #1e3a8a;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

.result-pass {
    background-color: #dcfce7;
    color: #065f46;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid green;
}

.result-fail {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid red;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Hệ thống Dự đoán Kết quả Học tập Sinh viên")

@st.cache_data
def load_data():

    np.random.seed(42)

    data = pd.DataFrame({
        'studytime': np.random.randint(1, 5, 649),
        'failures': np.random.randint(0, 4, 649),
        'absences': np.random.randint(0, 30, 649),
        'goout': np.random.randint(1, 5, 649),
        'internet': np.random.choice([0,1], 649),
        'G1': np.random.randint(0, 20, 649),
        'G2': np.random.randint(0, 20, 649),
        'Medu': np.random.randint(0, 5, 649),
        'age': np.random.randint(15, 22, 649)
    })

    score = (
        data['G1'] * 0.3 +
        data['G2'] * 0.4 +
        data['studytime'] * 1.5 -
        data['failures'] * 2 -
        data['absences'] * 0.1
    )

    data['result'] = np.where(score >= 10, 1, 0)

    return data

@st.cache_resource
def train_model():

    data = load_data()

    X = data.drop("result", axis=1)
    y = data["result"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model()

tab1, tab2, tab3 = st.tabs([
    "📌 Dự đoán",
    "📊 Phân tích",
    "📂 Dự đoán hàng loạt"
])

with tab1:

    st.sidebar.header("📥 Nhập thông tin sinh viên")

    studytime = st.sidebar.slider("📚 Thời gian học", 1, 4, 2)
    failures = st.sidebar.slider("❌ Số lần trượt", 0, 4, 0)
    absences = st.sidebar.slider("📅 Số buổi vắng", 0, 30, 5)
    goout = st.sidebar.slider("🎉 Mức độ đi chơi", 1, 5, 2)
    internet = st.sidebar.selectbox("🌐 Có Internet", [0,1])
    G1 = st.sidebar.slider("📝 Điểm học kỳ 1", 0, 20, 10)
    G2 = st.sidebar.slider("📝 Điểm học kỳ 2", 0, 20, 10)
    Medu = st.sidebar.slider("👩 Học vấn của mẹ", 0, 4, 2)
    age = st.sidebar.slider("🎂 Tuổi", 15, 22, 18)

    input_data = pd.DataFrame({
        'studytime':[studytime],
        'failures':[failures],
        'absences':[absences],
        'goout':[goout],
        'internet':[internet],
        'G1':[G1],
        'G2':[G2],
        'Medu':[Medu],
        'age':[age]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 Kết quả dự đoán")

    if prediction == 1:

        st.markdown(f"""
        <div class="result-pass">
        <h2>✅ PASS</h2>
        <h3>Xác suất đậu: {probability[1]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-fail">
        <h2>❌ FAIL</h2>
        <h3>Nguy cơ rớt: {probability[0]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': importance
    }).sort_values(by="Importance", ascending=False)

    st.subheader("📈 Mức độ ảnh hưởng của các yếu tố")

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader("📊 Đánh giá mô hình")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Dự đoán", y="Thực tế"),
        x=['Fail', 'Pass'],
        y=['Fail', 'Pass']
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    scatter_df = pd.DataFrame({
        "Thực tế": y_test,
        "Dự đoán": y_pred
    })

    fig_scatter = px.scatter(
        scatter_df,
        x="Thực tế",
        y="Dự đoán"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:

    st.subheader("📂 Upload CSV")

    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file:

        batch_data = pd.read_csv(uploaded_file)

        predictions = model.predict(batch_data)
        probabilities = model.predict_proba(batch_data)

        batch_data["Prediction"] = predictions
        batch_data["Pass_Probability"] = probabilities[:,1]

        st.dataframe(batch_data)

        csv = batch_data.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Result CSV",
            csv,
            "prediction_result.csv",
            "text/csv"
        )
