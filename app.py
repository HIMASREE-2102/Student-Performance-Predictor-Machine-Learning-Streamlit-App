# ============================================================
# ADVANCED STUDENT PERFORMANCE PREDICTOR 🎓
# ============================================================
# Features:
# ✅ More input features (stress, social media, etc.)
# ✅ Model performance metrics and visualizations
# ✅ Interactive Streamlit interface with graphs
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ============================================================
# STEP 1: CREATE SYNTHETIC DATASET
# ============================================================
def create_sample_data(csv_path="students.csv", n=400):
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(15, 22, size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
        "study_time_hours_per_week": np.random.normal(10, 3, n).clip(0),
        "attendance_percent": np.random.randint(60, 100, size=n),
        "previous_grade": np.random.randint(40, 100, size=n),
        "parental_education": np.random.choice(["HighSchool", "Bachelor", "Master", "Other"], size=n),
        "extracurricular": np.random.choice(["Yes", "No"], size=n),
        "sleep_hours": np.random.normal(7, 1, n).clip(4, 10),
        "internet_access": np.random.choice(["Yes", "No"], size=n),
        "family_support": np.random.choice(["Low", "Medium", "High"], size=n),
        "stress_level": np.random.randint(1, 10, size=n),
        "social_media_hours": np.random.normal(3, 1.5, n).clip(0, 10),
        "physical_activity_hours": np.random.normal(2, 1, n).clip(0, 5)
    })
    # Target variable (final score)
    df["final_score"] = (
        0.25 * df["previous_grade"] +
        0.2 * df["attendance_percent"] +
        2 * df["study_time_hours_per_week"] +
        np.where(df["extracurricular"] == "Yes", 2, -2) +
        np.where(df["family_support"] == "High", 3, 0) +
        np.where(df["internet_access"] == "Yes", 1, -1) -
        1.5 * df["stress_level"] -
        0.8 * df["social_media_hours"] +
        1.5 * df["physical_activity_hours"] +
        np.random.normal(0, 5, n)
    ).clip(0, 100)
    df.to_csv(csv_path, index=False)
    return df


# ============================================================
# STEP 2: TRAIN MODEL & SAVE PIPELINE
# ============================================================
def train_and_save_model(data_path="students.csv", model_path="student_pipeline.joblib"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["final_score"])
    y = df["final_score"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    model = RandomForestRegressor(random_state=42, n_estimators=250)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.session_state["metrics"] = {"RMSE": rmse, "R²": r2}

    joblib.dump(pipeline, model_path)
    print(f"✅ Model trained — RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return df


# ============================================================
# STEP 3: STREAMLIT APP
# ============================================================
def run_streamlit_app(model_path="student_pipeline.joblib"):
    st.set_page_config(page_title="Student Performance Predictor 🎓", layout="wide")
    st.title("🎓 Student Performance Predictor")
    st.caption("An AI-powered app to predict student performance based on study, lifestyle, and habits.")

    # Sidebar Navigation
    page = st.sidebar.radio("Navigate", ["📊 Predict Score", "📈 Model Insights", "📘 About Project"])

    if page == "📊 Predict Score":
        pipeline = joblib.load(model_path)
        st.header("🧠 Enter Student Details")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 10, 25, 18)
            gender = st.selectbox("Gender", ["Male", "Female"])
            study_time = st.number_input("Study Time (hours/week)", 0.0, 50.0, 10.0)
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            previous_grade = st.number_input("Previous Grade (%)", 0.0, 100.0, 70.0)
            parental_education = st.selectbox("Parental Education", ["HighSchool", "Bachelor", "Master", "Other"])
        with col2:
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
            sleep_hours = st.number_input("Sleep Hours (per day)", 4.0, 12.0, 7.0)
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            family_support = st.selectbox("Family Support", ["Low", "Medium", "High"])
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            social_media = st.slider("Social Media Use (hrs/day)", 0.0, 10.0, 3.0)
            physical_activity = st.slider("Physical Activity (hrs/day)", 0.0, 5.0, 1.5)

        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "study_time_hours_per_week": study_time,
            "attendance_percent": attendance,
            "previous_grade": previous_grade,
            "parental_education": parental_education,
            "extracurricular": extracurricular,
            "sleep_hours": sleep_hours,
            "internet_access": internet_access,
            "family_support": family_support,
            "stress_level": stress_level,
            "social_media_hours": social_media,
            "physical_activity_hours": physical_activity
        }])

        if st.button("🎯 Predict Score"):
            prediction = pipeline.predict(input_df)[0]
            st.success(f"Predicted Final Score: **{prediction:.2f} / 100**")

            if prediction < 50:
                st.error("⚠️ Low predicted score. Suggest more study time and better attendance.")
            elif prediction < 75:
                st.warning("🙂 Moderate performance. Consistency and balance can improve results.")
            else:
                st.balloons()
                st.success("🌟 Excellent performance predicted!")

    elif page == "📈 Model Insights":
        st.header("📊 Model Insights and Data Visualization")
        df = pd.read_csv("students.csv")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Feature vs Final Score")
            feature = st.selectbox("Select Feature", df.columns[:-1])
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=df[feature], y=df["final_score"], alpha=0.7)
            st.pyplot(fig2)

        if "metrics" in st.session_state:
            st.info(f"Model Performance — RMSE: {st.session_state['metrics']['RMSE']:.2f}, R²: {st.session_state['metrics']['R²']:.2f}")

    else:
        st.header("📘 About the Project")
        st.markdown("""
        This **Student Performance Predictor** uses machine learning to estimate a student's final exam score 
        based on academic, behavioral, and lifestyle factors.

        **Key Features:**
        - Predicts performance from real-world factors  
        - Interactive input interface  
        - Visualization of model insights and correlations  
        - Powered by Random Forest Regression & Streamlit  

        💡 *Built for students, teachers, and education researchers to gain insights into learning outcomes.*
        """)


# ============================================================
# MAIN EXECUTION LOGIC
# ============================================================
if __name__ == "__main__":
    if not os.path.exists("students.csv"):
        create_sample_data()
    if not os.path.exists("student_pipeline.joblib"):
        train_and_save_model()
    run_streamlit_app()
