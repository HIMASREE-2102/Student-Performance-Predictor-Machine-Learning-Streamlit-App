# 🎓 Student Performance Predictor

This **Streamlit web application** predicts student performance using a **Machine Learning (Random Forest)** model.  
It analyzes academic and behavioral factors such as study time, attendance, family support, and stress levels to forecast a student’s performance score.

---

## ✨ Features
- 📊 Predicts student scores using a trained ML model  
- 🔍 Visualizes correlations between features (attendance, study time, etc.)  
- 🧠 Explains predictions using SHAP interpretability  
- 💻 Interactive and user-friendly Streamlit interface  
- 📁 Automatically retrains and saves the model for new datasets  

---

## ⚙️ Installation and Running Locally

Follow the steps below to run the project on your local system.

### 1️⃣ Clone the Repository
```bash
2️⃣ Install the Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py


Then open the provided local URL in your browser (usually http://localhost:8501
).

🧠 Tech Stack

Frontend: Streamlit

Backend: Python

Machine Learning: Scikit-Learn (RandomForestRegressor)

Visualization: Matplotlib, SHAP

Data Handling: Pandas, NumPy

📁 Project Structure
performance_predictor/
│
├── app.py                  # Main Streamlit application
├── student_pipeline.joblib # Saved machine learning model
├── students.csv            # Dataset used for training/testing
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🚀 Model Description

The model used is a Random Forest Regressor, which provides high accuracy and handles both linear and non-linear relationships.

Input Features:

Study time

Attendance

Parental education

Sleep hours

Internet access

Extracurricular activities

Family support

Output:

Predicted performance score (0–100)

Evaluation Metric:

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

🧾 Example Use

Open the Streamlit web app.

Enter student details such as study time, attendance, and sleep hours.

Click Predict Performance.

The app will display the predicted score with insights and visual feedback.

📈 Results and Insights

Students with regular study habits and consistent attendance show higher predicted scores.

Poor sleep or lack of family support negatively impacts performance.

The model achieved RMSE below 5 and strong R² values on test data.

🧩 Future Enhancements

Integrate with school/college databases for real data input.

Deploy to Streamlit Cloud or AWS EC2 for public access.

Add classification mode (Pass/Fail prediction).

Introduce explainable AI dashboards for detailed feature insights.

🧑‍💻 Author

Hima Sree
B.Tech | Artificial Intelligence Project Developer

GitHub: https://github.com/HIMASREE-2102
git clone https://github.com/YOUR-USERNAME/performance_predictor.git
cd performance_predictor
