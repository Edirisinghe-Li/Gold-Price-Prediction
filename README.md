# 📈 Gold Price Predictor

A machine learning web application that predicts gold closing prices based on market features (Open, High, Low, and Volume). This project uses a **Random Forest Regressor** and is served via a **Streamlit** web interface.

## 🚀 Project Overview
This project was developed to provide a lightweight yet accurate tool for predicting gold market trends. 
- **Model Accuracy:** ~90% (R² Score)
- **Model Size:** 12MB (Optimized for fast deployment)
- **Technologies:** Python, Scikit-Learn, Pandas, Streamlit

---

## 💻 Instructions for Use (New PC Setup)

Follow these steps to get the project running on a new machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** or **Anaconda** installed on your system.

### 2. Clone the Repository
Open your terminal (or Git Bash) and run:

git clone [https://github.com/Sonal-Github1/Gold-Price-Prediction.git](https://github.com/Sonal-Github1/Gold-Price-Prediction.git)
cd Gold-Price-Prediction

3. Set Up a Virtual Environment (Recommended)

# Using Python
python -m venv env
.\env\Scripts\activate

# OR Using Anaconda
conda create -n gold-project python=3.9
conda activate gold-project

pip install pandas numpy scikit-learn streamlit

**streamlit run app.py**

🛠️ Model Optimization Note
Originally, the trained model file was over 240MB. To ensure the project was suitable for GitHub and web deployment, the model was optimized using tree pruning (limiting max_depth to 10).

Result: File size reduced by 95% while maintaining a high accuracy of 90%.
