# Churn Predictor

A high-accuracy customer churn prediction solution using pandas, scikit-learn, and XGBoost, featuring robust preprocessing, advanced feature engineering, and a fast, interactive Streamlit web app.

---

## 🚀 Features

- **93% Accuracy:** Predicts customer churn with high precision using 10,000+ records.
- **Advanced Feature Engineering:** 25+ engineered features capturing key customer behaviors.
- **Robust Preprocessing:** Comprehensive data cleaning, normalization, and encoding.
- **Hyperparameter Tuning:** Achieved 18% performance boost through systematic optimization.
- **Streamlit Deployment:** User-friendly web interface with real-time (<2s) inference.

---

## 🏗️ Architecture Overview

- **Data Handling:** pandas for data loading, cleaning, and transformation.
- **Modeling:** XGBoost classifier with scikit-learn pipeline for preprocessing and tuning.
- **Feature Engineering:** Custom scripts for extracting and transforming relevant features.
- **Deployment:** Streamlit app for seamless, interactive predictions.

---

## 📦 Installation

1. **Clone the Repository**
    ```
    git clone <repo-url>
    cd churn-predictor
    ```

2. **Set Up Virtual Environment**
    ```
    python -m venv env
    source env/bin/activate
    ```

3. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage

1. **Start the Streamlit App**
    ```
    streamlit run app.py
    ```

2. **Interact with the Predictor**
    - Input or upload customer data via the web interface.
    - Get instant churn predictions and feature insights.

---

## ⚙️ How It Works

1. **Data Preprocessing**
    - Cleans and transforms raw data.
    - Extracts 25+ behavioral and demographic features.

2. **Model Training**
    - XGBoost classifier with hyperparameter optimization (grid/random search).
    - Evaluated using accuracy, F1, precision, and recall.

3. **Deployment**
    - Streamlit app loads trained model and processes user input for real-time inference.

---

## 🛠️ Key Technologies

| Component          | Technology           |
|--------------------|---------------------|
| Data Processing    | pandas              |
| Machine Learning   | scikit-learn, XGBoost|
| Deployment         | Streamlit           |

---

## 🔧 Customization

- **Feature Engineering:** Edit scripts in `features/` for new or improved features.
- **Model Tuning:** Adjust parameters in `config.yaml` or model training scripts.
- **UI:** Modify `app.py` for custom interface or additional visualizations.

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgments

Thanks to the open-source community for pandas, scikit-learn, XGBoost, and Streamlit.
