
## 🩺 Blood Glucose Prediction and Diabetes Risk Assessment

**Non-Invasive Glucose Estimation using Machine Learning**

<img width="1280" height="665" alt="image" src="https://github.com/user-attachments/assets/00d9512d-dbc8-4b4d-8a31-3346636053e0" />
<img width="1280" height="651" alt="image" src="https://github.com/user-attachments/assets/12254ef9-1b35-4e03-9918-038b81a2ffea" />



---

### 📌 Project Overview

This project focuses on building a predictive machine learning pipeline that estimates **blood glucose levels** and assesses **diabetes risk** using non-invasive sweat sensor data and patient biometrics. Designed for real-world healthcare applications, the system handles noisy biomedical inputs, performs robust feature engineering, and delivers predictions through an interactive web app.

---

### 🔧 Key Features

* 💧 Non-invasive glucose level estimation using **sweat sensor currents (µA)**
* 🏥 Personalized risk profiling using patient historical biometrics
* 🔎 Data cleaning with custom transformers and encoders
* ⚙️ Hyperparameter tuning and cross-validation
* 🌐 Real-time deployment on **Streamlit + AWS Elastic Beanstalk**

---

### 🔬 Methodology

#### 🧹 1. Data Preprocessing

* Managed missing data through **custom imputation strategies**
* Categorical encoding with **OneHotEncoder** and **OrdinalEncoder**
* Implemented custom `scikit-learn` transformers:

  * `CategoricalToString`
  * `RareLabelGrouper`
* Logging and **custom exception handling** integrated for robust debugging

#### 🧪 2. Model Training

* Built with **TensorFlow**, **scikit-learn**
* Applied **RandomizedSearchCV** for hyperparameter tuning
* Used **train/test split** and **cross-validation** to validate generalizability

#### 🚀 3. Deployment

* Best-performing model integrated into a **Streamlit dashboard**
* Deployed to **AWS Elastic Beanstalk** for real-time user interaction

---

### 🧰 Tech Stack

* Python, TensorFlow, scikit-learn, Pandas, NumPy
* Streamlit
* AWS Elastic Beanstalk
* Jupyter, VS Code

---

### 📁 Project Structure

```bash
├── data/                   # Sensor + patient biometric data
├── pipeline/               # Preprocessing and model training pipeline
├── models/                 # Trained models
├── scripts/                # Streamlit app scripts
├── app/                    # Deployment-ready Streamlit interface
└── README.md
```

---

### 📈 Results

* Accurate glucose predictions with non-invasive inputs
* Seamless pipeline from raw biomedical signals to deployed prediction
* Designed with extensibility for other healthcare use cases

---

### 🧠 Learnings

* Built a complete ML pipeline for **biomedical signal data**
* Gained exposure to **production-level deployment** using AWS
* Learned to optimize pipelines using real-world constraints like **noisy inputs** and **limited labeled data**

---

