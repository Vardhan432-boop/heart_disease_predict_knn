
# ❤️ Heart Disease Prediction using KNN & PCA

## 📌 Project Overview

This project builds a **machine learning pipeline** to predict the presence of heart disease using clinical features.
The focus is not only on model accuracy but on applying **proper data preprocessing, dimensionality reduction, and evaluation techniques** similar to real-world ML workflows.

---

## 🎯 Objective

To develop a classification model that can accurately identify patients at risk of heart disease while minimizing missed diagnoses (False Negatives).

---

## 🧠 Machine Learning Workflow

The project follows a complete ML pipeline:

1. **Data Cleaning**

   * Missing values handled using median imputation

2. **Outlier Handling**

   * IQR method used to detect and cap extreme values

3. **Feature Scaling**

   * Standardization applied to numeric features

4. **Dimensionality Reduction**

   * PCA used to retain 95% variance and reduce noise

5. **Data Splitting**

   * Train set: 60%
   * Validation set: 20%
   * Test set: 20%

6. **Model Training**

   * K-Nearest Neighbors (KNN) classifier

7. **Hyperparameter Tuning**

   * Best **K** selected using validation recall

8. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix

---

## 📊 Final Results

| Metric    | Value    |
| --------- | -------- |
| Accuracy  | **90%**  |
| Precision | 0.91     |
| Recall    | **0.91** |
| F1 Score  | 0.91     |

Confusion Matrix:

| Actual / Predicted | No Disease | Disease |
| ------------------ | ---------- | ------- |
| No Disease         | 86         | 10      |
| Disease            | 10         | 99      |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 💡 Key Learnings

* Importance of preprocessing before training ML models
* Effect of scaling on distance-based algorithms
* How PCA helps reduce dimensionality and noise
* Why Recall is critical in medical diagnosis models
* Proper use of validation set for hyperparameter tuning

---

## 🚀 Future Improvements

* Try other classifiers (Random Forest, SVM, Logistic Regression)
* Use GridSearchCV for automated tuning
* Deploy model using Flask or Streamlit

---

## ▶️ How to Run

python main.py


---

## 📁 Project Structure


├── main.py
├── heart.csv
├── README.md



---

## 📬 Author

P Balaji Vardhan
