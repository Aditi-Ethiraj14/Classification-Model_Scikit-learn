# Adult Income Classification with Scikit-learn & XGBoost

This project builds a machine learning classification model to predict whether an individual's income exceeds $50K/year based on demographic information from the **Adult Income Dataset**.  
The model uses **Scikit-learn** for preprocessing and evaluation, and **XGBoost** for training and prediction.

---

## 📌 Table of Contents
1. [Dataset](#dataset)  
2. [Objective](#objective)  
3. [Tech Stack](#tech-stack)  
4. [Pipeline Overview](#pipeline-overview)  
5. [Installation & Usage](#installation--usage)  
6. [Model Performance](#model-performance)  
7. [Why XGBoost](#why-xgboost)  
8. [Future Work](#future-work)  
9. [References](#references)  

---

## Dataset
We use the **Adult Income Dataset** from the UCI Machine Learning Repository:

- **URL:** [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)  
- Contains demographic features such as age, education, occupation, and workclass.  
- Target: whether an individual's income exceeds $50K/year (`<=50K`, `>50K`).  

**Attributes include:**  
`age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country`

---

## Objective
Build a classification model that:
- Loads and preprocesses the dataset  
- Encodes categorical features  
- Normalizes numerical features  
- Trains a classification model  
- Evaluates performance  
- Demonstrates example predictions

---

## Tech Stack
- Python 3.x  
- Scikit-learn  
- Pandas  
- NumPy  
- XGBoost  
- Matplotlib / Seaborn  

---

## Pipeline Overview
The project pipeline consists of:

1. **Data Loading** — Load the dataset into a Pandas DataFrame  
2. **Preprocessing** — Handle missing values, encode categorical variables, scale numerical features  
3. **Model Training** — Train an XGBoost Classifier  
4. **Evaluation** — Compute accuracy, precision, recall, F1 score, and confusion matrix  
5. **Prediction** — Demonstrate predictions for sample cases

---

## Installation & Usage

1. Clone the repository:
```
git clone https://github.com/your-username/adult-income-classification.git
cd adult-income-classification/src
```

2. Install dependencies:
```
pip install -r requirements.txt
```
3. Data Preprocessing:
```
python data_preprocessing.py
```
4. Training the Model:
```
python train_model.py
```
This will train the XGBoost model and save it as xgb_model.pkl.

5. Making Predictions:
```
python predict.py
```
This will load the saved model and demonstrate predictions on example cases.

## Model Performance

Example results after training:

| Model    | Accuracy | Precision | Recall  | F1 Score |
|----------|----------|-----------|---------|----------|
| XGBoost  | 0.8790   | 0.7915    | 0.6766  | 0.7296   |

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/bf89e882-95c8-4d58-aebe-d957ddafda4e" />
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/13b6a385-72b1-445e-935a-e7f60b2e1a68" />


---

## Why XGBoost

XGBoost is chosen for:

- **High accuracy and efficiency**  
- **Robust handling of missing values**  
- **Regularization to prevent overfitting**  
- **Ability to handle both numerical and categorical features after preprocessing**  
- **State-of-the-art performance in tabular data classification tasks**  

Compared to Random Forest or Logistic Regression, XGBoost tends to yield higher accuracy due to gradient boosting optimization and advanced regularization.

---

## Future Work

- Test on unseen datasets for generalization  
- Add cross-validation and stratified sampling  
- Experiment with LightGBM and CatBoost  
- Build a GUI or web app for interactive predictions  

---

## References

- Adult Dataset — [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
