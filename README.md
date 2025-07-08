# -Diabetes-Prediction-Using-Machine-Learning

A machine learning project aimed at predicting diabetes likelihood based on diagnostic health data. Built using Python and key data science libraries, this project walks through the full data science lifecycle — from data preprocessing to model evaluation.

## 📊 Dataset

- **File**: `diabetes_prediction_dataset.xlsx`
- **Records**: Several thousand rows (from Stark Health)
- **Target**: `diabetes` (binary classification: 0 = No, 1 = Yes)
- **Features**: Includes BMI, Age, HbA1c_level, Blood glucose level, Gender, Smoking history, and more

## 🧪 Project Workflow

1. **Problem Definition**
   - Predict if an individual is likely to have diabetes using clinical parameters.

2. **Data Preprocessing**
   - Missing value handling
   - Feature encoding
   - Normalization/scaling

3. **Exploratory Data Analysis (EDA)**
   - Visualizations (distributions, pair plots, correlation heatmaps)
   - Feature-target relationships

4. **Model Building**
   - Algorithms used:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Model selection based on accuracy, precision, recall, F1 score

5. **Model Evaluation**
   - Confusion matrix
   - ROC curve
   - Cross-validation

## 📈 Sample Visuals

Visualizations from the notebook include:

- Correlation Heatmap
- Class Distribution
- ROC Curve
- Feature Importance
- Age vs. BMI Scatter
- Blood Glucose Distributions

> 👉 Visuals are available in the `diabetes_project_images/` folder.

## 📁 Project Structure

| File | Description |
|------|-------------|
| `diabets_starkHealth.ipynb` | Full notebook with code and results |
| `diabetes_prediction_dataset.xlsx` | Raw dataset |
| `README.md` | Project documentation |
| `diabetes_project_images/` | Extracted visual assets |

## 🔧 Requirements

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## 🚀 Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~ | ~ | ~ | ~ |
| Random Forest | ~ | ~ | ~ | ~ |
| XGBoost | ~ | ~ | ~ | ~ |

_(Update these with actual values from notebook results)_

## 📌 Future Improvements

- Try deep learning models (e.g., MLP, LSTM)
- Optimize hyperparameters using GridSearchCV
- Deploy using Streamlit or Flask for live predictions

## 🙌 Acknowledgments

This project is inspired by real-world health diagnostics and aims to support preventive care efforts using machine learning insights.
