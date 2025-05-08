# 🧠 Loan Prediction ML Project

This project is a machine learning pipeline to predict loan approval status using a Random Forest Classifier. It involves data cleaning, label encoding, model training, evaluation with confusion matrix and classification report, and feature importance analysis.

## 📁 Dataset

The dataset used comes from a folder named `archive/`. It should contain a `.csv` file related to loan applications. This dataset includes features like:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Credit_History
- Property_Area
- Loan_Status (target variable)

## 📊 Features Engineered

- Label Encoding of categorical features
- Drop of unimportant ID columns
- Target variable: `Loan_Status`

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn` (for preprocessing, modeling, evaluation)

## 🧪 Evaluation Metrics

- Accuracy Score
- Classification Report
- Confusion Matrix
- Feature Importance plot

## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/your-username/loan-prediction-ml.git
cd loan-prediction-ml

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python loan_prediction.py
