import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Step 1: Load CSV File
def get_csv_file():
    files = []
    for dirname, _, filenames in os.walk('archive'):
        for file in filenames:
            files.append(os.path.join(dirname, file))
    return files

file_path = get_csv_file()[2]  # Adjust if needed
df = pd.read_csv(file_path)
original_df = df.copy()  # Backup copy for visuals

# Step 2: Clean Data (fill missing values)
def clean_data(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df

df = clean_data(df)

# Step 3: Encode categorical variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Step 4: Prepare features and labels
df.drop('Loan_ID', axis=1, inplace=True)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Step 5: Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Step 7: Evaluate model
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Step 9: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance')
plt.show()
