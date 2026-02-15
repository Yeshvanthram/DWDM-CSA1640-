import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["pandas", "numpy", "matplotlib", "scikit-learn"]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print("Installing", package, "...")
        install(package)

# ------------------------------------------------------------
# STEP 2: Import libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nAll modules are ready.\n")

# ------------------------------------------------------------
# STEP 3: Function to convert HH:MM → Minutes
# ------------------------------------------------------------

def time_to_minutes(t):
    hours, minutes = map(int, t.split(":"))
    return hours * 60 + minutes

# ------------------------------------------------------------
# STEP 4: Create Dataset with Real Time Format
# ------------------------------------------------------------

data = {
    "Amount": [
        100, 500, 2000, 50, 7000,
        120, 3000, 80, 9000, 150,
        400, 6000, 75, 8500, 220,
        130, 7200, 95, 110, 9800
    ],

    "Time": [
        "09:15", "10:30", "11:45", "08:20", "14:50",
        "09:40", "13:10", "07:55", "15:35", "10:05",
        "11:20", "14:25", "08:45", "16:10", "09:55",
        "08:35", "15:15", "07:40", "09:05", "17:30"
    ],

    "Location": [
        "Chennai", "Mumbai", "Delhi", "Chennai", "Bangalore",
        "Hyderabad", "Delhi", "Chennai", "Kolkata", "Mumbai",
        "Pune", "Bangalore", "Hyderabad", "Kolkata", "Pune",
        "Chennai", "Bangalore", "Hyderabad", "Mumbai", "Kolkata"
    ],

    "No_of_Transactions": [
        2, 5, 8, 1, 15,
        2, 10, 1, 18, 3,
        4, 14, 1, 16, 3,
        2, 15, 1, 2, 20
    ],

    "Class": [
        0, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        0, 1, 0, 1, 0,
        0, 1, 0, 0, 1
    ]
}

df = pd.DataFrame(data)

print("Dataset with Real Time:\n")
print(df)

# ------------------------------------------------------------
# STEP 5: Convert Time to Minutes
# ------------------------------------------------------------

df["Time"] = df["Time"].apply(time_to_minutes)

# ------------------------------------------------------------
# STEP 6: Encode Location Names
# ------------------------------------------------------------

encoder = LabelEncoder()
df["Location"] = encoder.fit_transform(df["Location"])

# ------------------------------------------------------------
# STEP 7: Prepare Data
# ------------------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# STEP 8: Split Data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# STEP 9: Train Model
# ------------------------------------------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# STEP 10: Test Accuracy
# ------------------------------------------------------------

predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

print("\nClassification Report:\n",
      classification_report(y_test, predictions))

# ------------------------------------------------------------
# STEP 11: Fraud Detection Function
# ------------------------------------------------------------

def detect_fraud(amount, time_str, location, no_of_transactions):

    time_minutes = time_to_minutes(time_str)

    location_encoded = encoder.transform([location])[0]

    sample = pd.DataFrame({
        "Amount": [amount],
        "Time": [time_minutes],
        "Location": [location_encoded],
        "No_of_Transactions": [no_of_transactions]
    })

    sample_scaled = scaler.transform(sample)

    result = model.predict(sample_scaled)

    if result[0] == 1:
        print("\nFraudulent Transaction Detected!")
    else:
        print("\nNormal Transaction")

# ------------------------------------------------------------
# STEP 12: Test Example
# ------------------------------------------------------------

print("\nTesting New Transaction:")
detect_fraud(9000, "16:45", "Kolkata", 18)

# ------------------------------------------------------------
# STEP 13: Visualization
# ------------------------------------------------------------

counts = df["Class"].value_counts()

plt.bar(["Normal", "Fraud"], counts)

plt.title("Fraud vs Normal Transactions")

plt.xlabel("Transaction Type")

plt.ylabel("Number of Transactions")

plt.show()

# ============================================================
# END
# ============================================================
