import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- STEP 2: LOAD AND EXPLORE DATASET ---
print("Step 2: Loading data from Excel...")
# Aapka file path yahan update kar diya gaya hai
file_path = 'data/student_performance_dataset.xlsx'

try:
    df = pd.read_excel(file_path)
    print("File successfully loaded!")
    print("\n--- First 5 Rows of Data ---")
    print(df.head())
except Exception as e:
    print(f"Error: File nahi mili. Check karein ki path '{file_path}' sahi hai ya nahi.")
    print(f"Specific Error: {e}")
    exit()

# --- STEP 3: PREPROCESS DATA ---
print("\nStep 3: Preprocessing data...")

# Text labels (Pass/Fail) ko numbers (1/0) mein badalna
le = LabelEncoder()
if 'status' in df.columns:
    df['status'] = le.fit_transform(df['status'])
else:
    print("Error: Excel file mein 'status' column nahi mila!")
    exit()

# Features (X) aur Target (y) select karna
# Note: Check karein ki aapki Excel mein column names yahi hain
X = df[['study_hours', 'attendance', 'previous_scores']]
y = df['status']

# Data ko train aur test sets mein split karna (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: TRAIN ML MODEL ---
print("\nStep 4: Training the Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy check karna
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- TEST WITH NEW DATA ---
# Example: Ek student jisne 7 ghante padha, 85% attendance, aur 75 score laya
sample_data = [[7, 85, 75]]
prediction = model.predict(sample_data)
result = "Pass" if prediction[0] == 1 else "Fail"
print(f"\nPrediction for test student {sample_data}: {result}")