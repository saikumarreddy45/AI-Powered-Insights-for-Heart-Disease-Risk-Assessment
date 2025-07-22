import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Load data
os.makedirs('model', exist_ok=True)
df = pd.read_csv('data/heart.csv')

# Use the correct target column name
if 'target' in df.columns:
    target_col = 'target'
elif 'HeartDisease' in df.columns:
    target_col = 'HeartDisease'
else:
    raise ValueError('Target column not found!')

X = df.drop(target_col, axis=1)
y = df[target_col]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
df_train, df_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(df_train, y_train)

# Evaluate
y_pred = model.predict(df_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl') 