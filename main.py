!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
display(df.head())
input_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
output_feature = 'status'
X = df[input_features]
y = df[output_feature]
print("Input Features (X) head:")
display(X.head())
print("\nOutput Feature (y) head:")
display(y.head())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=input_features)
print("Scaled Input Features (X_scaled) head:")
display(X_scaled_df.head())
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
print("Model chosen: Logistic Regression")
from sklearn.metrics import accuracy_score
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model accuracy on the validation set: {accuracy:.4f}")
if accuracy >= 0.8:
    print("Accuracy meets the requirement of at least 0.8.")
else:
    print("Accuracy does not meet the requirement of at least 0.8. Further model tuning or feature engineering might be needed.")
import joblib
joblib.dump(model, 'my_model.joblib')
