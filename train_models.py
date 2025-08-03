# train_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# === Step 1: Load & Clean Data ===
df = pd.read_csv("loan_default_cleaned.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# === Step 2: Encode Target ===
le = LabelEncoder()
df['loan_default'] = le.fit_transform(df['loan_default'])

# === Step 3: Preprocess ===
X = df.drop('loan_default', axis=1)
y = df['loan_default']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'minmax_scaler.pkl')

# === Step 4: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Step 5: Train Random Forest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save Random Forest Model
joblib.dump(rf_model, 'random_forest_model.pkl')

# === Step 6: Train LSTM ===
# Reshape data for LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_lstm, y_test),
    callbacks=[early_stop]
)

# Save LSTM model
lstm_model.save('lstm_model.h5')

print("\n✅ Models trained and saved successfully!")
