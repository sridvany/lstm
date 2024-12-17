import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
 

# 7.1. Veri Entegrasyonu

file_path = '/kaggle/input/bist100-2000-2024/BIST100.2024.xlsx'
data = pd.read_excel(file_path, decimal=',')

# 7.2. Date sütununu Endeks Haline Getirilmesi

data = data.set_index('Date')

# 7.3. Tanımlayıcı İstatistikler & Genel Bilgi

data.describe()
print(data.describe())
data.info()

# 7.4. Hedef ve Özellikleri Belirleme

y = data['Close']
X = data.drop(columns=['Close'])
print("Features (X):", X.columns.tolist())
print("Target Variable (y): Close")
X = X.astype(float)  #Volume de float64 haline getirildi
y = y.astype(float) 
print("X'in veri tipleri:\n", X.dtypes)

# 7.5. Verinin Eğitim, Test ve Doğrulama Setlerine Ayrılması

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.1)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print("Training set dates:", X_train.index[0], "to", X_train.index[-1])
print("Validation set dates:", X_val.index[0], "to", X_val.index[-1])
print("Test set dates:", X_test.index[0], "to", X_test.index[-1])

# 7.6. Setlerin Ölçeklendirilmesi

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 7.7. Ardışık Verilerin Öğrenimi İçin Dizi Oluşturma

def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60 #Her dizi önceki 60 günlük veriyi kullanarak sonraki günü tahmin etmeye çalışır

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

# 7.8. Modelin Kurulumu

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 7.9. Model Eğitimi

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_seq, y_val_seq),
    verbose=1
)

# 7.10. Test Verisi Üzerinde Tahminler

y_pred_scaled = model.predict(X_test_seq)

# 7.11. Ters Ölçekleme

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test_seq)

# 7.12. Performans Ölçümleri

mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(8, 4))
plt.plot(y_test.index[time_steps:], y_test_orig, label='Actual Values')
plt.plot(y_test.index[time_steps:], y_pred, label='Predictions')
plt.title('LSTM Model Predictions vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
