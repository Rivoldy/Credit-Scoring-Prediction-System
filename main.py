import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings

warnings.filterwarnings('ignore')

# Memuat dataset
df = pd.read_csv('creditcard.csv')
print(df.head())

# Mengubah kolom yang berisi nilai numerik yang disimpan sebagai string ke tipe data numerik
df['Per Capita Income - Zipcode'] = df['Per Capita Income - Zipcode'].str.replace(r'[\$,]', '', regex=True).astype(float)
df['Yearly Income - Person'] = df['Yearly Income - Person'].str.replace(r'[\$,]', '', regex=True).astype(float)
df['Total Debt'] = df['Total Debt'].str.replace(r'[\$,]', '', regex=True).astype(float)

# Mengisi missing values pada kolom 'Apartment' dengan nilai median
df['Apartment'].fillna(df['Apartment'].median(), inplace=True)

# Mengonversi kolom kategorikal menjadi numerik
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Visualisasi data
plt.figure(figsize=(10, 6))
sns.histplot(df['Current Age'], bins=30, kde=True)
plt.title('Distribusi Usia Saat Ini')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Yearly Income - Person'], bins=30, kde=True)
plt.title('Distribusi Pendapatan Tahunan')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Total Debt'], bins=30, kde=True)
plt.title('Distribusi Total Utang')
plt.show()

# Menghitung korelasi hanya untuk kolom numerik
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi Antar Fitur')
plt.show()

# Membuat fitur baru
df['Debt-to-Income Ratio'] = df['Total Debt'] / df['Yearly Income - Person']
df['Age Difference'] = df['Retirement Age'] - df['Current Age']

# Menampilkan informasi data yang telah diubah
print(df.info())

# Memisahkan fitur dan label
X = df.drop(['Person', 'Address', 'City', 'State', 'Zipcode', 'FICO Score'], axis=1)
y = (df['FICO Score'] > 700).astype(int)  # Menggunakan FICO Score > 700 sebagai target (contoh prediksi kredit baik atau buruk)

# Membagi dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melakukan standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model menggunakan XGBoost dengan hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Menggunakan RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=XGBClassifier(random_state=42, tree_method='hist'), param_distributions=param_dist, 
                                   scoring='accuracy', cv=3, verbose=1, n_iter=10, n_jobs=-1)
random_search.fit(X_train, y_train)

# Mengambil model terbaik dari random search
best_model = random_search.best_estimator_

# Melakukan prediksi
y_pred = best_model.predict(X_test)

# Mengevaluasi akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred))

# Menampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# Menghitung ROC-AUC
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Menyimpan model terbaik
joblib.dump(best_model, 'best_credit_model.pkl')
print("Model terbaik telah disimpan sebagai 'best_credit_model.pkl'")

# Menyimpan scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler telah disimpan sebagai 'scaler.pkl'")

# Menyimpan hasil prediksi
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv('predictions.csv', index=False)
print("Hasil prediksi telah disimpan sebagai 'predictions.csv'")
