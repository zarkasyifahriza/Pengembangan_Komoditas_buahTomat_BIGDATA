import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from skimage.feature import local_binary_pattern
import joblib
import logging
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import folium

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Pastikan folder untuk unggahan gambar ada
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk forecasting harga tomat
def forecast_tomato_prices():
    # Load dataset
    data_file = "dataset/Tomato.csv"
    data = pd.read_csv(data_file)

    # Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    target = data['Average']

    # Stationarity Check (ADF Test)
    adf_test = adfuller(target)
    if adf_test[1] > 0.05:
        target_diff = target.diff().dropna()
    else:
        target_diff = target

    # Fit SARIMA model
    model = SARIMAX(target, 
                    order=(1, 1, 1),  
                    seasonal_order=(1, 1, 1, 7),  
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    results = model.fit()

    # Forecasting
    forecast_steps = 30  
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=target.index[-1], periods=forecast_steps+1, freq='D')[1:]
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Save forecast to JSON
    forecast_data = [{"date": str(date), "price": float(price)} for date, price in zip(forecast_index, forecast_mean)]
    with open("static/forecasting_results.json", "w") as f:
        json.dump(forecast_data, f)

    # Plot actual data and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(target, label='Actual Data', color='blue')
    plt.plot(forecast_index, forecast_mean, label='Forecast', color='orange')
    plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
    plt.title("SARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend()
    plt.savefig('static/plot_forecast.png')

# Fungsi untuk memuat dan memproses gambar
def load_and_preprocess_image(image_data, size=(128, 128)):
    try:
        resized_image = cv2.resize(image_data, size)
        return resized_image
    except Exception as e:
        logging.error(f"Error in resizing image: {e}")
        return None

# Fungsi untuk ekstraksi fitur warna
def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])
    return [mean_hue, mean_saturation, mean_value]

# Fungsi untuk ekstraksi fitur LBP
def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist

# Memuat model dan scaler
try:
    model = joblib.load('model_svm_tomat.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    data_file = "dataset/ANBIG_DATASET_with_coordinates.csv"
    data = pd.read_csv(data_file)
    required_columns = ['Min Price', 'Max Price', 'Modal Price', 'lat', 'lon']
    data = data.dropna(subset=required_columns)  
    X = data[required_columns]  
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_colors = ['#ff0000', '#ff8000', '#ffff00', '#00ff00']
    m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="OpenStreetMap")

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=cluster_colors[row['Cluster']],
            fill=True,
            fill_color=cluster_colors[row['Cluster']],
            fill_opacity=0.8,
            tooltip=f"<b>Kota:</b> {row['State']}<br><b>Klaster:</b> {row['Cluster']}<br><b>Harga Modal:</b> {row['Modal Price']}<br><b>Harga Maksimum:</b> {row['Max Price']}<br>"
        ).add_to(m)

    map_html = m._repr_html_()
    if request.method == 'POST':
        return predict(map_html)
    return render_template('index.html', map_html=map_html)

def predict(map_html):
    try:
        if 'file' not in request.files:
            return 'Tidak ada file yang diunggah', 400

        file = request.files['file']
        if file.filename == '':
            return 'Tidak ada file yang dipilih', 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            logging.error("Gagal memuat gambar.")
            return 'File gambar tidak valid', 400

        processed_image = load_and_preprocess_image(image)
        color_features = extract_color_features(processed_image)
        lbp_features = extract_lbp_features(processed_image)
        features = np.array(color_features + list(lbp_features)).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        categories = ['rusak', 'matang', 'belum matang']
        predicted_category = categories[prediction[0]]

        result = {'filename': file.filename, 'prediction': predicted_category, 'image_url': os.path.join('static', 'uploads', file.filename)}
        return render_template('index.html', results=[result], map_html=map_html)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return 'Terjadi kesalahan saat melakukan prediksi', 500

if __name__ == '__main__':
    app.run(debug=True)
