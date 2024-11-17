from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho tất cả các rout

# Tải mô hình đã lưu
random_forest_model = joblib.load("random_forest_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ FE
    data = request.json
    race = data["race"]
    sex = data["sex"]
    age = data["age"]
    temperature = data["temperature"]
    heart_rate = data["heartRate"]
    respiratory_rate = data["respiratoryRate"]

    # Chuẩn bị dữ liệu cho mô hình
    input_data = np.array([[race, sex, age, temperature, heart_rate, respiratory_rate]])

    # Dự đoán bệnh
    prediction = random_forest_model.predict(input_data)

    # Trả về kết quả cho FE
    return jsonify({"diagnosis": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
