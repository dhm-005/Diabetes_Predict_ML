# Diabetes_Predict_ML

Hệ thống dự đoán khả năng mắc bệnh tiểu đường sử dụng Machine Learning.

## Công nghệ sử dụng

* Python
* Scikit-learn
* Pandas
* NumPy
* Flask
* Jupyter Notebook
* VS Code / Anaconda

## Cách chạy hệ thống

### 1. Clone repository

git clone <repository-url>

cd Diabetes_Predict_ML

### 2. Tạo môi trường

conda create -n diabetes-env python=3.10

conda activate diabetes-env

### 3. Cài đặt thư viện

pip install -r requirements.txt

### 4. Chạy ứng dụng

python app.py

Sau đó truy cập:

http://127.0.0.1:5000

## Huấn luyện mô hình

Mở file `.ipynb` bằng VS Code và chọn kernel `diabetes-env`.

Notebook thực hiện:
- Khảo sát và trực quan hóa dữ liệu
- Tiền xử lý dữ liệu
- Huấn luyện 5 mô hình Machine Learning
- So sánh các mô hình
- Lưu mô hình KNN và StandardScaler để triển khai Flask