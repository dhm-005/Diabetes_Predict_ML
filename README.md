# Diabetes Prediction ML

Dự án xây dựng hệ thống dự đoán tiểu đường bằng Machine Learning, sử dụng Random Forest và Flask để triển khai mô hình thành ứng dụng web.

## 1. Công nghệ sử dụng

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Flask
- Jupyter Notebook

## 2. Cấu trúc dự án

```text
Diabetes_Predict_ML/
├── app.py
├── rf_diabetes.pkl
├── requirements.txt
├── README.md
├── train_model.ipynb
├── train_dl.ipynb
├── dl_data.pkl
├── dl_result.pkl
└── templates/
    ├── index.html
    └── result.html
```

Các file `.pkl` được tạo trong quá trình huấn luyện mô hình. Nếu không có sẵn, cần chạy các notebook tương ứng để tạo lại.

## 3. Dữ liệu

Sử dụng bộ dữ liệu `diabetes_100k.csv` phục vụ huấn luyện và đánh giá mô hình.

Đặt file dữ liệu tại đường dẫn được khai báo trong notebook, hoặc cập nhật đường dẫn cho phù hợp với máy tính.

## 4. Cài đặt

Clone repository:

```bash
git clone <repository-url>
cd Diabetes_Predict_ML
```

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường trên Windows:

```bash
venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 5. Huấn luyện mô hình

1. Mở `train_model.ipynb` bằng Jupyter Notebook và chạy các cell đầu tiên để tiền xử lý dữ liệu, huấn luyện các mô hình Machine Learning.
2. Tại cell 52, chương trình dừng và lưu dữ liệu phục vụ huấn luyện Deep Learning vào file `dl_data.pkl`.
3. Mở `train_dl.ipynb`, sử dụng `dl_data.pkl` để huấn luyện mô hình Deep Learning và lưu kết quả vào `dl_result.pkl`.
4. Quay lại `train_model.ipynb`, tiếp tục chạy từ cell 53 để hoàn tất quá trình đánh giá và lựa chọn mô hình.
5. Mô hình Random Forest được lưu vào `rf_diabetes.pkl`, bao gồm mô hình đã huấn luyện, scaler, danh sách cột đầu vào và các giá trị tham chiếu dùng cho ứng dụng Flask.

## 6. Chạy ứng dụng

Từ thư mục gốc của dự án, chạy:

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000/
```

Nhập thông tin theo biểu mẫu để xem kết quả dự đoán.

## 7. Chức năng

- Nhập thông tin cá nhân và các chỉ số sức khỏe.
- Tiền xử lý dữ liệu đầu vào theo cấu trúc mô hình đã huấn luyện.
- Dự đoán bằng mô hình Random Forest.
- Hiển thị kết quả, xác suất dự đoán và thông tin đầu vào.
- So sánh một số chỉ số với giá trị tham chiếu trong dữ liệu huấn luyện.

## 8. Lưu ý

Kết quả chỉ phục vụ mục đích học tập và tham khảo. Mô hình không thay thế cho chẩn đoán hoặc tư vấn y khoa chuyên nghiệp.