﻿# Nursery Dataset - Random Forest Demo

## Mục đích
Đây là repo phục vụ mục đích học tập về xử lý dữ liệu lệch lớp (imbalanced data), huấn luyện mô hình và xây dựng ứng dụng web dự đoán phân loại trẻ dựa trên bộ dữ liệu Nursery.

## Nội dung repo
- Tiền xử lý dữ liệu, kiểm tra và trực quan hóa phân phối lớp.
- Huấn luyện mô hình Random Forest (main), Decision Tree, Navie Bayes ; đánh giá bằng các chỉ số phù hợp (accuracy, F1-score).
- Lưu lại mô hình tốt nhất và các encoder.
- Xây dựng web Flask cho phép nhập dữ liệu và dự đoán trực tiếp.

## Hướng dẫn sử dụng
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Cài đặt các thư viện check clean code:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy web Flask:
   ```bash
   python app.py
   ```
4. Truy cập web tại `http://localhost:5000` để nhập dữ liệu và xem kết quả dự đoán.

## Thông tin dữ liệu
- Dữ liệu: [UCI Nursery Data Set](https://archive.ics.uci.edu/ml/datasets/nursery)
- Đặc điểm: Dữ liệu nhiều lớp, bị lệch mạnh (class imbalance).

## Quyền sở hữu
1. Sinh viên 1:
   ```bash
   Họ tên: Chau Sô Na
   MSSV: B2205890
   ```
2. Sinh viên 2:
   ```bash
   Họ tên: Nguyễn Văn Kha
   MSSV: B2205877
   ```

## Ghi chú
- Repo này chỉ dùng cho mục đích học tập, tham khảo.

