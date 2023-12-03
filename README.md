# Phân loại hoa diên vĩ Backend

## 1. Khởi tạo môi trường ảo

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

## 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 3. Cấu hình biến môi trường

Copy file _.env.example_ sang _.env_ và điền đầy đủ thông tin

## 4. Huấn luyện mô hình

```bash
flask train-model
```

## 5. Chạy server

```bash
flask run
```
