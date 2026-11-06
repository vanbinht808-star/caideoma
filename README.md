# caideoma

YOLO26 camera với Pose Estimation, hand landmarks và ước lượng khoảng cách.

## Deploy Render

1. Mở Render Dashboard và chọn **New + > Blueprint**.
2. Kết nối repository GitHub `vanbinht808-star/caideoma`.
3. Render đọc `render.yaml` và khởi tạo Web Service.
4. Dùng **Free plan**; Build Command và Start Command đã cấu hình trong `render.yaml`.
5. Mở URL `https://<service-name>.onrender.com` và nhấn **Bật camera**.

Camera được lấy từ thiết bị người dùng qua `getUserMedia`, nên trang cần HTTPS. Render tự cung cấp HTTPS.

## Chạy local

```bash
python3 -m pip install -r requirements.txt
python3 backend/app.py
```

Mở `http://localhost:4173`.
