# Hệ Thống Đánh Giá Hành Vi Học Sinh Theo Thời Gian Thực

Hệ thống này sử dụng mô hình D-FINE để phân tích hành vi của học sinh thông qua video hoặc webcam theo thời gian thực. Hệ thống có khả năng nhận diện 5 loại hành vi:
- Mất tập trung (Distracted)
- Tập trung (Focused)
- Giơ tay (Raising hand)
- Buồn ngủ (Sleep)
- Dùng điện thoại (Using phone)

## Cài đặt

1. Đảm bảo bạn đã cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision opencv-python pillow flask
```

2. Tải mô hình D-FINE đã được huấn luyện và file cấu hình

## Cách sử dụng

### Khởi động server

```bash
cd tools/deployment
python behavior_fe.py -c <path_to_config> -r <path_to_model> -d <device> -p <port>
```

Trong đó:
- `<path_to_config>`: Đường dẫn đến file cấu hình, ví dụ: `configs/dfine/dfine_hgnetv2_l_coco.yml`
- `<path_to_model>`: Đường dẫn đến file mô hình đã huấn luyện (file .pth)
- `<device>`: Thiết bị để chạy mô hình (`cpu` hoặc `cuda`)
- `<port>`: Cổng để chạy server web (mặc định: 5000)

Sau khi khởi động, bạn có thể truy cập trang web qua địa chỉ: `http://localhost:5000`

### Sử dụng giao diện web

1. **Tải mô hình**:
   - Nhập đường dẫn đến file cấu hình và mô hình
   - Chọn thiết bị phù hợp (CPU hoặc CUDA)
   - Nhập tên lớp học để phân tích
   - Bấm "Tải Model"

2. **Chọn nguồn video**:
   - Chọn Webcam hoặc Video File
   - Nếu chọn Video File, hãy tải lên một video từ máy tính

3. **Phân tích hành vi**:
   - Bấm "Bắt đầu" để bắt đầu phân tích
   - Hệ thống sẽ hiển thị video đã được xử lý với các khung bao quanh đối tượng
   - Biểu đồ miền (Area chart) sẽ hiển thị tỉ lệ các hành vi theo thời gian
   - Khi kết thúc hoặc bấm "Dừng", hệ thống sẽ hiển thị biểu đồ cột tổng kết tỉ lệ các hành vi

4. **Các tính năng bổ sung**:
   - Chụp ảnh màn hình: Lưu lại khoảnh khắc hiện tại của video
   - Xuất kết quả: Xuất dữ liệu phân tích dưới dạng CSV
   - Lưu kết quả lớp học: Lưu kết quả để so sánh với các lớp khác sau này
   - So sánh giữa các lớp học: Xem biểu đồ so sánh hành vi giữa các lớp khác nhau

## Phân tích kết quả

Sau khi phân tích, hệ thống sẽ hiển thị:
- Biểu đồ miền theo thời gian
- Biểu đồ cột tổng kết
- Bảng thống kê chi tiết
- Nhận xét về hành vi của học sinh
- Đề xuất cải thiện dựa trên kết quả phân tích

## Cấu trúc thư mục

```
deployment/
├── templates/           # Thư mục chứa các file giao diện
│   ├── static/          # CSS, JS và các file tĩnh
│   │   ├── css/         # File CSS
│   │   └── js/          # File JavaScript
│   └── index.html       # Trang web chính
├── uploads/             # Thư mục lưu video đã tải lên
├── behavior_fe.py       # File chính của server Flask
└── README.md            # Hướng dẫn sử dụng
```

## Phát triển thêm

Một số ý tưởng phát triển thêm:
- Thêm tính năng phân tích nhiều video cùng lúc
- Xây dựng báo cáo tự động và gửi qua email
- Tích hợp với hệ thống quản lý lớp học hiện có
- Thêm các thuật toán phân tích xu hướng để dự đoán sự thay đổi hành vi 