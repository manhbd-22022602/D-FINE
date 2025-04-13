#!/bin/bash

# Giá trị mặc định
INPUT_PATH=""
CUDA_DEVICE="cuda:0"  # Mặc định dùng cuda:0

# Xử lý tham số đầu vào
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_PATH="$2"; shift ;;  # Lấy giá trị sau --input
        --device) CUDA_DEVICE="$2"; shift ;;  # Lấy giá trị sau --device
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Kiểm tra xem đã truyền input chưa
if [ -z "$INPUT_PATH" ]; then
    echo "Usage: $0 --input <image_path> [--device <cuda_device>]"
    exit 1
fi

# Thực hiện lệnh inference
python tools/inference/torch_inf.py \
    -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml \
    -r /home/buma04/D-FINE/output/dfine_hgnetv2_s_custom/last.pth \
    --input "$INPUT_PATH" \
    --device "$CUDA_DEVICE"
