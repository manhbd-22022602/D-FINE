"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh=0.4):
    class_colors = {
        'distracted': (0, 0, 255),    # đỏ
        'focused': (128, 0, 128),     # tím
        'raising_hand': (255, 165, 0), # cam
        'sleep': (255, 255, 0),       # vàng
        'using_phone': (255, 0, 0)    # xanh dương
    }

    for i, im in enumerate(images):
        img_np = np.array(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, (b, label, score) in enumerate(zip(box, lab, scrs)):
            x1, y1, x2, y2 = map(int, b)
            class_id = int(label.item())
            color = colors[class_id]
            
            # Get label text and size
            text = f'{class_names[class_id]} {score.item():.2f}'
            
            # Calculate text size and background rectangle size
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw filled rectangle for text background with alpha blending
            overlay = img_np.copy()
            
            # Draw main bounding box with slightly transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Draw text background
            cv2.rectangle(
                overlay, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width + 10, y1),
                color, 
                -1
            )
            
            # Apply alpha blending
            alpha = 0.3
            img_np = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)
            
            # Draw solid border for bounding box
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            
            # Draw white text
            cv2.putText(
                img_np, 
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        im = Image.fromarray(img_np)
        im.save("torch_results.jpg")


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


def process_video(model, device, file_path, interval_seconds=5):
    cap = cv2.VideoCapture(file_path)

    # Lấy thông tin về video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"FPS: {fps}")
    print(f"Tổng số frame: {frame_count}")
    print(f"Thời lượng video: {duration:.2f} giây")
    print(f"Kích thước frame gốc: {orig_w}x{orig_h}")

    # Thiết lập màu cho từng class
    class_colors = {
        'distracted': (255, 0, 0),    # xanh dương (dạng BGR)
        'focused': (128, 0, 128),     # tím
        'raising_hand': (0, 165, 255), # cam
        'sleep': (0, 255, 255),       # vàng
        'using_phone': (0, 0, 255)    # đỏ
    }

    # Số frame cần bỏ qua giữa mỗi lần inference
    frames_per_extraction = int(fps * interval_seconds)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_index = 0
    last_output = None
    print("Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ thực hiện inference khi đến đúng khoảng thời gian
        if frame_index % frames_per_extraction == 0:
            # Convert frame to PIL image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)

            im_data = transforms(frame_pil).unsqueeze(0).to(device)

            output = model(im_data, orig_size)
            last_output = output
            print(f"Processed frame at {frame_index/fps:.2f} seconds")
        
        # Sử dụng kết quả inference gần nhất
        if last_output is not None:
            labels, boxes, scores = last_output
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw([frame_pil], labels, boxes, scores)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'torch_results.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
