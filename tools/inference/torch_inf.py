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
    class_names = {
        1: 'distracted',
        2: 'focused',
        3: 'raising_hand',
        4: 'sleep',
        5: 'using_phone'
    }
    
    # Define colors for each class (in RGB format)
    colors = {
        1: (0, 0, 255),  # đỏ
        2: (128, 0, 128),  # tím
        3: (255, 165, 0),     # cam
        4: (255, 255, 0),  # vàng
        5: (255, 0, 0)   # xanh dương
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


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")

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
