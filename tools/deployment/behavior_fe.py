import nest_asyncio
nest_asyncio.apply()

import os
import sys
import time
import json
import base64
from datetime import datetime
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import deque, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'transforms' not in st.session_state:
    st.session_state.transforms = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'class_name' not in st.session_state:
    st.session_state.class_name = ""
if 'chart_placeholder' not in st.session_state:
    st.session_state.chart_placeholder = None
if 'last_chart_update' not in st.session_state:
    st.session_state.last_chart_update = 0
if 'selected_classes' not in st.session_state:
    st.session_state.selected_classes = []
if 'comparison_chart_data' not in st.session_state:
    st.session_state.comparison_chart_data = None

class_names = {
    1: 'distracted',
    2: 'focused',
    3: 'raising_hand',
    4: 'sleep',
    5: 'using_phone'
}
colors = {
    1: (0, 0, 255),  # đỏ
    2: (128, 0, 128),  # tím
    3: (255, 165, 0),  # cam
    4: (255, 255, 0),  # vàng
    5: (255, 0, 0)    # xanh dương
}

# Store last 30 seconds of data
MAX_WINDOW_SIZE = 30  # Store 30 seconds of data
FRAMES_PER_SECOND = 5  # 5fps

def reset_buffers():
    """Reset all buffers for new processing session"""
    st.session_state.timestamps_buffer = deque(maxlen=MAX_WINDOW_SIZE)
    st.session_state.class_stats_buffer = {class_id: deque(maxlen=MAX_WINDOW_SIZE) for class_id in class_names}
    st.session_state.overall_stats = {class_id: [] for class_id in class_names}
    st.session_state.frame_buffer = []  # Buffer to store frames for each second
    st.session_state.last_update_time = 0  # Time of last chart update
    st.session_state.chart_placeholder = None  # Reset chart placeholder

# Initialize buffers in session state
if 'timestamps_buffer' not in st.session_state:
    reset_buffers()

def load_model(config_path, model_path, device_name="cpu"):
    """Load model and save to session state"""
    try:
        st.session_state.device = device_name
        cfg = YAMLConfig(config_path, resume=model_path)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        if model_path:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
            cfg.model.load_state_dict(state)
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        st.session_state.model = Model().to(st.session_state.device)
        st.session_state.model.eval()
        
        st.session_state.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        st.session_state.model_loaded = True
        return "Model loaded successfully"
    except Exception as e:
        st.session_state.model_loaded = False
        raise e

def draw_boxes(img, labels, boxes, scores, threshold=0.4):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img
    
    draw = ImageDraw.Draw(img_pil)
    
    valid_idx = scores > threshold
    valid_labels = labels[valid_idx]
    valid_boxes = boxes[valid_idx]
    valid_scores = scores[valid_idx]
    
    # Count detections by class
    class_counts = {class_id: 0 for class_id in class_names}
    
    for j, box in enumerate(valid_boxes):
        label = valid_labels[j].item()
        draw.rectangle(list(box), outline=colors[label], width=3)
        draw.text(
            (box[0], box[1]),
            text=f"{class_names[label]} {round(valid_scores[j].item(), 2)}",
            fill="white",
        )
        class_counts[label] = class_counts.get(label, 0) + 1
    
    if isinstance(img, np.ndarray):
        result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return result_img, class_counts
    else:
        return img_pil, class_counts

def process_frame(frame):
    if frame is None:
        return None, {}
    
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h = frame_pil.size
    orig_size = torch.tensor([[w, h]]).to(st.session_state.device)
    
    img_tensor = st.session_state.transforms(frame_pil).unsqueeze(0).to(st.session_state.device)
    
    with torch.no_grad():
        output = st.session_state.model(img_tensor, orig_size)
    
    labels, boxes, scores = output
    processed_frame, class_counts = draw_boxes(frame, labels[0], boxes[0], scores[0])
    
    # Calculate percentages based on total detected bounding boxes
    total_detected = sum(class_counts.values())
    
    # Ensure we have at least one detection to avoid division by zero
    if total_detected > 0:
        class_percentages = {class_id: count / total_detected for class_id, count in class_counts.items()}
    else:
        # If no detections, set all percentages to 0
        class_percentages = {class_id: 0 for class_id in class_names}
    
    # Ensure all class IDs have a percentage value
    for class_id in class_names:
        if class_id not in class_percentages:
            class_percentages[class_id] = 0
    
    return processed_frame, class_percentages

@st.cache_data
def get_chart_layout():
    return dict(
        title=dict(
            text="Real-time Behavior Analysis",
            font=dict(size=20, color='black')
        ),
        xaxis=dict(
            title=dict(
                text="Time (seconds)",
                font=dict(color='black')
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title=dict(
                text="Percentage",
                font=dict(color='black')
            ),
            tickformat='.0%',
            range=[0, 1],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='black')
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(color='black')
        ),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

def update_chart(chart_placeholder):
    """Update real-time area chart"""
    if len(st.session_state.timestamps_buffer) == 0:
        return

    # Get last 30 seconds of data
    current_time = st.session_state.timestamps_buffer[-1]
    window_start = max(0, current_time - 30)
    
    # Find index where time > window_start
    start_idx = 0
    for i, t in enumerate(st.session_state.timestamps_buffer):
        if t >= window_start:
            start_idx = i
            break
    
    times = list(st.session_state.timestamps_buffer)[start_idx:]
    
    fig = go.Figure()

    for class_id in list(class_names.keys()):
        values = list(st.session_state.class_stats_buffer[class_id])[start_idx:]
        rgb_color = colors[class_id]
        rgba_color = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.7)'
        
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            name=class_names[class_id],
            fill='tonexty',
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor=rgba_color,
            mode='lines',
            hovertemplate="%{y:.1%}<extra>" + class_names[class_id] + "</extra>"
        ))
    
    # Sử dụng layout đã cache
    layout = get_chart_layout()
    layout['xaxis']['range'] = [window_start, current_time]
    fig.update_layout(layout)
    
    # Cập nhật biểu đồ với update_mode="dynamic"
    chart_placeholder.plotly_chart(fig, use_container_width=True, update_mode="dynamic")
    
    # Cập nhật thời gian cập nhật cuối
    st.session_state.last_chart_update = time.time()

def get_dominant_class(frame_results):
    """Get the most common class from a list of frame results"""
    if not frame_results:
        return {class_id: 0 for class_id in class_names}
    
    # Combine all class percentages
    combined_stats = {class_id: 0 for class_id in class_names}
    for result in frame_results:
        for class_id, percentage in result.items():
            combined_stats[class_id] += percentage
    
    # Find the dominant class
    total_frames = len(frame_results)
    return {class_id: count/total_frames for class_id, count in combined_stats.items()}

def export_json(avg_stats, class_name):
    """Export statistics to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "class_name": class_name,
        "timestamp": timestamp,
        "statistics": avg_stats
    }
    
    # Ensure database directory exists
    os.makedirs("database/class", exist_ok=True)
    
    # Save to database
    file_path = os.path.join("database/class", f"behavior_stats_{class_name}_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    # Create download button that doesn't trigger page reload
    with open(file_path, "r") as f:
        json_str = f.read()
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"behavior_stats_{class_name}_{timestamp}.json",
            mime="application/json",
            key=f"download_{timestamp}"  # Unique key to prevent reuse
        )

def load_database_files():
    """Load all JSON files from database/class folder"""
    database_path = "database/class"
    if not os.path.exists(database_path):
        return {}
    
    imported_data = {}
    for file_name in os.listdir(database_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(database_path, file_name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    class_name = data.get("class_name", "Unknown")
                    stats = data.get("statistics", {})
                    timestamp = data.get("timestamp", "Unknown")
                    
                    # Create unique key for each record
                    record_key = f"{class_name}_{timestamp}"
                    
                    imported_data[record_key] = {
                        "class_name": class_name,
                        "stats": stats,
                        "timestamp": timestamp,
                        "file_name": file_name
                    }
            except Exception as e:
                st.error(f"Error loading {file_name}: {str(e)}")
    
    return imported_data

def create_comparison_chart(current_stats, selected_classes=None):
    """Create comparison chart between current class and selected classes from database"""
    # Load database
    database_data = load_database_files()
    
    if not database_data and not selected_classes:
        return
    
    # Create figure for comparison
    fig = go.Figure()
    
    # Add current class data
    current_class = st.session_state.class_name or "Current Class"
    x_data = []
    
    # Add current class data first
    x_data.append(current_class)
    for class_id in class_names:
        value = current_stats[class_id]
        fig.add_trace(go.Bar(
            name=class_names[class_id],
            x=[current_class],
            y=[value],
            marker_color=f'rgba({colors[class_id][0]}, {colors[class_id][1]}, {colors[class_id][2]}, 0.7)'
        ))
    
    # Add selected classes from database
    if selected_classes:
        for class_name in selected_classes:
            if class_name in database_data:
                x_data.append(class_name)
                stats = database_data[class_name]["stats"]
                for class_id in class_names:
                    value = stats.get(str(class_id), 0)
                    fig.add_trace(go.Bar(
                        name=class_names[class_id],
                        x=[class_name],
                        y=[value],
                        marker_color=f'rgba({colors[class_id][0]}, {colors[class_id][1]}, {colors[class_id][2]}, 0.7)'
                    ))
    
    # Update layout
    fig.update_layout(
        title="Behavior Distribution Comparison",
        xaxis_title="Classes",
        yaxis_title="Percentage",
        yaxis=dict(
            range=[0, 1],
            tickformat='.0%'
        ),
        barmode='stack',
        height=500,
        font=dict(color='black'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def toggle_class(class_name):
    """Toggle class selection in session state"""
    if class_name in st.session_state.selected_classes:
        st.session_state.selected_classes.remove(class_name)
    else:
        st.session_state.selected_classes.add(class_name)

def process_video(video_source):
    """Process video frames and update UI in real-time"""
    if video_source is None:
        st.error("No video source selected")
        return
        
    # Reset buffers for new session
    reset_buffers()
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: Could not open video source")
        return

    # Set up video parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Scale down if resolution > 1080p
    if height > 1080:
        scale_factor = 1080 / height
        width = int(width * scale_factor)
        height = 1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_fps = FRAMES_PER_SECOND
    skip_frames = max(1, int(fps / target_fps))
    frame_interval = 1.0 / target_fps  # Thời gian giữa các frame ở target_fps
    
    # Create two columns for video and chart
    col1, col2 = st.columns([1, 1])
    
    # Set up UI elements
    with col1:
        frame_placeholder = st.empty()
        class_stats_placeholder = st.empty()
    
    with col2:
        if st.session_state.chart_placeholder is None:
            st.session_state.chart_placeholder = st.empty()
        chart_placeholder = st.session_state.chart_placeholder
    
    stop_button = st.button("Stop")
    
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    last_second = -1
    last_ui_update_time = start_time
    min_ui_update_interval = 0.1  # Minimum time between UI updates (seconds)
    
    try:
        while cap.isOpened() and not stop_button:
            # Kiểm tra thời gian để duy trì đúng target_fps
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Sleep ngắn để giảm CPU usage
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            current_second = int(current_time - start_time)
            
            # Process at target FPS
            if frame_count % skip_frames == 0:
                processed_frame, class_percentages = process_frame(frame)
                
                # Add frame results to buffer
                st.session_state.frame_buffer.append(class_percentages)
                
                # Only update UI if enough time has passed since last update
                # time_since_last_update = current_time - last_ui_update_time
                # if time_since_last_update >= min_ui_update_interval:
                    # Display current frame
                with col1:
                    display_frame = cv2.resize(processed_frame, (640, 480))
                    frame_placeholder.image(
                        cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    last_frame_time = current_time  # Cập nhật thời gian frame cuối
                
                # Update statistics and chart every second
                if current_second > last_second and len(st.session_state.frame_buffer) > 0:
                    # Get dominant class for the last second
                    dominant_classes = get_dominant_class(st.session_state.frame_buffer)
                    
                    # Update buffers
                    st.session_state.timestamps_buffer.append(current_second)
                    for class_id, percentage in dominant_classes.items():
                        st.session_state.class_stats_buffer[class_id].append(percentage)
                        st.session_state.overall_stats[class_id].append(percentage)
                    
                    # Display current class percentages as a table
                    df = pd.DataFrame({
                        'Class': [class_names[cid] for cid in class_names],
                        'Percentage': [f"{dominant_classes[cid]:.1%}" for cid in class_names]
                    })
                    class_stats_placeholder.table(df)
                    
                    # Update chart
                    update_chart(chart_placeholder)
                    
                    # Clear frame buffer for next second
                    st.session_state.frame_buffer = []
                    last_second = current_second
            
            frame_count += 1
            last_frame_time = current_time
            
    finally:
        cap.release()
        
    # Display final statistics
    if len(st.session_state.overall_stats[1]) > 0:
        st.header("Session Summary")
        avg_stats = {class_id: np.mean(stats) for class_id, stats in st.session_state.overall_stats.items()}
        
        # Add JSON export button
        export_json(avg_stats, st.session_state.class_name)
        
        # Create single class chart
        fig = go.Figure()
        
        # Add bars for current class
        class_name = st.session_state.class_name or "Unnamed Class"
        for class_id in class_names:
            fig.add_trace(go.Bar(
                name=class_names[class_id],
                x=[class_name],
                y=[avg_stats[class_id]],
                marker_color=f'rgba({colors[class_id][0]}, {colors[class_id][1]}, {colors[class_id][2]}, 0.7)'
            ))
        
        # Update layout
        fig.update_layout(
            title="Overall Behavior Distribution",
            xaxis_title="Class",
            yaxis_title="Average Percentage",
            yaxis=dict(
                range=[0, 1],
                tickformat='.0%'
            ),
            barmode='stack',
            height=400,
            font=dict(color='black'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

def clean_upload_folder():
    """Xóa tất cả file trong thư mục uploads"""
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")

def save_uploaded_file(video_file):
    """Lưu file upload với timestamp và trả về đường dẫn"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = os.path.splitext(video_file.name)[1]
    filename = f"video_{timestamp}{file_ext}"
    file_path = os.path.join(upload_dir, filename)
    
    # Lưu file
    with open(file_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    return file_path

def load_example_videos():
    """Load all video files from database/video_class folder"""
    video_path = "database/video_class"
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)
        return []
    
    video_files = []
    for file_name in os.listdir(video_path):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
            video_files.append({
                'name': file_name,
                'path': os.path.join(video_path, file_name)
            })
    
    return sorted(video_files, key=lambda x: x['name'])

def main():
    st.set_page_config(page_title="Student Behavior Analysis", layout="wide")
    st.title("Student Behavior Analysis")
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("Model Configuration")
        config_path = st.text_input("Config Path", "/home/buma04/D-FINE/configs/dfine/custom/dfine_hgnetv2_s_custom.yml")
        model_path = st.text_input("Model Path", "/home/buma04/D-FINE/output/dfine_s_original/last.pth")
        device_name = st.selectbox("Device", ["cpu", "cuda:0", "cuda:1"])
        
        if st.button("Load Model"):
            try:
                message = load_model(config_path, model_path, device_name)
                st.success(message)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.model_loaded = False
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["Analysis", "Comparison"])
    
    with tab1:
        st.header("Video Input")
        
        if not st.session_state.model_loaded:
            st.warning("Please load the model first")
        else:
            # Add Class Name input at the top of Analysis tab
            col1, col2 = st.columns([2, 1])
            with col1:
                st.session_state.class_name = st.text_input("Class Name", value=st.session_state.class_name or "")
            
            # Add some spacing
            st.markdown("---")
            
            source_type = st.radio("Select Input Source", ["Upload Video", "Webcam"])
            
            video_source = None
            if source_type == "Upload Video":
                # Create two columns for upload and example selection
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Upload New Video")
                    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
                    if video_file:
                        video_source = save_uploaded_file(video_file)
                
                with col2:
                    st.subheader("Or Select Example")
                    example_videos = load_example_videos()
                    if example_videos:
                        selected_example = st.selectbox(
                            "Select an example video",
                            options=["None"] + [video['name'] for video in example_videos],
                            format_func=lambda x: "Select an example video" if x == "None" else x
                        )
                        
                        if selected_example != "None":
                            # Find the selected video path
                            video_path = next(
                                (video['path'] for video in example_videos if video['name'] == selected_example),
                                None
                            )
                            if video_path:
                                video_source = video_path
                    else:
                        st.info("No example videos available in database/video_class")
            else:
                video_source = 0  # Use webcam
            
            if st.button("Start Analysis"):
                if video_source is not None:
                    try:
                        process_video(video_source)
                    finally:
                        # Only delete if it's an uploaded file (not an example video)
                        if isinstance(video_source, str) and video_source.startswith("uploads/"):
                            if os.path.exists(video_source):
                                os.unlink(video_source)
                            clean_upload_folder()  # Dọn dẹp thư mục uploads
                else:
                    st.error("Please select a video first")
    
    with tab2:
        st.header("Class Comparison")
        
        # Load database data
        database_data = load_database_files()
        
        if not database_data:
            st.info("No class data available for comparison. Please analyze some classes first.")
            return
            
        # Add filter options
        col1, col2 = st.columns([2, 1])
        with col1:
            available_records = list(database_data.keys())
            selected = st.multiselect(
                "Select classes to compare",
                available_records,
                format_func=lambda x: f"{database_data[x]['class_name']} ({database_data[x]['timestamp']})",
                key="comparison_selector"
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Timestamp", "Class Name"],
                key="sort_option"
            )
            
        if selected:
            # Sort selected records
            if sort_by == "Timestamp":
                selected.sort(key=lambda x: database_data[x]["timestamp"])
            else:
                selected.sort(key=lambda x: database_data[x]["class_name"])
                
            # Create comparison chart
            fig = go.Figure()
            
            # Track which behavior classes have been added to legend
            legend_added = set()
            
            # Add data for each selected record
            for record_key in selected:
                if record_key in database_data:
                    record = database_data[record_key]
                    stats = record["stats"]
                    display_name = f"{record['class_name']}\n({record['timestamp']})"
                    
                    for class_id in class_names:
                        value = float(stats.get(str(class_id), 0))  # Convert to float to handle string values
                        show_legend = class_id not in legend_added
                        
                        fig.add_trace(go.Bar(
                            name=class_names[class_id],
                            x=[display_name],
                            y=[value],
                            marker_color=f'rgba({colors[class_id][0]}, {colors[class_id][1]}, {colors[class_id][2]}, 0.7)',
                            showlegend=show_legend
                        ))
                        
                        # Mark this behavior class as having been added to legend
                        legend_added.add(class_id)
            
            # Update layout
            fig.update_layout(
                title="Behavior Distribution Comparison",
                xaxis_title="Classes",
                yaxis_title="Percentage",
                yaxis=dict(
                    range=[0, 1],
                    tickformat='.0%'
                ),
                barmode='stack',
                height=500,
                font=dict(color='black'),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                xaxis=dict(
                    tickangle=45
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add data table
            st.subheader("Detailed Statistics")
            data = []
            for record_key in selected:
                if record_key in database_data:
                    record = database_data[record_key]
                    stats = record["stats"]
                    row = {
                        "Class": record["class_name"],
                        "Timestamp": record["timestamp"]
                    }
                    for class_id in class_names:
                        value = float(stats.get(str(class_id), 0))
                        row[class_names[class_id]] = f"{value:.1%}"
                    data.append(row)
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()