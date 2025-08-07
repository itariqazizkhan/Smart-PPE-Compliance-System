import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from fpdf import FPDF
import io
import pandas as pd
import plotly.express as px # Import Plotly Express for advanced charting

# --- Configuration ---
MODEL_PATH = 'best_v12.pt'  # Path to your trained YOLOv12 model
CLASSES = ['Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots', 'Protective Helmet', 'Safety Vest', 'Shield']
REPORT_DIR = 'reports' # Directory to save reports (not used for direct download in Streamlit, but good practice)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Smart PPE Compliance System",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Metric Cards ---
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6; /* Light gray background */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px; /* Space between cards if stacked */
        height: 100%; /* Ensure cards in a row have equal height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem; /* Adjust metric label font size */
        color: #555;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5rem; /* Adjust metric value font size */
        font-weight: bold;
        color: #333; /* Darker color for value */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Session State Initialization ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'stop_time' not in st.session_state:
    st.session_state.stop_time = None
if 'session_results' not in st.session_state:
    st.session_state.session_results = None
if 'confidence_threshold' not in st.session_state: # Initialize confidence threshold in session state
    st.session_state.confidence_threshold = 0.5
if 'frame_skip_interval' not in st.session_state: # Initialize frame skip
    st.session_state.frame_skip_interval = 1
if 'inference_imgsz' not in st.session_state: # Initialize inference image size
    st.session_state.inference_imgsz = 640
if 'required_ppe_selection' not in st.session_state: # Initialize required PPE selection
    # Default to all except 'Shield' as initially in your code
    st.session_state.required_ppe_selection = [item for item in CLASSES if item != 'Shield']


# --- Helper Functions ---

@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLO model and caches it."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

def draw_boxes(image, detections, class_names, original_shape):
    """Draws bounding boxes and labels on the image, scaling them if inference was done on a smaller image."""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # Attempt to load a common font, or fall back to a default if not found
    try:
        font = ImageFont.truetype("arial.ttf", 20) 
    except IOError:
        font = ImageFont.load_default() # Fallback font

    # Calculate scaling factors
    original_height, original_width = original_shape[:2]
    
    for det in detections:
        # Extract bounding box coordinates from the inference resolution
        x1_inf, y1_inf, x2_inf, y2_inf = map(int, det.xyxy[0])
        confidence = det.conf[0]
        class_id = int(det.cls[0])
        label = class_names[class_id]

        # Scale bounding box coordinates back to original frame size
        # Assuming inference_imgsz is square, and the original frame might not be.
        # We need to consider the aspect ratio if the original frame is not square and was padded/resized
        # For simplicity, if original_width or original_height is different from inference_imgsz,
        # we'll scale based on the ratio that preserves aspect (e.g., if it was scaled to fit)
        # A more robust solution would involve knowing how the image was resized for inference (e.g., letterbox)
        
        # Simple direct scaling (might distort if aspect ratios differ significantly)
        scale_x = original_width / st.session_state.inference_imgsz
        scale_y = original_height / st.session_state.inference_imgsz

        x1 = int(x1_inf * scale_x)
        y1 = int(y1_inf * scale_y)
        x2 = int(x2_inf * scale_x)
        y2 = int(y2_inf * scale_y)

        # Draw rectangle
        color = (0, 255, 0) # Green color for bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        text = f"{label} {confidence:.2f}"
        # Calculate text bounding box to draw background rectangle
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)

        # Draw text
        draw.text((x1 + 2, y1 - text_height - 3), text, fill=(0, 0, 0), font=font)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_pdf_report(detection_summary, start_time, stop_time, overall_compliance_status, missing_items, required_ppe_list):
    """Generates a PDF report of the detection session."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 10, "PPE Compliance Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Session Start: {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A'}", 0, 1)
    pdf.cell(0, 10, f"Session End: {stop_time.strftime('%Y-%m-%d %H:%M:%S') if stop_time else 'N/A'}", 0, 1)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Detection Summary", 0, 1, "L")
    pdf.ln(5)

    if detection_summary:
        # Prepare data for table
        data = [["PPE Item", "Detections", "Compliance Status"]]
        
        for item in CLASSES: # Iterate through all known classes for the table
            count = detection_summary.get(item, 0)
            status = "Detected" if count > 0 else "NOT DETECTED"
            if item in required_ppe_list: # Check if this item is required
                if count == 0:
                    status = "MISSING (Required)" # Highlight if required and missing
            data.append([item, str(count), status])

        # Add table to PDF
        pdf.set_font("Arial", "", 10)
        col_width = pdf.w / 4.5
        row_height = 8
        
        # Table Header
        pdf.set_fill_color(200, 220, 255)
        for col in data[0]:
            pdf.cell(col_width, row_height, col, 1, 0, 'C', 1)
        pdf.ln(row_height)

        # Table Rows
        for row in data[1:]:
            for item in row:
                pdf.cell(col_width, row_height, str(item), 1, 0, 'C')
            pdf.ln(row_height)
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Overall Compliance: {overall_compliance_status}", 0, 1, "C")
        if overall_compliance_status == "NON-COMPLIANT":
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Missing PPE: {', '.join(missing_items)}", 0, 1, "C")

    else:
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "No detections recorded during this session.", 0, 1)

    return pdf.output(dest='S').encode('latin1')

# --- Main Dashboard Layout ---

st.title("👷 Smart PPE Compliance System")
st.markdown(
    """
    Welcome to the Smart PPE Compliance Dashboard. This system utilizes a YOLOv12 model
    to detect Personal Protective Equipment (PPE) in real-time via your webcam.
    """
)

# --- Sidebar Content ---
with st.sidebar:
    st.title("PPE Compliance Controls") # More prominent title for the sidebar
    st.markdown("---") # Separator

    # Load Model (remains at the top of the sidebar)
    with st.spinner("Loading YOLOv12 model..."):
        model = load_yolo_model(MODEL_PATH)
    # Removed the "Model ready." message as requested.

    # --- Session Management (Buttons moved here) ---
    st.subheader("Session Control") # New subheader for the buttons
    col1, col2 = st.columns(2) # These are columns within the sidebar context

    if col1.button("Start Detection", use_container_width=True, type="primary"):
        if not st.session_state.running:
            st.session_state.running = True
            st.session_state.detection_log = []
            st.session_state.start_time = datetime.now()
            st.session_state.stop_time = None
            st.session_state.session_results = None
            st.experimental_rerun() # Rerun to start detection loop

    if col2.button("Stop Detection", use_container_width=True, type="secondary"):
        if st.session_state.running:
            st.session_state.running = False
            st.session_state.stop_time = datetime.now()
            
            # Process session results
            detection_counts = {cls: 0 for cls in CLASSES}
            for det_info in st.session_state.detection_log:
                detection_counts[det_info['class_name']] += 1
            st.session_state.session_results = detection_counts
            st.experimental_rerun() # Rerun to display results

    st.markdown("---") # Separator

    # --- Real-time Detection Fragment (Sliders) ---
    @st.experimental_fragment
    def realtime_detection_fragment_content(): 
        st.subheader("Real-time Parameters") # Subheader for this section
        st.session_state.confidence_threshold = st.slider( 
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Adjust the minimum confidence score for a detection to be displayed."
        )

        st.session_state.frame_skip_interval = st.slider( 
            "Frame Skip Interval",
            min_value=1,
            max_value=10,
            value=st.session_state.frame_skip_interval,
            step=1,
            help="Process every Nth frame to reduce load. Higher value means smoother display but less frequent detection."
        )

        st.session_state.inference_imgsz = st.slider( 
            "Inference Image Size (pixels)",
            min_value=320,
            max_value=1280,
            value=st.session_state.inference_imgsz,
            step=32,
            help="Resolution for model inference. Lower values increase speed but may reduce accuracy."
        )
    
    realtime_detection_fragment_content() # Call the fragment within the sidebar context
    st.markdown("---") # Separator

    # --- Compliance Configuration ---
    st.subheader("Compliance Settings") # Subheader for this section
    st.session_state.required_ppe_selection = st.multiselect(
        "Select Required PPE for Compliance",
        options=CLASSES,
        default=st.session_state.required_ppe_selection,
        help="Select which PPE items are considered mandatory for compliance. If any selected item is not detected, the system will report 'NON-COMPLIANT'."
    )
    st.markdown("---") # Separator

    # Download Report Button (remains in its section)
    st.subheader("Report Generation") # New subheader for report
    if st.session_state.session_results:
        # Recalculate compliance status for PDF generation as these are not stored in session_state directly
        temp_overall_compliance_status = "COMPLIANT"
        temp_missing_items = []
        
        # Use the user-selected required PPE for PDF generation
        current_required_ppe_for_report = st.session_state.required_ppe_selection
        
        for item in current_required_ppe_for_report: # Iterate over the selected required PPE
            if st.session_state.session_results.get(item, 0) == 0:
                temp_overall_compliance_status = "NON-COMPLIANT"
                temp_missing_items.append(item)

        pdf_output = generate_pdf_report(
            st.session_state.session_results,
            st.session_state.start_time,
            st.session_state.stop_time,
            overall_compliance_status=temp_overall_compliance_status,
            missing_items=temp_missing_items,
            required_ppe_list=current_required_ppe_for_report # Pass this to the function
        )
        
        st.download_button( # No st.sidebar here because we are already in the sidebar context
            label="Download Session Report (PDF)",
            data=pdf_output,
            file_name=f"PPE_Compliance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary" 
        )
    else:
        st.info("Complete a session to enable report download.")

# --- Real-time Detection Area (This part remains outside the fragment and sidebar) ---
st.subheader("Real-time Webcam Feed")
frame_placeholder = st.empty()
status_text = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please ensure it's connected and not in use.")
        st.session_state.running = False
        # No return here, allow the rest of the script to run to display placeholders
    else:
        frame_count = 0
        status_text.info(f"Detection active. Confidence >= {st.session_state.confidence_threshold:.2f}, Frame Skip: {st.session_state.frame_skip_interval}, Inference Size: {st.session_state.inference_imgsz}x{st.session_state.inference_imgsz}...")
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                status_text.error("Failed to grab frame from webcam. Stopping detection.")
                st.session_state.running = False
                break

            original_frame_shape = frame.shape # Store original shape for scaling bounding boxes

            # Implement frame skipping
            frame_count += 1
            if frame_count % st.session_state.frame_skip_interval != 0:
                # Display the skipped frame without detection but keep the video flowing
                frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                continue # Skip detection for this frame

            # Resize frame for inference if necessary
            if st.session_state.inference_imgsz != max(original_frame_shape[:2]):
                resized_frame = cv2.resize(frame, (st.session_state.inference_imgsz, st.session_state.inference_imgsz))
            else:
                resized_frame = frame

            # Perform inference using the dynamic confidence threshold and image size
            results = model(resized_frame, conf=st.session_state.confidence_threshold, imgsz=st.session_state.inference_imgsz, verbose=False) # verbose=False to suppress console output

            # Process results and draw bounding boxes
            annotated_frame = frame.copy() # Draw on the original frame
            
            if results and results[0].boxes:
                # Pass original_frame_shape to draw_boxes for correct scaling
                annotated_frame = draw_boxes(annotated_frame, results[0].boxes, CLASSES, original_frame_shape)
                for det in results[0].boxes:
                    class_id = int(det.cls[0])
                    class_name = CLASSES[class_id]
                    st.session_state.detection_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'class_name': class_name,
                        'confidence': float(det.conf[0])
                    })
            
            # Display the frame
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        status_text.success("Detection session ended.")
        
else:
    if st.session_state.session_results is None:
        frame_placeholder.image("https://placehold.co/600x400/D3D3D3/000000?text=Webcam+Feed+Area", caption="Webcam Feed Area (Click 'Start Detection')", use_column_width=True)
        status_text.info("Ready to start PPE compliance detection.")
    else:
        # Display the last captured frame or a placeholder after stopping
        frame_placeholder.image("https://placehold.co/600x400/D3D3D3/000000?text=Session+Ended", caption="Detection Session Ended", use_column_width=True)
        status_text.success("Detection session ended. See results below.")


# --- Session Results Display Area (Advanced) ---
st.markdown("---")
st.subheader("Session Summary")

if st.session_state.session_results:
    # Calculate key metrics
    total_detections = sum(st.session_state.session_results.values())
    session_duration = st.session_state.stop_time - st.session_state.start_time
    session_duration_str = str(session_duration).split('.')[0] # Remove microseconds for cleaner display

    overall_compliance_status = "COMPLIANT"
    missing_items = []
    
    # Use the user-selected required PPE for overall compliance status calculation
    current_required_ppe = st.session_state.required_ppe_selection
    
    for item in current_required_ppe: # Iterate over the selected required PPE
        if st.session_state.session_results.get(item, 0) == 0:
            overall_compliance_status = "NON-COMPLIANT"
            missing_items.append(item)
    
    num_missing_ppe_types = len(missing_items)

    # Display cards using st.columns and custom HTML/CSS
    st.markdown("#### Key Metrics")
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

    with col_metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Compliance</div>
            <div class="metric-value">{overall_compliance_status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_metric2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">{total_detections}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_metric3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Missing PPE Types</div>
            <div class="metric-value">{num_missing_ppe_types}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_metric4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Session Duration</div>
            <div class="metric-value">{session_duration_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Detected PPE Breakdown")

    # Create a DataFrame for charting
    results_df_chart = pd.DataFrame(
        st.session_state.session_results.items(),
        columns=["PPE Item", "Total Detections"]
    )
    
    # Bar Chart for PPE Detections
    fig_bar = px.bar(
        results_df_chart,
        x="PPE Item",
        y="Total Detections",
        title="Total Detections per PPE Item",
        labels={"PPE Item": "PPE Category", "Total Detections": "Number of Detections"},
        color="PPE Item", # Color bars by PPE Item
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Corrected Pie Chart for Overall Required PPE Status
    st.markdown("#### Overall Required PPE Status (Types Detected vs. Missing)")
    
    detected_required_types_count = 0
    missing_required_types_count = 0

    if current_required_ppe: # Ensure there are required items selected by the user
        for item in current_required_ppe:
            if st.session_state.session_results.get(item, 0) > 0:
                detected_required_types_count += 1
            else:
                missing_required_types_count += 1
    
    if detected_required_types_count + missing_required_types_count > 0:
        compliance_data_pie = {
            "Status": ["Required PPE Detected", "Required PPE Missing"],
            "Count": [detected_required_types_count, missing_required_types_count]
        }
        compliance_df_pie = pd.DataFrame(compliance_data_pie)

        fig_pie = px.pie(
            compliance_df_pie,
            values="Count",
            names="Status",
            title="Required PPE Status (Types Detected vs. Missing)",
            color_discrete_map={"Required PPE Detected": "green", "Required PPE Missing": "red"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No required PPE items selected or no compliance data for selected items.")
    
else:
    st.info("No detection session has been completed yet. Start a session to see results here.")

st.markdown("---")
st.caption("Developed for Smart PPE Compliance System using YOLOv12")
