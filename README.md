The enforcement of Personal Protective Equipment (PPE) compliance in industrial and
construction environments is a critical but often challenging task. Traditional manual
monitoring methods are prone to human error, inconsistency, and inefficiency, which can lead
to significant safety risks, accidents, and regulatory violations. This thesis presents the
design, development, and evaluation of a robust, real-time Smart PPE Compliance System
aimed at addressing these limitations.
The proposed solution leverages a state-of-the-art YOLOv12-based object detection model,
which was trained on a custom dataset of seven key PPE classes. The end-to-end system
integrates a live video feed from a webcam, performs real-time inference, and provides a
user-friendly interface via a Streamlit application. The system's performance was rigorously
evaluated using a comprehensive suite of metrics. The model achieved a high mean Average
Precision (mAP@0.5) of 0.908, demonstrating its strong ability to accurately detect and
localize PPE items. While the model showed excellent performance across most classes, a
detailed analysis identified a key limitation in detecting the Shield class due to its transparent
nature and the complexity of visual features.
In conclusion, this project successfully demonstrates the feasibility and effectiveness of an
automated, AI-driven approach to workplace safety. By providing a scalable and highly
accurate solution that enhances real-time oversight and data-driven analysis, this work offers
a significant contribution to improving safety standards and mitigating risks in industrial
settings.
