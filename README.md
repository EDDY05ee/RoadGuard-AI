🚧 RoadGuard AI

AI-Powered Road Hazard Detection and Risk Assessment

RoadGuard AI is an AI-powered road safety prototype designed to detect road hazards from images, assess potential risk, and help users manage and prioritize road hazard reports.

The project combines computer vision, risk assessment, and a dashboard to support safer roads and more organized hazard reporting.

✨ Features

- AI Road Inspection: Analyze uploaded road images using a YOLO-based object detection model.
- Hazard Detection: Detect supported road damage categories, including potholes and road cracks.
- Risk Assessment: Display an indicative risk score and priority for detected hazards.
- Hazard Reporting: Create reports with location, severity, and description details.
- Report Management: Search, filter, review, and update report statuses.
- Location Analytics: Explore recorded hazards and location-based report trends.
- Interactive Dashboard: View key metrics and summaries of hazard reports.
- Database Storage: Save report information locally using SQLite.

🛠️ Technology Stack

- Python
- Streamlit
- Ultralytics YOLO
- Hugging Face Hub
- SQLite
- Pandas
- Pillow

📂 Project Structure

RoadGuard-AI/
├── app.py
├── database.py
├── download_model.py
├── requirements.txt
├── README.md
└── .gitignore

The AI model weights and SQLite database may be created locally when the application runs.

🚀 Installation and Setup

1. Clone the repository

git clone https://github.com/EDDY05ee/RoadGuard-AI.git
cd RoadGuard-AI

2. Create a virtual environment

On Windows, run:

python -m venv .venv

3. Install dependencies

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

4. Download the AI model

Run the model download script:

.\.venv\Scripts\python.exe download_model.py

Ensure the script downloads the model to the location expected by "app.py".

5. Launch the application

.\.venv\Scripts\python.exe -m streamlit run app.py

Open the local URL displayed in your terminal to use the dashboard.

🤖 AI Model

RoadGuard AI uses the following Hugging Face model repository:

"vinothvikas1987/pothole-detection-yolov8" (https://huggingface.co/vinothvikas1987/pothole-detection-yolov8)

Review the model card and license before redistributing model weights or deploying the application. The model may be downloaded separately instead of being stored in GitHub.

🛡️ Safety and Limitations

- This project is a hackathon prototype.
- AI detections may be inaccurate or may miss hazards.
- Risk scores are indicative and should not be treated as official engineering assessments.
- Detected hazards should be reviewed by a person before decisions are made.
- Do not commit passwords, API tokens, private reports, or sensitive personal information.

🔮 Future Improvements

- GPS-based road hazard mapping
- Live hazard reporting and notifications
- Integration with road maintenance authorities
- Improved detection accuracy and severity estimation
- Mobile-friendly reporting
- More advanced location analytics

🎯 Project Goal

The goal of RoadGuard AI is to explore how artificial intelligence can assist with identifying road hazards, organizing reports, and prioritizing potential road safety issues.

👨‍💻 Project Status

Hackathon prototype — under development.