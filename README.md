RoadGuard AI 🚧

AI-powered road hazard detection, risk assessment, and reporting dashboard.

RoadGuard AI is a prototype designed to help identify road hazards using computer vision and organize reports so that potential road safety issues can be reviewed and prioritized.

Features

- AI Road Inspection: Analyze uploaded road images using a YOLO-based object detection model.
- Hazard Detection: Detect supported road damage categories such as potholes and road cracks.
- Risk Assessment: Display a risk score and suggested action to help prioritize reported hazards.
- Manual Reporting: Submit road hazard reports with location and description details.
- Report Management: Search, filter, review, and update report statuses.
- Location Analytics: Explore recorded reports and hazard trends.
- Dashboard: View key metrics and summaries of reported hazards.
- Data Storage: Save report information in a local SQLite database.

Technology Stack

- Python
- Streamlit
- Ultralytics YOLO
- Hugging Face model repository
- SQLite
- Pandas
- Pillow

Getting Started

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/RoadGuard-AI.git
cd RoadGuard-AI

Replace "YOUR-USERNAME" with your GitHub username.

2. Create and activate a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Download the AI model

If the project includes "download_model.py", run:

python download_model.py

Make sure the model download script and "app.py" use the same model path.

5. Start the application

python -m streamlit run app.py

Open the local URL displayed in your terminal.

Model

This project uses the "vinothvikas1987/pothole-detection-yolov8" model from Hugging Face. Refer to the model card and its license before redistribution or deployment.

Important Notes

- This is a prototype intended to assist road hazard reporting and prioritization.
- AI predictions can be inaccurate or miss hazards. Results should be reviewed by a person.
- Risk scores are indicative and should not be treated as official engineering assessments.
- Keep private information and real-world sensitive reports out of public repositories.
- The model weights may need to be downloaded separately before running the application.

Future Improvements

- GPS-based hazard mapping
- Integration with official road maintenance workflows
- Improved detection and severity estimation
- Mobile-friendly reporting
- Notifications for high-priority hazards

Project Status

Hackathon prototype — under development.