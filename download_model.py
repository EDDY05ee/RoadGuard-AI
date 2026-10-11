from huggingface_hub import hf_hub_download

print("Downloading the road damage AI model...")

model_path = hf_hub_download(
    repo_id="vinothvikas1987/pothole-detection-yolov8",
    filename="best.pt"
)

print("Download complete!")
print("Model saved at:", model_path)