import os
import urllib.request
import zipfile
import tarfile
import platform
import subprocess
from pathlib import Path
import requests
import sys

def download_file(url, save_path, desc=None):
    """Download a file with progress reporting"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading {desc or os.path.basename(save_path)}...")
    downloaded = 0
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            file.write(data)
            done = int(50 * downloaded / total_size)
            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}/{total_size/1024/1024:.2f} MB")
            sys.stdout.flush()
    sys.stdout.write('\n')
    print(f"Downloaded {desc or os.path.basename(save_path)}")

def download_yolo():
    """Download YOLOv8 model"""
    models_dir = os.path.join("app", "models", "yolo")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download YOLOv8n model
    yolo_model_path = os.path.join(models_dir, "yolov8n.pt")
    if not os.path.exists(yolo_model_path):
        print("Downloading YOLOv8n model...")
        download_file(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            yolo_model_path,
            "YOLOv8n model"
        )
    else:
        print("YOLOv8n model already exists.")

def download_deepsort():
    """Download DeepSORT model"""
    models_dir = os.path.join("app", "models", "deep_sort")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download DeepSORT model
    deepsort_model_path = os.path.join(models_dir, "ckpt.t7")
    if not os.path.exists(deepsort_model_path):
        print("Downloading DeepSORT model...")
        download_file(
            "https://github.com/ZQPei/deep_sort_pytorch/releases/download/v1.0.2/ckpt.t7",
            deepsort_model_path,
            "DeepSORT model"
        )
    else:
        print("DeepSORT model already exists.")

def download_llm():
    """Download LLM model"""
    models_dir = os.path.join("app", "models", "llm")
    os.makedirs(models_dir, exist_ok=True)
    
    # For DeepSeek-Coder model, we'll use a smaller version for local execution
    llm_model_path = os.path.join(models_dir, "deepseek-coder-1.3b-instruct.Q4_K_M.gguf")
    if not os.path.exists(llm_model_path):
        print("Downloading DeepSeek-Coder LLM model (1.3B)...")
        download_file(
            "https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
            llm_model_path,
            "DeepSeek-Coder LLM model"
        )
    else:
        print("DeepSeek-Coder LLM model already exists.")

def create_model_dirs():
    """Create model directories"""
    os.makedirs(os.path.join("app", "models"), exist_ok=True)
    os.makedirs(os.path.join("app", "data"), exist_ok=True)

def main():
    """Main function to download all models"""
    print("Setting up Vision AI models...")
    create_model_dirs()
    download_yolo()
    download_deepsort()
    download_llm()
    print("All models have been downloaded successfully!")

if __name__ == "__main__":
    main() 