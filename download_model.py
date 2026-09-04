#!/usr/bin/env python3
"""
Download script for Depth Anything 3 DA3METRIC-LARGE ONNX model.
"""

import os
import sys
import requests
from tqdm import tqdm

MODEL_URL_FP16 = "https://huggingface.co/Heliosoph/da3metric-large-onnx/resolve/main/model_fp16.onnx"
MODEL_CONFIG_URL = "https://huggingface.co/Heliosoph/da3metric-large-onnx/resolve/main/config.json"

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FP16_PATH = os.path.join(MODELS_DIR, "model_fp16.onnx")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File already exists: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return

    print(f"Downloading {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB chunks

    with open(dest_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rProgress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
        print()
    print(f"Successfully downloaded {dest_path}")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_file(MODEL_CONFIG_URL, CONFIG_PATH)
    download_file(MODEL_URL_FP16, FP16_PATH)

