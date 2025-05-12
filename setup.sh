#!/bin/bash

# Navigate to the user's home directory
cd ~

# Clone the FramePack repository from GitHub
git clone https://github.com/sooxt98/FramePack

# Change the current directory to the FramePack directory
cd FramePack

# Install PyTorch, TorchVision, and TorchAudio with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# speed boost
pip install sageattention==1.0.6

# for extend.py
pip install decord

# Install the Python dependencies listed in requirements.txt
pip install -r requirements.txt

# Run the Gradio demo script with sharing enabled
python gradio_demo_f1.py --port 8888