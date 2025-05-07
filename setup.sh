#!/bin/bash

# Navigate to the user's home directory
cd ~

# Clone the FramePack repository from GitHub
git clone https://github.com/sooxt98/FramePack

# Change the current directory to the FramePack directory
cd FramePack

# Install PyTorch, TorchVision, and TorchAudio with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install sageattention==1.0.6

# Install the Python dependencies listed in requirements.txt
pip install -r requirements.txt

# Run the Gradio demo script with sharing enabled
python demo_gradio_f1.py --port 8888