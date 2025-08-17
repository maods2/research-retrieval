FROM maods/cuda-12.6.3-devel-ubuntu22.04-pytorch:latest

RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install ruamel.yaml==0.18.14 timm==1.0.15

