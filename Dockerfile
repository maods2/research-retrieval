FROM maods/cuda-12.6.3-devel-ubuntu22.04-pytorch:v2

RUN apt-get update && apt-get install -y unzip
RUN pip install gdown