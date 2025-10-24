#!/bin/bash
CudaVersion="12.2.2"
UbuntuVersion="22.04"
ImageTag="v2"
Dockerfile="Dockerfile.toBuild.cu12.2"

docker build -t maods/cuda-${CudaVersion}-devel-${UbuntuVersion}-pytorch:${ImageTag} -f $Dockerfile .
# docker push maods/cuda-${CudaVersion}-devel-${UbuntuVersion}-pytorch:${ImageTag}
