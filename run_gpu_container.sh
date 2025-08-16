#!/bin/bash

#buid docker image
# docker build -t retrieval-gpu-experiments .

# run docker image
docker run -it --rm --gpus all \
  --shm-size=16g \
  -v "$(pwd)":/workspaces/research-template \
  retrieval-gpu-experiments 
