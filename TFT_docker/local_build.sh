#! /bin/bash

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# clean up docker resources on local machine
docker system prune -y

# build new docker image
docker build -t pytorch-tft-container-test .