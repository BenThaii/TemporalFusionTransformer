#! /bin/bash

algorithm_name=pytorch-tft-container-test

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# clean up docker resources on local machine
docker system prune -f
docker rmi $(docker images | grep ${algorithm_name})         # delete previous version of the image prior to building the new one (save space)

# build new docker image
docker build -t ${algorithm_name} .