
#! /bin/bash

s3_location=s3://sagemaker-us-east-1-551329315830/tensorboard/tft-pytorch-spot-1-2022-08-22-10-05-32-622/tensorboard-output/TemporalFusionTransformer/


# download tensorboard from S3
aws s3 cp --recursive ${s3_location}  ./logs/fit

# launch tensorboard on local instance
tensorboard --logdir logs/fit