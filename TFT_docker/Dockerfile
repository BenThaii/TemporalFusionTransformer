# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.1-cpu-py36-ubuntu16.04

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY TFT.py /opt/ml/code/TFT.py

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt

# Defines cifar10.py as script entrypoint 
# ENV SAGEMAKER_PROGRAM TFT.py
ENTRYPOINT ["python", "/opt/ml/code/TFT.py"]