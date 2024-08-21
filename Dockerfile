FROM python:3.10.14-slim

ENV PROJ_DIR=/my_project
RUN mkdir -p ${PROJ_DIR}
WORKDIR ${PROJ_DIR}

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip
COPY ./requirements.txt ${PROJ_DIR}/requirements.txt
RUN pip install -r ${PROJ_DIR}/requirements.txt
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY ./src ${PROJ_DIR}/src
COPY ./main.py ${PROJ_DIR}/main.py
COPY ./sampleCaptchas ${PROJ_DIR}/sampleCaptchas
COPY ./models ${PROJ_DIR}/models

RUN mkdir -p ${PROJ_DIR}/results

CMD [ "python", "./main.py"]
