FROM python:3.7

RUN mkdir -p /home/ubuntu/object_detection/

WORKDIR /home/ubuntu/object_detection/

ADD . /home/ubuntu/object_detection/

RUN pip3 install -r /home/ubuntu/object_detection/requirements.txt && pip install coco
RUN apt update
RUN apt install -y libgl1-mesa-dev