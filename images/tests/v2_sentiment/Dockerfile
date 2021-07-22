FROM ubuntu:20.04
ADD files/main.py /server/main.py
WORKDIR /server/
RUN apt update && apt install python3-pip -y && pip install pytest requests
CMD python3 -m pytest main.py
