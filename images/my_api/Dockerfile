FROM ubuntu:20.04

ADD files/ /server/

WORKDIR /server/

RUN ls && apt update && apt install python3-pip -y && pip install -r requirements.txt

EXPOSE 8000

CMD ls && uvicorn main:app --host 0.0.0.0
