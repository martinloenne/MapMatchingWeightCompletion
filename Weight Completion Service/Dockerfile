FROM python:3.6-slim-buster

RUN apt update \
  && apt install -y python3-pip \
  && apt install -y build-essential libssl-dev libffi-dev python3-dev

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "-u", "app.py", "prod"]

