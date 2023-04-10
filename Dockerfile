FROM tiangolo/uwsgi-nginx-flask:python3.10

RUN apt-get update && apt-get install -y locales && rm -r /var/lib/apt/lists/*
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && dpkg-reconfigure --frontend=noninteractive locales

COPY ./app/predict_server.py ./
COPY ./shared_libs/ ./../shared_libs/

COPY ./requirements.txt ./
COPY ./uwsgi.ini ./

RUN pip install -r ./requirements.txt