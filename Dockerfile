FROM ubuntu:20.04

RUN apt update && \
    apt install -y --no-install-recommends software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y --no-install-recommends python3.9 python3.9-distutils libpq-dev postgresql-client postgresql-client-common git

WORKDIR /app

COPY requirements.txt .

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py && \
    python3.9 -m pip install -r requirements.txt && \
    python3.9 -m pip install jupyter==1.0.0

COPY . .

CMD python3.9 -m jupyter notebook --allow-root --port 8888 --ip '0.0.0.0'
