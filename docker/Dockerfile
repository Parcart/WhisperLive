FROM python:3.10-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV NSM_FLAG=""
ARG RabbitMQ_HOST="localhost"
ARG RabbitMQ_PORT=5672

# install lib required for pyaudio
RUN apt update && apt install -y portaudio19-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# update pip to support for whl.metadata -> less downloading
RUN pip install --no-cache-dir -U "pip>=24"

# create a working directory
RUN mkdir /app
WORKDIR /app

 install the requirements for running the whisper-live server
COPY requirements/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
#
#COPY requirements/server_rabbit.txt /app/
#RUN pip install --no-cache-dir -r server_rabbit.txt && rm server_rabbit.txt

# make the paths of the nvidia libs installed as wheels visible. equivalent to:
# export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib"

COPY whisper_live /app/whisper_live
COPY run_server.py /app

CMD ["sh", "-c", "python run_server.py --port 9090 --backend faster_whisper -rh $RabbitMQ_HOST -rp $RabbitMQ_PORT"]
# docker build -f docker/Dockerfile.vast -t whisperlivegpu .

