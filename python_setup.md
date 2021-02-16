# Installing Python Dependeceis

## Guide for setting up a Virtual Env

```bash
sudo apt install -y python3-venv
mkdir environments
cd environments
python3 -m venv my_env

```


## Gunicorn
Server setup for running Model

```bash
pip3 install gunicorn
```
Start Gunicorn Server
```bash
gunicorn detector:flaskserver --bind=127.0.0.1:8080 --threads=16 --timeout=100
```

## Celery
Celery Server setup for Queue
```bash
pip3 install celery
sudo apt-get install rabbitmq-server
```

Start Server
```bash
celery --app=taskworkers worker --loglevel=info -E
```

Clear celery Queue
```bash
celery purge
```

## Pytorch Setup
Pytorch for Cuda 11.0
```bash 
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
## Basic Dependeceis
```bash 
pip3 install requests
pip3 install flask
pip3 install keras
pip3 install matplotlib
pip3 install scikit-image
pip install scikit-learn==0.22.2
pip3 install filterpy
pip3 install shapely
```

## OPENCV build from source for gstreamer
https://www.learnopencv.com/install-opencv-4-on-ubuntu-18-04/




