from celery import Celery
import traceback
import requests
import ast
import time

from args import batchsize, imgsize, vehicle_model_endpoint,fsm_model_endpoint

celeryserver = Celery('celeryworker', backend='rpc://', broker='amqp://localhost:5672')
celeryserver.config_from_object('celeryconfig')


@celeryserver.task(ignore_result=True)
def send_data(data):
    headers = {
    "Content-Type": 'application/json',
    "Accept": 'application/json',
    "ApplicationId": '0x01', 
    "SessionToken": '0x01',
    "UserId": '0x01', 
    "CustId": '',
    "MessageId": 'message_id_',
    }    
    headers["MessageId"] += time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(time.time()))

    f = ''  # URL End point

    print(data)
    print(headers)
    count = 0
    while True:
        response = requests.put(f, headers=headers, data=str(data))
        print(response.content)
        if response.status_code == 200:
            print('SENT: <Response 200>')
            break
        else:
            count += 1

            if count > 5:
                break
            
            print(response.status_code)
            time.sleep(1)

