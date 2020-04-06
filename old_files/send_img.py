import requests
import time
url = 'http://192.168.1.34:5000/'

for _ in range(5):
    try:
        with open('test.jpg','rb') as f:
            r = requests.post(url, f.read())
    except:
        time.sleep(3)
