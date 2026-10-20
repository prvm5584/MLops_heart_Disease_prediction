import json,urllib.request
payload=json.dumps({'age':54,'sex':1,'cp':4,'trestbps':130,'chol':246,'fbs':0,'restecg':2,'thalach':150,'exang':0,'oldpeak':1.2,'slope':2,'ca':0,'thal':3}).encode()
print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())
req=urllib.request.Request('http://127.0.0.1:8000/predict',data=payload,headers={'Content-Type':'application/json'},method='POST'); print(urllib.request.urlopen(req).read().decode()); print('metrics',urllib.request.urlopen('http://127.0.0.1:8000/metrics').status)
