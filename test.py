# test.py

import requests

url = "http://127.0.0.1:8000/describe-image"
files = {"file": open("sample.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())