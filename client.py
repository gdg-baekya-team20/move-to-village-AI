import requests

url = "http://0.0.0.0:8000/generate"
data = {
    "purpose": "귀농",
    "priorities": "교통",
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Generated Text:", response.json()["generated_text"])
else:
    print("Error:", response.status_code, response.json())
