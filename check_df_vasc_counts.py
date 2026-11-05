import requests

BASE = "https://api.isic-archive.com/api/v2/images/search/"

queries = {
    "Dermatofibroma": 'diagnosis_3:"Dermatofibroma"',
    "Vascular lesion": 'diagnosis_2:"Vascular lesions"',
    "Hemangioma": 'diagnosis_3:"Hemangioma"',
    "Angioma": 'diagnosis_3:"Angioma"',
}

for name, q in queries.items():
    resp = requests.get(BASE, params={"query": q, "limit": 1})
    if resp.status_code != 200:
        print(f"{name}: ERROR {resp.status_code} - {resp.text[:200]}")
        continue
    data = resp.json()
    print(f"{name}: {data.get('count', 'no count field - see raw below')}")
    if 'count' not in data:
        print("   raw keys:", list(data.keys()))
