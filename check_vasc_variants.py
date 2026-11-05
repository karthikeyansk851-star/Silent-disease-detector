import requests

BASE = "https://api.isic-archive.com/api/v2/images/search/"

queries = {
    "diagnosis_1=Vascular": 'diagnosis_1:"Vascular lesions"',
    "diagnosis_2=Vascular": 'diagnosis_2:"Vascular lesions"',
    "diagnosis_3=Vascular lesion": 'diagnosis_3:"Vascular lesion"',
    "diagnosis_2=Hemangioma": 'diagnosis_2:"Hemangioma"',
    "diagnosis_3=Cherry angioma": 'diagnosis_3:"Cherry angioma"',
    "diagnosis_3=Angiokeratoma": 'diagnosis_3:"Angiokeratoma"',
    "diagnosis_3=Pyogenic granuloma": 'diagnosis_3:"Pyogenic granuloma"',
}

for name, q in queries.items():
    resp = requests.get(BASE, params={"query": q, "limit": 1})
    if resp.status_code != 200:
        print(f"{name}: ERROR {resp.status_code} - {resp.text[:200]}")
        continue
    data = resp.json()
    print(f"{name}: {data.get('count', 'no count field')}")
