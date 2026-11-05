import requests

BASE = "https://api.isic-archive.com/api/v2/images/search/"
resp = requests.get(BASE, params={"query": 'diagnosis_3:"Hemangioma"', "limit": 5}, timeout=15)
data = resp.json()
print(f"count={data.get('count')}, results_len={len(data.get('results', []))}")
if data.get("results"):
    print("sample:", data["results"][0].get("isic_id"), data["results"][0].get("files"))
