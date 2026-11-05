import requests

BASE = "https://api.isic-archive.com/api/v2/images/search/"

queries = {
    "SCC (Invasive)": 'diagnosis_3:"Squamous cell carcinoma, Invasive"',
    "Solar lentigo": 'diagnosis_3:"Solar lentigo"',
    "Pigmented benign keratosis": 'diagnosis_3:"Pigmented benign keratosis"',
    "Dysplastic nevus": 'diagnosis_4:"Nevus, Atypical, Dysplastic, or Clark"',
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