import requests, os, time

BASE = "https://api.isic-archive.com/api/v2/images/search/"

SOURCES = {
    "df": 'diagnosis_3:"Dermatofibroma"',
    "vasc": [
        'diagnosis_2:"Hemangioma"',
        'diagnosis_3:"Angiokeratoma"',
        'diagnosis_3:"Pyogenic granuloma"',
    ],
}

def download_query(query, outdir):
    os.makedirs(outdir, exist_ok=True)
    url = BASE
    params = {"query": query, "limit": 50}
    downloaded = 0
    while url:
        resp = requests.get(url, params=params if url == BASE else None)
        if resp.status_code != 200:
            print("ERROR:", resp.status_code, resp.text[:300])
            break
        data = resp.json()
        for item in data["results"]:
            isic_id = item["isic_id"]
            files = item.get("files", {})
            file_url = None
            for key in ("full", "download", "thumbnail_256"):
                if key in files and files[key].get("url"):
                    file_url = files[key]["url"]
                    break
            if not file_url:
                continue
            out_path = os.path.join(outdir, f"{isic_id}.jpg")
            if os.path.exists(out_path):
                downloaded += 1
                continue
            img_resp = requests.get(file_url)
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            downloaded += 1
            if downloaded % 25 == 0:
                print(f"    downloaded {downloaded}...")
            time.sleep(0.05)
        url = data.get("next")
        params = None
    return downloaded

for dx_label, queries in SOURCES.items():
    outdir = f"data/isic_new/{dx_label}"
    if isinstance(queries, str):
        queries = [queries]
    total = 0
    for q in queries:
        print(f"{dx_label}: querying {q}")
        total += download_query(q, outdir)
    print(f"{dx_label} TOTAL: {total} images in {outdir}\n")
