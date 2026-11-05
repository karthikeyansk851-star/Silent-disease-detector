import requests, os, time

BASE = "https://api.isic-archive.com/api/v2/images/search/"
QUERY = 'diagnosis_3:"Squamous cell carcinoma, Invasive"'
OUTDIR = "data/isic_new/scc"

os.makedirs(OUTDIR, exist_ok=True)

url = BASE
params = {"query": QUERY, "limit": 50}
total_downloaded = 0

while url:
    resp = requests.get(url, params=params if url == BASE else None)
    if resp.status_code != 200:
        print("ERROR:", resp.status_code, resp.text[:300])
        break
    data = resp.json()

    if total_downloaded == 0 and data["results"]:
        print("Sample result keys:", list(data["results"][0].keys()))
        print("Sample files field:", data["results"][0].get("files"))

    for item in data["results"]:
        isic_id = item["isic_id"]
        files = item.get("files", {})
        file_url = None
        for key in ("full", "download", "thumbnail_256"):
            if key in files and files[key].get("url"):
                file_url = files[key]["url"]
                break
        if not file_url:
            print(f"  no downloadable url for {isic_id}, skipping")
            continue

        out_path = os.path.join(OUTDIR, f"{isic_id}.jpg")
        if os.path.exists(out_path):
            total_downloaded += 1
            continue

        img_resp = requests.get(file_url)
        with open(out_path, "wb") as f:
            f.write(img_resp.content)
        total_downloaded += 1
        if total_downloaded % 25 == 0:
            print(f"  downloaded {total_downloaded}...")
        time.sleep(0.05)

    url = data.get("next")
    params = None  # next url already has query params baked in

print(f"Done. Total downloaded: {total_downloaded}")
