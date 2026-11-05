import requests, os, time

BASE = "https://api.isic-archive.com/api/v2/images/search/"
OUTDIR = "data/isic_new/vasc"
os.makedirs(OUTDIR, exist_ok=True)

url = BASE
params = {"query": 'diagnosis_3:"Hemangioma"', "limit": 50}
downloaded = 0
failed = []

while url:
    resp = requests.get(url, params=params if url == BASE else None, timeout=15)
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
        out_path = os.path.join(OUTDIR, f"{isic_id}.jpg")
        if os.path.exists(out_path):
            downloaded += 1
            continue
        try:
            img_resp = requests.get(file_url, timeout=15)
            img_resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            downloaded += 1
            if downloaded % 25 == 0:
                print(f"  downloaded {downloaded}...")
        except Exception as e:
            failed.append(isic_id)
            print(f"  FAILED {isic_id}: {type(e).__name__}")
        time.sleep(0.05)
    url = data.get("next")
    params = None

print(f"\nHemangioma download complete: {downloaded} images")
if failed:
    print(f"Failed ({len(failed)}): {failed}")
