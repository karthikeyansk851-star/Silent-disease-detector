import requests

BASE = "https://api.isic-archive.com/api/v2/images/search/"

def test(field, value):
    q = f'{field}:"{value}"'
    print(f"Testing: {q}")
    try:
        resp = requests.get(BASE, params={"query": q, "limit": 5}, timeout=15)
        data = resp.json()
        print(f"  count={data.get('count')}, results_len={len(data.get('results', []))}")
        if data.get("results"):
            sample = data["results"][0]
            print("  sample id:", sample.get("isic_id"))
            print("  sample files:", sample.get("files"))
    except requests.exceptions.Timeout:
        print("  TIMED OUT after 15s")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    print()

test("diagnosis_3", "Angiokeratoma")
test("diagnosis_2", "Hemangioma")
test("diagnosis_1", "Hemangioma")
