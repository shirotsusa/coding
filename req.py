import os, json, hashlib, requests
from pathlib import Path

url = "YOUR_URL"
headers = {"Authorization": "Bearer ***"}  # 実際の値で
payload = {"x": 1}

print("=== runtime ===")
print("cwd:", os.getcwd())
print("python:", os.sys.executable)
for k in ["HTTP_PROXY","HTTPS_PROXY","NO_PROXY","REQUESTS_CA_BUNDLE","SSL_CERT_FILE"]:
    print(f"{k}:", os.getenv(k))

for f in [".env", "config.yaml", "cert.pem"]:
    p = Path(f).resolve()
    print(f"{f}: exists={p.exists()} path={p}")

print("\n=== request ===")
print("url:", url)
print("payload_sha256:", hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())

res = requests.post(url, headers=headers, json=payload, timeout=20)
print("\n=== response ===")
print("status:", res.status_code)
print("content-type:", res.headers.get("Content-Type"))
print("len:", len(res.content))
print("head(bytes):", res.content[:120])
print("head(text):", repr(res.text[:200]))

# JSONを安全に読む
txt = res.text.lstrip("\ufeff").strip()  # BOM対策
if res.status_code == 204 or txt == "":
    data = None
elif "application/json" in (res.headers.get("Content-Type") or "").lower():
    data = json.loads(txt)
else:
    raise RuntimeError("JSON以外が返却されています")
print("parsed:", type(data))
