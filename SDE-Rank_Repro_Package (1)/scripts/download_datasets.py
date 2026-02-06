import argparse, os, urllib.request, gzip, shutil

SNAP_URLS = {
    "email-eu-core": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "wiki-vote": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
    "soc-epinions1": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
}

def download(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)

def gunzip(gz_path: str, out_path: str):
    print(f"Extracting {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--dataset", default="email-eu-core", choices=list(SNAP_URLS.keys())+["all"])
    args = ap.parse_args()

    datasets = list(SNAP_URLS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        url = SNAP_URLS[ds]
        gz_path = os.path.join(args.out, f"{ds}.txt.gz")
        txt_path = os.path.join(args.out, f"{ds}.txt")
        download(url, gz_path)
        gunzip(gz_path, txt_path)

if __name__ == "__main__":
    main()
