from __future__ import annotations

import shutil
import urllib.request
import zipfile
from collections import Counter

from _common import ROOT, argparse  # noqa: F401

from src.config import DATA_DIR
from src.data import load_all

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
MD5_BEIR = "a89dba18a62ef92f7d323ec890a0d38d"   # md5 do nfcorpus.zip publicado no README do BEIR

def main():
    ap = argparse.ArgumentParser(description='Baixa e valida o NFCorpus (versão BEIR) em data/nfcorpus/.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--from-dir", default=None)
    args = ap.parse_args()
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if args.from_dir:
        shutil.copytree(args.from_dir, DATA_DIR, dirs_exist_ok=True)
    else:
        path = args.zip
        if path is None:
            path = str(DATA_DIR.parent / "nfcorpus.zip")
            print("baixando", URL)
            urllib.request.urlretrieve(URL, path)
        import hashlib
        md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
        print('md5 do zip:', md5, '(confere com o BEIR)' if md5 == MD5_BEIR else '(DIFERE do md5 publicado pelo BEIR; verifique o arquivo)')
        with zipfile.ZipFile(path) as z:
            z.extractall(DATA_DIR.parent)
    corpus, queries, qrels, info = load_all(validate=True)
    print("OK — dataset validado:", info["stats"])
    for split, qr in qrels.items():
        grades = Counter(g for v in qr.values() for g in v.values())
        n_rel = [sum(1 for g in v.values() if g > 0) for v in qr.values()]
        print(f"[{split}] graus: {dict(sorted(grades.items()))} | relevantes por consulta: "
              f"média {sum(n_rel) / len(n_rel):.1f}, mín {min(n_rel)}, máx {max(n_rel)}")
    print("consultas do RAG (seed 42):", info["rag_ids"])

if __name__ == "__main__":
    main()
