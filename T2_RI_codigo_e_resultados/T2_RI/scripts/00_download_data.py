"""Baixa e valida o NFCorpus (versão BEIR) em data/nfcorpus/.

  python scripts/00_download_data.py                       # baixa o zip oficial do BEIR
  python scripts/00_download_data.py --zip nfcorpus.zip    # usa um zip já baixado
  python scripts/00_download_data.py --from-dir pasta/     # copia de uma pasta com corpus.jsonl, queries.jsonl e qrels/
Ao final, confere as estatísticas oficiais (3.633 documentos, 3.237 consultas, 323 consultas no test, 324 no dev)
e se o código do enunciado reproduz as 10 consultas do experimento de RAG."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile

from _common import ROOT, argparse  # noqa: F401

from src.config import DATA_DIR
from src.data import load_all

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
MD5_BEIR = "a89dba18a62ef92f7d323ec890a0d38d"   # md5 do nfcorpus.zip publicado no README do BEIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    print("consultas do RAG (seed 42):", info["rag_ids"])


if __name__ == "__main__":
    main()
