import json
import zipfile
from pathlib import Path
import requests

def download_and_extract(url, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "nfcorpus.zip"
    if not zip_path.exists():
        print(f"Baixando {url} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    else:
        print(f"Zip já existe em {zip_path}, pulando download.")
    dataset_dir = data_dir / "nfcorpus"
    if not (dataset_dir / "corpus.jsonl").exists():
        print("Descompactando ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
    return dataset_dir

def load_corpus(dataset_dir):
    corpus = {}
    with open(Path(dataset_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    return corpus

def doc_text(doc):
    return (doc["title"] + " " + doc["text"]).strip()

def load_queries(dataset_dir):
    queries = {}
    with open(Path(dataset_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries

def load_qrels(dataset_dir, split):
    qrels = {}
    with open(Path(dataset_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        next(f)  # pula o cabeçalho
        for line in f:
            qid, did, grade = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(grade)
    return qrels

def split_queries(queries, qrels):
    return {qid: queries[qid] for qid in sorted(qrels)}
