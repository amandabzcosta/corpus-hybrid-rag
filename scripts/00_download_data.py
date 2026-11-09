import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import download_and_extract, load_corpus, load_qrels, load_queries
from src.io_utils import load_config, project_path

sys.stdout.reconfigure(encoding="utf-8")

def main():
    cfg = load_config()
    ds = cfg["dataset"]
    dataset_dir = download_and_extract(ds["url"], project_path(cfg["paths"]["data_dir"]))

    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    print(f"\nDocumentos: {len(corpus)}")
    print(f"Consultas (queries.jsonl): {len(queries)}")

    assert len(corpus) == ds["expected_docs"], "número de documentos diferente do enunciado"
    assert len(queries) == ds["expected_queries"], "número de consultas diferente do enunciado"

    for split in ["train", "dev", "test"]:
        qrels = load_qrels(dataset_dir, split)
        n_pairs = sum(len(v) for v in qrels.values())
        grades = Counter(g for v in qrels.values() for g in v.values())
        n_rel = [sum(1 for g in v.values() if g > 0) for v in qrels.values()]
        missing_q = [q for q in qrels if q not in queries]
        missing_d = {d for v in qrels.values() for d in v if d not in corpus}
        print(f"\n[{split}] consultas: {len(qrels)} | pares julgados: {n_pairs}")
        print(f"  distribuição dos graus: {dict(sorted(grades.items()))}")
        print(f"  relevantes (grau>0) por consulta: média {sum(n_rel)/len(n_rel):.1f}, "
              f"mín {min(n_rel)}, máx {max(n_rel)}")
        print(f"  consultas ausentes do queries.jsonl: {len(missing_q)} | "
              f"docs ausentes do corpus: {len(missing_d)}")
        if split == "test":
            assert len(qrels) == ds["expected_test_queries"], "test deveria ter 323 consultas"

    empty = [d for d, v in corpus.items() if not (v["title"] + v["text"]).strip()]
    print(f"\nDocumentos com título+texto vazio: {len(empty)}")
    print("\nOK: contagens validadas.")

if __name__ == "__main__":
    main()
