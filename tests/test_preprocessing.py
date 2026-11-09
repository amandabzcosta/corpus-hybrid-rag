import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import Preprocessor

def test_pipeline_basico():
    p = Preprocessor(stopwords_keep=["d"])
    assert p.tokenize("The Effects of Vitamin D on Omega-3 levels!") == ["effects", "vitamin", "d", "omega", "3", "levels"]
    assert p("The Effects of Vitamin D on Omega-3 levels!") == ["effect", "vitamin", "d", "omega", "3", "level"]

def test_mesma_funcao_doc_e_consulta():
    p = Preprocessor()
    assert p("Treating Asthma") == p("treating   ASTHMA.")

def test_sem_numeros_e_sem_stemming():
    p = Preprocessor(stemmer="none", keep_numbers=False)
    assert p("95% CI in 2010 for b12 users") == ["ci", "b12", "users"]

def test_analyze_guarda_token_original():
    p = Preprocessor()
    assert p.analyze("running studies") == [("running", "run"), ("studies", "studi")]

def test_unicode_nfkc():
    p = Preprocessor()
    assert p.tokenize("10 µg β-carotene") == ["10", "μg", "β", "carotene"]
