from src.preprocess import Preprocessor, raw_tokens

def test_mesma_funcao_doc_e_consulta():
    p = Preprocessor(stemmer="snowball")
    assert p("Treating Asthma") == p("treating   ASTHMA.")

def test_tokens_antes_do_stemming():
    assert raw_tokens("Running Studies on Omega-3 Café") == ["running", "studies", "on", "omega", "3", "cafe"]
    assert Preprocessor(stemmer="porter")("Running Studies on Omega-3") == ["run", "studi", "omega", "3"]

def test_sem_stopwords_e_sem_stemming():
    assert Preprocessor(remove_stopwords=False, stemmer=None)("The Effects") == ["the", "effects"]
