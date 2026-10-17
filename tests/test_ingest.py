from legal_agent.ingest import tokenize


def test_arabic_indic_digits_normalized():
    assert tokenize("المادة ١٦٣") == ["المادة", "163"]


def test_mixed_text():
    assert tokenize("وفقا للمادة 163 والمواد ٥") == ["وفقا", "للمادة", "163", "والمواد", "5"]
