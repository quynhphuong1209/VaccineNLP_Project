# -*- coding: utf-8 -*-
import pytest
import sys
from pathlib import Path

# Add api_service path so tests can run outside package root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.hardening import canonicalize, obfuscation_report

def test_homoglyph():
    # Cyrillic 'а' and 'е' looking like Latin 'a' and 'e'
    text = "vаccinе"  # containing Cyrillic characters
    canon = canonicalize(text)
    assert "vaccine" in canon or "vacxin" in canon
    
    report = obfuscation_report(text)
    assert report["flags"]["confusable"] is True
    assert report["level"] in ("low", "high")

def test_zero_width():
    # zero-width space characters inserted
    text = "t\u200bi\u200bê\u200bm"
    canon = canonicalize(text)
    assert "tiêm" in canon
    
    report = obfuscation_report(text)
    assert report["flags"]["zero_width"] is True
    assert report["level"] == "high"

def test_leet():
    text = "v4cc1n3"
    canon = canonicalize(text)
    assert "vaccine" in canon
    
    report = obfuscation_report(text)
    assert report["flags"]["leet"] is True
    assert report["level"] in ("low", "high")

def test_separator():
    text = "v.a.c.x.i.n"
    canon = canonicalize(text)
    assert "vacxin" in canon.replace(" ", "")
    
    report = obfuscation_report(text)
    assert report["flags"]["separator"] is True
    assert report["level"] in ("low", "high")

def test_spacing():
    text = "t i ê m"
    canon = canonicalize(text)
    assert "tiêm" in canon.replace(" ", "")
    
    report = obfuscation_report(text)
    assert report["flags"]["spacing"] is True
    assert report["level"] in ("low", "high")

def test_combo():
    text = "v_4_c_c_1_n_e"
    canon = canonicalize(text)
    assert "vaccine" in canon.replace(" ", "") or "vacxin" in canon.replace(" ", "")
    
    report = obfuscation_report(text)
    assert report["level"] == "high"

def test_clean_texts():
    text1 = "Tiêm vaccine rất tốt cho trẻ"
    assert canonicalize(text1) == "Tiêm vaccine rất tốt cho trẻ"
    report1 = obfuscation_report(text1)
    assert report1["level"] == "none"
    
    # Numbers should not be incorrectly de-leeted when no vaccine keyword is formed/adjacent
    text2 = "Tôi mua 5 liều cho 3 bé"
    assert canonicalize(text2) == "Tôi mua 5 liều cho 3 bé"
    report2 = obfuscation_report(text2)
    assert report2["level"] == "none"

def test_regression_deleet_false_positive():
    # Should not de-leet '4' just because 'mùi' (vaccine keyword) is in the same sentence
    text = "Tôi có 4 chai khử mùi thơm"
    canon = canonicalize(text)
    assert "4 chai" in canon
    assert "mùi" in canon
    
    report = obfuscation_report(text)
    assert report["flags"]["leet"] is False

def test_idempotent():
    cases = [
        "v4cc1n3 g4y v0 s1nh",
        "t\u200bi\u200bê\u200bm",
        "Tôi mua 5 liều cho 3 bé",
        "v.a.c.x.i.n",
        "Tiêm vaccine rất tốt cho trẻ"
    ]
    for c in cases:
        c1 = canonicalize(c)
        c2 = canonicalize(c1)
        assert c1 == c2
