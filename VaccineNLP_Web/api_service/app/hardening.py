# -*- coding: utf-8 -*-
"""Hardening module for VaccineNLP Web API (Character Evasion Resistance)."""

import os
import re
import json
import logging
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Safe Defaults ---
DEFAULT_CONFUSABLES = {
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "ԁ": "d", "һ": "h", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S",
    "Т": "T", "Х": "X", "ο": "o", "ρ": "p"
}
DEFAULT_ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060", "\ufeff"]
DEFAULT_LEET_MAP = {"4": "a", "1": "i", "3": "e", "0": "o", "5": "s"}
DEFAULT_SEPARATORS = [".", "-", "_", "·", "•", "*", "~"]

def load_tables() -> dict:
    """Load hardening tables from JSON with fallback path discovery."""
    env_path = os.environ.get("HARDENING_TABLES_PATH", "").strip()
    paths_to_try = []
    if env_path:
        paths_to_try.append(Path(env_path))
    
    here = Path(__file__).resolve()
    # Fallback 1: VaccineNLP_Web/data/hardening_tables.json (relative to this module)
    paths_to_try.append(here.parents[2] / "data" / "hardening_tables.json")
    # Fallback 2: VaccineNLP_Web/data/hardening_tables.json (relative to CWD)
    paths_to_try.append(Path.cwd() / "VaccineNLP_Web" / "data" / "hardening_tables.json")
    # Fallback 3: data/hardening_tables.json (relative to CWD)
    paths_to_try.append(Path.cwd() / "data" / "hardening_tables.json")

    for p in paths_to_try:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded hardening tables from {p}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load hardening tables from {p}: {e}")
                
    logger.warning("No hardening tables found. Using defaults.")
    return {}

# Load tables
tables = load_tables()
confusables = tables.get("confusables", DEFAULT_CONFUSABLES)
zero_width = tables.get("zero_width", DEFAULT_ZERO_WIDTH)
leet_map = tables.get("leet_map", DEFAULT_LEET_MAP)
intra_word_separators = tables.get("intra_word_separators", DEFAULT_SEPARATORS)

# --- Vaccine Keywords for Gated De-leet ---
VACCINE_KEYWORDS = {
    "vaccine", "vacxin", "vắc xin", "tiêm", "tiêm chủng",
    "mũi", "bác sĩ", "bệnh", "y tế", "thuốc",
    "phòng dịch", "dịch bệnh", "cúm", "sởi", "hpv", "covid", "miễn dịch"
}

def _strip_accents_lower(s: str) -> str:
    folded = unicodedata.normalize("NFD", s or "")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.lower().replace("đ", "d")

VACCINE_KEYWORDS_UNACCENTED = {
    _strip_accents_lower(kw) for kw in VACCINE_KEYWORDS
}
VACCINE_KEYWORDS_UNACCENTED.update(["vaccine", "vacxin", "tiem", "covid", "mui", "tiem chung", "y te", "mien dich"])

def deleet_string(s: str, l_map: dict) -> str:
    return "".join(l_map.get(c, c) for c in s)

def de_leet_gated(text: str, l_map: dict) -> str:
    """De-leet text token by token only if it matches vaccine keywords."""
    tokens = re.split(r'(\s+)', text)
    new_tokens = []
    for token in tokens:
        if not token or token.isspace():
            new_tokens.append(token)
            continue
        
        deleeted_token = deleet_string(token, l_map)
        
        # Calculate flat form
        # 1. de-leet (done above)
        # 2. remove intra_word_separators
        flat_token = deleeted_token
        for sep in intra_word_separators:
            flat_token = flat_token.replace(sep, "")
        
        # 3. _strip_accents_lower
        flat_token = _strip_accents_lower(flat_token)
        
        has_vaccine = any(kw in flat_token for kw in VACCINE_KEYWORDS_UNACCENTED)
        if has_vaccine:
            new_tokens.append(deleeted_token)
        else:
            new_tokens.append(token)
            
    return "".join(new_tokens)

def collapse_separators(text: str, separators: list) -> str:
    """Merge intra-word separators (like dots or spaces) within tokens."""
    for sep in separators:
        if sep == " ":
            # Space collapse logic for single letters (e.g. "v a c x i n" -> "vacxin")
            parts = re.split(r'\s{2,}', text)
            new_parts = []
            for part in parts:
                tokens = part.split(" ")
                if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
                    new_parts.append("".join(tokens))
                else:
                    new_tokens = []
                    temp_chain = []
                    for tok in tokens:
                        if len(tok) == 1:
                            temp_chain.append(tok)
                        else:
                            if temp_chain:
                                new_tokens.append("".join(temp_chain))
                                temp_chain = []
                            new_tokens.append(tok)
                    if temp_chain:
                        new_tokens.append("".join(temp_chain))
                    new_parts.append(" ".join(new_tokens))
            text = "  ".join(new_parts)
        else:
            # Collapse punctuation separators (e.g. "t.i.ê.m" -> "tiêm")
            escaped_sep = re.escape(sep)
            pattern = rf'(\w){escaped_sep}(?=\w)'
            old_text = ""
            while old_text != text:
                old_text = text
                text = re.sub(pattern, r'\1', text)
    return text

def canonicalize(text: str) -> str:
    """Convert text to canonical form (NFKC -> zero-width removal -> homoglyph -> de-leet -> collapse sep -> NFC)."""
    if not text:
        return ""
    
    # 1. NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Remove zero-width characters
    for zw in zero_width:
        text = text.replace(zw, "")
        
    # 3. Fold confusables character by character
    text = "".join(confusables.get(c, c) for c in text)
    
    # 4. De-leet GATED
    text = de_leet_gated(text, leet_map)
    
    # 5. Collapse intra_word_separators in tokens
    text = collapse_separators(text, intra_word_separators)
    
    # 6. NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    return text

# Vietnamese letters list for non-Vietnamese Latin detection
VN_LETTERS = set("abcdefghijklmnopqrstuvwxyzăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZĂÂĐÊÔƠƯÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ")

def obfuscation_report(text: str) -> dict:
    """Generate obfuscation metric report for a text."""
    if not text:
        return {
            "score": 0.0,
            "level": "none",
            "flags": {
                "zero_width": False,
                "confusable": False,
                "leet": False,
                "separator": False,
                "spacing": False,
                "non_vn_latin_ratio": 0.0
            }
        }
        
    has_zw = any(zw in text for zw in zero_width)
    has_conf = any(conf in text for conf in confusables)
    
    # Leet detection: check if de-leet gate substituted any leet characters
    deleeted_txt = de_leet_gated(text, leet_map)
    orig_leet_count = sum(1 for c in text if c in leet_map)
    del_leet_count = sum(1 for c in deleeted_txt if c in leet_map)
    has_leet = orig_leet_count > del_leet_count
    
    # Separator detection: check if non-space separators exist between word characters
    has_sep = False
    for sep in intra_word_separators:
        if sep != " ":
            escaped_sep = re.escape(sep)
            if re.search(rf'(\w){escaped_sep}(?=\w)', text):
                has_sep = True
                break
                
    # Spacing detection: single letters separated by space
    has_space = bool(re.search(r'\b\w\s\w\b', text))
    
    # Ratio of non-Vietnamese Latin letters
    letters = [c for c in text if c.isalpha()]
    non_vn_letters = [c for c in letters if c not in VN_LETTERS]
    ratio = len(non_vn_letters) / len(letters) if letters else 0.0
    
    # Obfuscation score calculation
    score = 0.0
    if has_zw:
        score += 0.6
    if has_conf:
        score += 0.6
    if has_leet:
        score += 0.6
    if has_sep:
        score += 0.5
    if has_space:
        score += 0.5
    if ratio > 0.05:
        score += 0.5
        
    score = min(1.0, score)
    
    # Level threshold mapping
    obfuscation_tau = float(os.environ.get("OBFUSCATION_TAU", "0.5"))
    if score == 0.0:
        level = "none"
    elif score >= obfuscation_tau:
        level = "high"
    else:
        level = "low"
        
    return {
        "score": round(score, 4),
        "level": level,
        "flags": {
            "zero_width": has_zw,
            "confusable": has_conf,
            "leet": has_leet,
            "separator": has_sep,
            "spacing": has_space,
            "non_vn_latin_ratio": round(ratio, 4)
        }
    }
