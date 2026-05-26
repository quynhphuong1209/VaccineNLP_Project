"""
VaccineNLP App Core Module
==========================
Shared model loading, prediction, XAI reasoning, and data fetching utilities.

This module contains the core business logic extracted from both Streamlit and Gradio apps,
enabling them to share a single source of truth for:
- Model loading and inference
- XAI reasoning and explanation generation  
- Data fetching from various sources
- Text utilities and translations

Usage:
    from src.app_core import predictor, xai_engine, fetchers
    
    # Load model
    model, tokenizer, ok = predictor.load_model("PhoBERT-v2")
    
    # Make prediction
    result = predictor.predict_cached(text, "PhoBERT-v2")
    
    # Get XAI explanation
    reasoning = xai_engine.find_xai_reasoning(text, xai_cache)
    
    # Fetch from URL
    texts, source = fetchers.fetch_url_as_list(url)
"""

__version__ = "1.0.0"
__author__ = "VaccineNLP Team"

from . import predictor
from . import xai_engine
from . import fetchers

__all__ = [
    "predictor",
    "xai_engine",
    "fetchers",
]
