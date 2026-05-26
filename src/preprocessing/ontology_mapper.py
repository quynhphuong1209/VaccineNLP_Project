import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from src.common.paths import ROOT_DIR

log = logging.getLogger(__name__)

class OntologyMapper:
    def __init__(self, ontology_path: Path = None, mapping_path: Path = None):
        # Đã đồng bộ với cấu trúc VaccineNLP_Clean_V1
        ref_dir = ROOT_DIR / "reference_data" / "ontology_v3"
        
        if ontology_path is None:
            ontology_path = ref_dir / "vaccine_ontology_ai_agent_v3.json"
        
        if mapping_path is None:
            mapping_path = ref_dir / "vaccine_canonical_mapping_v3.csv"
            
        self.ontology_path = ontology_path
        self.mapping_path = mapping_path
        
        self.ontology: dict = {}
        self.mapping: Dict[str, Set[str]] = {}  # term -> set of full canonical groups
        self.exact_matches: Dict[str, str] = {} # term -> topic domain
        
        self._load_data()
        
    def _load_data(self):
        try:
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
        except Exception as e:
            log.error(f"Failed to load ontology from {self.ontology_path}: {e}")
            import traceback; traceback.print_exc()
            self.ontology = {}
            
        try:
            with open(self.mapping_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    group = row.get("canonical_group", "")
                    term = row.get("canonical_term", "").lower().strip()
                    if not term or not group:
                        continue
                    if term not in self.mapping:
                        self.mapping[term] = set()
                    self.mapping[term].add(group)
                    
            # Derive domains from mapping for fast lookup
            for term, groups in self.mapping.items():
                for group in groups:
                    domain = group.split(".")[0]
                    self.exact_matches[term] = domain
                    
        except Exception as e:
            log.error(f"Failed to load mapping from {self.mapping_path}: {e}")
            import traceback; traceback.print_exc()
            self.mapping = {}

    def get_keywords_by_domain(self, target_domain: str) -> List[str]:
        """Extracts flat list of keywords belonging to a specific domain (e.g., 'misinformation_and_antivax')"""
        keywords = set()
        for domain in self.ontology.get("domains", []):
            if domain.get("name") == target_domain:
                for subdomain in domain.get("subdomains", []):
                    keywords.update(subdomain.get("keywords", []))
                    keywords.update(subdomain.get("variants", []))
        return list(keywords)

    def get_hashtags(self) -> List[str]:
        return self.ontology.get("hashtags", [])

    def tag_text(self, text: str) -> Dict[str, object]:
        """
        Takes a raw normalized text and tags it based on the ontology mapping.
        Returns topic_labels, sentiment_labels, misinfo_labels.
        """
        text = text.lower()
        topic_labels = set()
        sentiment_labels = set()
        misinfo_labels = set()
        matched_terms = set()
        
        for term, domain in self.exact_matches.items():
            # Basic boundary check can be improved, but ' in ' is simple for Phase 1
            if f" {term} " in f" {text} " or term in text: # simple substring match
                matched_terms.add(term)
                
                if domain in ["vaccination_general", "covid_infection", "safety_and_side_effects", "immunity_and_health_system"]:
                    topic_labels.add(domain)
                elif domain == "sentiment_and_reaction":
                    # Determine positive/negative from groups
                    for grp in self.mapping.get(term, []):
                        if "positive_sentiment" in grp:
                            sentiment_labels.add("positive")
                        elif "negative_sentiment" in grp:
                            sentiment_labels.add("negative")
                elif domain == "misinformation_and_antivax":
                    topic_labels.add(domain)
                    misinfo_labels.add("likely_misinformation")
                    for grp in self.mapping.get(term, []):
                        if "anti_vax_core" in grp:
                            misinfo_labels.add("explicit_antivax")
                        elif "conspiracy_claims" in grp:
                            misinfo_labels.add("conspiracy")
                elif domain == "spoken_and_misspellings":
                    topic_labels.add("spoken_and_misspellings")

        # Resolve primary sentiment
        primary_sentiment = "neutral"
        if "positive" in sentiment_labels and "negative" in sentiment_labels:
            primary_sentiment = "mixed"
        elif "positive" in sentiment_labels:
            primary_sentiment = "positive"
        elif "negative" in sentiment_labels:
            primary_sentiment = "negative"
            
        # Resolve misinfo
        primary_misinfo = "none"
        if "explicit_antivax" in misinfo_labels or "conspiracy" in misinfo_labels:
            primary_misinfo = "explicit_antivax" if "explicit_antivax" in misinfo_labels else "conspiracy"
        elif "likely_misinformation" in misinfo_labels:
            primary_misinfo = "likely_misinformation"

        return {
            "topics": list(topic_labels),
            "sentiment": primary_sentiment,
            "misinformation": primary_misinfo,
            "matched_terms": list(matched_terms)
        }

import sys
if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    mapper = OntologyMapper()
    print("Hashtags:", mapper.get_hashtags()[:5])
    print("Tagging 'Tôi rất sợ tiêm mũi 4 vì sợ sốc phản vệ':")
    print(mapper.tag_text('Tôi rất sợ tiêm mũi 4 vì sợ sốc phản vệ'))
