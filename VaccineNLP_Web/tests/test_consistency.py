# -*- coding: utf-8 -*-
import unittest
import sys
import os

# Add api_service path to import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../api_service')))
from app.main import compute_consistency

class TestConsistencyLogic(unittest.TestCase):
    def test_unusual_stance_against_sentiment_positive(self):
        """H1: Against stance + Positive sentiment -> unusual consistency (triangular mismatch)."""
        res = compute_consistency("Real", "Against", "Positive")
        self.assertEqual(res, "unusual")

    def test_unusual_misinfo_fake_stance_favor_or_neutral(self):
        """H3: Fake misinfo + Favor/Neutral stance -> unusual consistency."""
        res_favor = compute_consistency("Fake", "Favor", "Negative")
        self.assertEqual(res_favor, "unusual")
        
        res_neutral = compute_consistency("Fake", "Neutral", "Neutral")
        self.assertEqual(res_neutral, "unusual")

    def test_high_risk_stance_against_sentiment_negative(self):
        """Profile: Against stance + Negative sentiment -> high_risk (pro-anti vaccine profile)."""
        res = compute_consistency("Real", "Against", "Negative")
        self.assertEqual(res, "high_risk")
        
        res_fake = compute_consistency("Fake", "Against", "Negative")
        self.assertEqual(res_fake, "high_risk")

    def test_plausible_normal_cases(self):
        """Standard matching cases -> plausible."""
        res1 = compute_consistency("Real", "Favor", "Positive")
        self.assertEqual(res1, "plausible")
        
        res2 = compute_consistency("Real", "Neutral", "Neutral")
        self.assertEqual(res2, "plausible")

if __name__ == '__main__':
    unittest.main()
