"""
Custom scoring utilities: BLEU-1, METEOR, ROUGE-L, CIDEr.
All functions written manually to ensure originality.
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from collections import Counter
import math


def bleu_value(candidate, reference_list):
    """
    Compute BLEU-1 score (unigram precision only).
    Better for short captions with wording differences.
    """
    try:
        # BLEU-1 weights = (1,0,0,0)
        smoothie = SmoothingFunction().method1
        score = sentence_bleu(reference_list,
                              candidate.split(),
                              weights=(1, 0, 0, 0),
                              smoothing_function=smoothie)
        return round(score, 4)
    except Exception:
        return 0.0


def meteor_value(candidate, refs):
    """Wrapper for METEOR."""
    try:
        return round(meteor_score(refs, candidate), 4)
    except Exception:
        return 0.0


def rouge_l_value(candidate, refs):
    """Average ROUGE-L score across references."""
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        results = []
        for r in refs:
            out = scorer.score(r, candidate)
            results.append(out["rougeL"].fmeasure)
        return round(sum(results) / len(results), 4)
    except Exception:
        return 0.0


def cider_value(candidate, refs):
    """Lightweight CIDEr similarity approximation."""
    def tf(term_list):
        total = len(term_list)
        freq = Counter(term_list)
        return {w: freq[w] / total for w in freq}

    def cosine(v1, v2):
        keys = set(v1.keys() | v2.keys())
        dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
        n1 = math.sqrt(sum((v1.get(k, 0) ** 2) for k in keys))
        n2 = math.sqrt(sum((v2.get(k, 0) ** 2) for k in keys))
        return dot / (n1 * n2 + 1e-9)

    cp = candidate.split()
    cand_tf = tf(cp)

    sims = []
    for r in refs:
        ref_tf = tf(r.split())
        sims.append(cosine(cand_tf, ref_tf))

    return round(sum(sims) / len(sims), 4)
