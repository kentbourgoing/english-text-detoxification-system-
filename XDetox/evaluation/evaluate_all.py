from .bertscore import get_bert_scores
from .bleu import get_bleu, calc_bleu
from .perplexity import get_perplexity

import numpy as np
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate as hf_evaluate


# ============================
# MeaningBERT (semantic metric)
# ============================

_MEANING_METRIC = None


def _lazy_load_meaningbert():
    """
    Lazy-load the MeaningBERT metric from HuggingFace Evaluate.
    """
    global _MEANING_METRIC
    if _MEANING_METRIC is None:
        _MEANING_METRIC = hf_evaluate.load("davebulaval/meaningbert")


def get_meaningbert_score(references, hypotheses):
    """
    Compute a single scalar MeaningBERT score over all (ref, hyp) pairs.

    The 'davebulaval/meaningbert' evaluate metric typically returns:
        {
          "mean_meaning_preservation": float,
          "scores": [float, float, ...]
        }

    We prefer the mean_meaning_preservation key if present, otherwise we
    average over 'scores'.
    """
    _lazy_load_meaningbert()

    res = _MEANING_METRIC.compute(
        references=references,
        predictions=hypotheses,
    )

    # Most recent implementations expose this:
    if "mean_meaning_preservation" in res:
        return float(res["mean_meaning_preservation"])

    # Fallback: average over raw scores if only scores[] are exposed
    if "scores" in res:
        scores = np.array(res["scores"], dtype=float)
        return float(scores.mean())

    # Extra fallback if future versions change keys again
    raise KeyError(f"Unexpected MeaningBERT output keys: {list(res.keys())}")


# ======================================
# Local toxicity scorer (XLM-R, HF hub)
# ======================================

_MODEL_NAME = "textdetox/xlmr-large-toxicity-classifier-v2"
_TOX_TOKENIZER = None
_TOX_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _lazy_load_tox():
    """
    Lazy-load the local toxicity classifier.
    """
    global _TOX_TOKENIZER, _TOX_MODEL
    if _TOX_TOKENIZER is None or _TOX_MODEL is None:
        _TOX_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _TOX_MODEL = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME
        ).to(_DEVICE)
        _TOX_MODEL.eval()


@torch.no_grad()
def get_toxicity_scores(texts, batch_size=32):
    """
    Returns a list of toxicity probabilities in [0,1] for each input text.
    Index 1 of the softmax is the 'toxic' probability.
    """
    _lazy_load_tox()
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = _TOX_TOKENIZER(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(_DEVICE)

        logits = _TOX_MODEL(**enc).logits
        probs = torch.softmax(logits, dim=-1)  # [B, 2]
        scores.extend(probs[:, 1].detach().cpu().tolist())

    return scores


# ============================
# Main evaluation entry point
# ============================

def evaluate_all(
    references,
    hypotheses,
    eval_ref=True,
    use_corpus_bleu=True,
    tox_threshold=0.5,
    tox_batch_size=32,
):
    """
    Compute all metrics used in XDetox:
      - BERTScore (F1)
      - MeaningBERT
      - BLEU-4 (corpus or sentence-level)
      - Perplexity (gen + orig)
      - Toxicity (gen + orig, mean + percent >= threshold)
    """
    # --------- BERTScore (F1) ---------
    bs = np.nanmean(get_bert_scores(zip(hypotheses, references))["f1"])

    # --------- MeaningBERT (semantic preservation) ---------
    mb = get_meaningbert_score(references, hypotheses)

    # --------- BLEU-4 ---------
    if use_corpus_bleu:
        bleu = get_bleu([[r] for r in references], hypotheses)
    else:
        bleu = calc_bleu(references, hypotheses)

    # --------- Perplexity ---------
    perp_hyp = np.nanmean(get_perplexity(hypotheses))
    perp_ref = np.nanmean(get_perplexity(references)) if eval_ref else None

    # --------- Toxicity (local HF model) ---------
    tox_hyp = get_toxicity_scores(hypotheses, batch_size=tox_batch_size)
    tox_hyp_mean = float(np.nanmean(tox_hyp)) if len(tox_hyp) else float("nan")
    tox_hyp_pct = (
        float((np.array(tox_hyp) >= tox_threshold).mean())
        if len(tox_hyp)
        else float("nan")
    )

    tox_ref_mean, tox_ref_pct = None, None
    if eval_ref:
        tox_ref = get_toxicity_scores(references, batch_size=tox_batch_size)
        tox_ref_mean = float(np.nanmean(tox_ref)) if len(tox_ref) else float("nan")
        tox_ref_pct = (
            float((np.array(tox_ref) >= tox_threshold).mean())
            if len(tox_ref)
            else float("nan")
        )

    return (
        bs,            # 0  - BERTScore F1
        mb,            # 1  - MeaningBERT
        bleu,          # 2  - BLEU-4
        tox_hyp_mean,  # 3  - mean toxicity (gen)
        perp_hyp,      # 4  - perplexity (gen)
        tox_ref_mean,  # 5  - mean toxicity (orig)
        perp_ref,      # 6  - perplexity (orig)
        tox_hyp_pct,   # 7  - % toxic (gen)
        tox_ref_pct,   # 8  - % toxic (orig)
    )


# ===================
# CLI glue / I/O
# ===================

def get_data(args):
    with open(args.orig_path, "r") as f:
        orig = [s.strip() for s in f.readlines()]
    with open(args.gen_path, "r") as f:
        gen = [s.strip() for s in f.readlines()]

    if len(orig) != len(gen):
        raise ValueError(f"Line count mismatch: {len(orig)} refs vs {len(gen)} hyps")

    return orig, gen


def eval_args(args):
    orig, gen = get_data(args)

    metrics = evaluate_all(
        orig,
        gen,
        eval_ref=not args.skip_ref,
        use_corpus_bleu=not args.use_sentence_bleu,
        tox_threshold=args.tox_threshold,
        tox_batch_size=args.tox_batch_size,
    )

    # Match naming used elsewhere (gen_stats.txt)
    save_path = args.gen_path[:-4] + "_stats.txt"  # gen.txt -> gen_stats.txt

    items = [
        "bertscore",          # 0
        "meaningbert",        # 1
        "bleu4",              # 2
        "toxicity gen",       # 3
        "perplexity gen",     # 4
        "toxicity orig",      # 5
        "perplexity orig",    # 6
        "percent toxic gen",  # 7
        "percent toxic ref",  # 8
    ]

    with open(save_path, "w") as f:
        for name, val in zip(items, metrics):
            if isinstance(val, (float, np.floating)):
                sval = f"{val:.5g}"   # 5 significant figures
            else:
                sval = str(val)
            print(name, ":", sval)
            f.write(name + ": " + sval + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path", required=True)
    parser.add_argument("--gen_path", required=True)
    parser.add_argument(
        "--skip_ref",
        action="store_true",
        help="Skip computing metrics on references",
    )
    parser.add_argument(
        "--use_sentence_bleu",
        action="store_true",
        help="Use sentence BLEU instead of corpus BLEU",
    )
    parser.add_argument(
        "--tox_threshold",
        type=float,
        default=0.5,
        help="Threshold for 'percent toxic' (default 0.5)",
    )
    parser.add_argument(
        "--tox_batch_size",
        type=int,
        default=32,
        help="Batch size for local toxicity scoring",
    )
    eval_args(parser.parse_args())
