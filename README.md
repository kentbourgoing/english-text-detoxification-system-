# An Exploration of Modern Text Detoxification Pipelines in a Modular Framework

**UC Berkeley W266 Natural Language Processing with Deep Learning - Final Project**

**Authors:** Benjamin He (ben_he@berkeley.edu) and Kent Bourgoing (kentbourgoing@ischool.berkeley.edu)

## Overview

This repository contains our research on **text detoxification** - the task of automatically rewriting toxic or offensive text into safer, non-toxic versions while preserving meaning and fluency. We built a modular framework that systematically compares different detoxification approaches, including traditional seq2seq models and modern LLM-based methods.

### Key Contribution

We developed and evaluated **11 different detoxification pipeline configurations** that combine:
- **2 masking strategies**: DecompX-based (explanation-driven) and LLM-based
- **2 infilling models**: MaRCo (Product-of-Experts) and Mistral-7B-Instruct
- **2 reranking methods**: DecompX-based and Global Reranking (our novel contribution)

## Main Findings

**T5-base + Global Reranking achieves the best overall trade-off**, reducing toxicity to 0.051 (lowest among all systems) while maintaining strong semantic similarity and fluency.

**MaRCo-based infilling frequently reintroduces severe toxic content**, including explicit slurs and threats, highlighting the critical importance of robust reranking.

**Reranking has the largest impact on safety** - our Global Reranking method consistently improves safety across all generator and masker combinations.

## Repository Structure

```
.
├── README.md                                    # This file
├── An Exploration of Modern Text...pdf          # Final research report
├── T5_base_Paradetox_training.ipynb            # T5 model training notebook
└── XDetox/                                      # Main project directory
    ├── README.md                                # Detailed XDetox documentation
    ├── DecompX/                                 # DecompX explainability framework
    ├── datasets/                                # 7 toxicity datasets
    ├── data/                                    # Model outputs and results
    │   ├── dexp_outputs/                        # DecompX experiment outputs
    │   └── model_outputs/                       # Results from all 11 pipelines
    ├── rewrite/                                 # Core detoxification pipeline
    │   ├── masking.py                           # MaRCo masking implementation
    │   ├── generation.py                        # BART-based infilling
    │   ├── infilling.py                         # Text infilling utilities
    │   └── rewrite_example.py                   # End-to-end pipeline example
    ├── evaluation/                              # Evaluation metrics
    │   ├── evaluate_all.py                      # Comprehensive evaluation
    │   ├── toxicity.py                          # Toxicity scoring
    │   ├── perplexity.py                        # Fluency measurement
    │   ├── bleu.py                              # BLEU-4 metric
    │   └── bertscore.py                         # BERTScore metric
    └── [11 Pipeline Notebooks]                  # Experimental configurations
```


## The 11 Pipeline Configurations

### T5-based Baselines (3 variants)
1. **T5-base** - Simple beam search generation
2. **T5-base + DecompX Reranking** - Token-level toxicity scoring
3. **T5-base + Global Reranking** - Multi-objective reranking (best overall)

### XDetox-style Pipelines (8 variants)

**DecompX Masking variants:**
4. DecompX Masking + MaRCo Infilling + DecompX Reranking
5. DecompX Masking + MaRCo Infilling + Global Reranking
6. DecompX Masking + LLM Infilling + DecompX Reranking
7. DecompX Masking + LLM Infilling + Global Reranking

**LLM Masking variants:**
8. LLM Masking + MaRCo Infilling + DecompX Reranking
9. LLM Masking + MaRCo Infilling + Global Reranking
10. LLM Masking + LLM Infilling + DecompX Reranking
11. LLM Masking + LLM Infilling + Global Reranking

## Evaluation

All models are evaluated on 671 ParaDetox test sentences using:

- **Toxicity**: XLM-R toxicity classifier probability
- **BERTScore**: Semantic similarity with reference
- **MeaningBERT**: Meaning preservation score
- **BLEU-4**: N-gram overlap with reference
- **Perplexity**: GPT-2-XL fluency measurement

### Results Summary

| Model | Toxicity ↓ | BERTScore ↑ | MeaningBERT ↑ | BLEU-4 ↑ | Perplexity ↓ |
|-------|-----------|-------------|---------------|----------|--------------|
| **T5-base + Global Reranking** | **0.051** | 0.936 | 67.25 | 53.34 | **171.53** |
| T5-base | 0.203 | **0.953** | **74.84** | 82.65 | 192.07 |
| DecompX + LLM + Global | 0.103 | 0.932 | 64.74 | 81.54 | 162.39 |

*(Full results in Table 1 of the final report)*

## Key Features

- **Models**: T5-base, BART-base, Mistral-7B-Instruct, RoBERTa, XLM-R
- **Frameworks**: PyTorch, HuggingFace Transformers, HuggingFace Datasets
- **Explainability**: DecompX (ACL 2023) for token-level attribution
- **Detoxification**: MaRCo (ACL 2023) Product-of-Experts
- **Evaluation**: BERTScore, MeaningBERT, SacreBLEU
- **Dataset**: **ParaDetox** (670 test examples)

## Acknowledgments

This project builds upon:
- **XDetox**: Lee et al. (2024) - https://github.com/LeeBumSeok/XDetox
- **DecompX**: Modarressi et al. (2023) - https://github.com/mohsenfayyaz/DecompX
- **MaRCo**: Hallinan et al. (2023) - https://github.com/shallinan1/MarcoDetoxification
- **ParaDetox**: Logacheva et al. (2022) - https://github.com/s-nlp/paradetox

## Contact

For questions or collaboration:
- Benjamin He: ben_he@berkeley.edu
- Kent Bourgoing: kentbourgoing@ischool.berkeley.edu

---

**Note**: This repository contains examples of toxic language from datasets and model outputs for research purposes. Content may be offensive or disturbing.
