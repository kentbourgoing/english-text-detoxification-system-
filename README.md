# An Exploration of Modern Text Detoxification Pipelines in a Modular Framework
 
An NLP system that automatically rewrites toxic online content into safer alternatives while preserving meaning and fluency. By testing 11 pipeline configurations combining explainability-driven masking (DecompX), LLM-based generation (Mistral-7B), and a novel multi-objective reranking algorithm, we achieved 75% toxicity reduction (0.208→0.051) with 93.6% semantic similarity.
 
---
## Table of Contents

- [Problem and Goal](#problem-and-goal)
- [Approach](#approach)
- [Results](#results)
  - [Technical Deliverables](#technical-deliverables)
  - [Key Outcomes](#key-outcomes)
- [Tech/Methods](#techmethods)
- [Repo Structure](#repo-structure)
- [Prerequisites](#prerequisites)
  - [Required Environment Variables](#required-environment-variables)
- [How to Run](#how-to-run)
  - [Quick Start (Recommended for Evaluation)](#quick-start-recommended-for-evaluation)
  - [Full Pipeline (Training and Evaluation)](#full-pipeline-training-and-evaluation)
    - [Phase 1: Train T5-Base Model](#phase-1-train-t5-base-model)
    - [Phase 2: Run T5-Based Pipelines](#phase-2-run-t5-based-pipelines)
    - [Phase 3: Run XDetox-Style Pipelines (DecompX Masking)](#phase-3-run-xdetox-style-pipelines-decompx-masking)
    - [Phase 4: Run XDetox-Style Pipelines (LLM Masking)](#phase-4-run-xdetox-style-pipelines-llm-masking)
    - [Phase 5: Aggregate Results and Analysis](#phase-5-aggregate-results-and-analysis)
- [Configuration](#configuration)
  - [Global Reranking Weights (Tunable)](#global-reranking-weights-tunable)
  - [DecompX Masking Threshold](#decompx-masking-threshold)
  - [Inference Speed vs. Quality](#inference-speed-vs-quality)
- [Notes: Limitations and Next Steps](#notes-limitations-and-next-steps)
  - [Current Limitations](#current-limitations)
  - [Next Steps](#next-steps)
- [Credits / Data / Licenses](#credits--data--licenses)
  - [Data Sources](#data-sources)
  - [Frameworks and Pre-trained Models](#frameworks-and-pre-trained-models)
  - [Academic Partner](#academic-partner)
- [Team Members](#team-members)

---
 
## Problem and Goal
 
- **Problem:** Toxic language proliferates across social media, forums, and user-generated content platforms, harming users and communities. Simple content filtering removes context and feels like censorship, while allowing toxic content unchecked creates hostile environments. Existing detoxification systems (MaRCo, XDetox) lack systematic component-level analysis—practitioners don't know whether masking strategy, generation model, or reranking method contributes most to safety, making it difficult to optimize pipelines for production deployment where compute budgets and accuracy requirements vary.
 
- **Why It Matters:** Content moderation is a $5B+ industry with platforms facing regulatory pressure and user safety demands. Effective detoxification enables proactive moderation tools for creators, automated editing suggestions in messaging apps, and safer LLM outputs. Understanding which pipeline components drive performance allows engineering teams to make informed trade-offs between accuracy, cost, latency, and interpretability when deploying text safety systems at scale.
 
- **Goal:** Build a modular detoxification framework that decouples masking, infilling, and reranking to enable systematic comparison of 11 pipeline configurations. Develop a novel Global Reranking method that jointly optimizes toxicity, semantic similarity, and fluency. Evaluate all systems on 671 ParaDetox test sentences using five automatic metrics (toxicity, BERTScore, MeaningBERT, BLEU-4, perplexity) plus qualitative failure analysis. Deliver actionable insights on which architectural choices maximize safety while preserving meaning,
 
---
 
## Approach
 
1. **Modular Pipeline Architecture:** Designed three-stage framework with swappable components: (1) **Masker** identifies toxic tokens and replaces with `<mask>`, (2) **Infiller** generates C=10 candidate rewrites per masked input, (3) **Reranker** scores candidates and selects final output. Built 11 end-to-end pipelines by combining 2 masking strategies × 2 infilling models × 2 reranking methods, plus 3 T5-only baselines for controlled comparison.
 
2. **DecompX Token Attribution Masking:** Implemented explanation-driven masking using DecompX (Modarressi et al., 2023) token decomposition on RoBERTa toxicity classifier. For each input token, computed toxic importance score as sum of class contributions across classifier layers, then masked tokens exceeding threshold=0.20. This grounds masking in model internals rather than heuristics, providing interpretable toxic span identification.
 
3. **LLM-Based Masking:** Developed prompt-engineered masking with Mistral-7B-Instruct using strict formatting rules to output masked sentences in `[<mask>]` format. Few-shot prompts instruct the LLM to identify toxic spans, replace with single `<mask>` tokens (collapsing adjacent masks), and preserve non-toxic words/punctuation unchanged. Enables context-aware masking that adapts to subtle toxicity without fixed thresholds.
 
4. **Dual Infilling Strategies:** Integrated two infilling models—(a) **MaRCo** (Hallinan et al., 2023): Product-of-Experts BART combining base, non-toxic expert, and toxic anti-expert models with α₁=4.75, α₂=1.5, temperature=2.5 to sample from detoxified distribution; (b) **Mistral-7B-Instruct**: Prompt-based infilling with temperature=0.7, top-p=0.95, generating 10 candidates per masked sentence while preserving non-masked context.
 
5. **Global Reranking Algorithm (Novel Contribution):** Engineered multi-objective scoring function combining three weighted signals: (a) **Toxicity**: 1 - XLM-R toxic probability (weight=0.5), (b) **Semantic Similarity**: LaBSE cosine similarity with reference (weight=0.3), (c) **Fluency**: Normalized GPT-2 perplexity score clipped between [5, 300] (weight=0.2). Selected candidate maximizing `Score = 0.5(1-p_tox) + 0.3S_sim + 0.2S_flu` to balance safety, meaning preservation, and naturalness.
 
6. **T5-Base Seq2Seq Baselines:** Fine-tuned T5-base on 7,500 ParaDetox parallel training examples using detoxify-prefix prompting. Implemented three variants: (a) greedy beam search (beam=5) single output, (b) stochastic sampling (top-k=50, top-p=0.95) with DecompX reranking, (c) stochastic sampling with Global Reranking—establishing performance ceiling for supervised approaches without explicit masking.
 
7. **Comprehensive Evaluation Pipeline:** Built automated evaluation system computing five metrics across 671 test sentences: toxicity (XLM-R classifier probability), semantic similarity (BERTScore + MeaningBERT), surface overlap (BLEU-4), fluency (GPT-2 perplexity). Supplemented with manual qualitative analysis categorizing failure modes (slurs, profanity, stance reversals, meaning drift) to identify which pipelines introduce severe toxic content vs. mild impoliteness.
 
8. **Controlled Experimental Design:** Held reranker fixed to compare masking strategies (DecompX vs. LLM), held masker fixed to compare infillers (MaRCo vs. LLM), and varied rerankers (DecompX vs. Global) across all masker-infiller combinations. Used identical hyperparameters (C=10 candidates, same prompts, same classifiers) across matched conditions to isolate component effects and ensure fair comparison.
 
---
 
## Results
 
### Technical Deliverables
 
- **11 Production-Ready Detoxification Pipelines:** Implemented and evaluated complete end-to-end systems including 3 T5-base variants (no reranking, DecompX reranking, Global reranking), 4 DecompX-masked variants (MaRCo/LLM infilling × DecompX/Global reranking), and 4 LLM-masked variants—all containerizable as independent services with standardized input/output formats.
 
- **Global Reranking Algorithm:** Novel multi-objective scoring method achieving 35-75% toxicity reduction over DecompX reranking baseline across all 11 configurations. Best configuration (T5-base + Global Reranking) reached **0.051 toxicity** (75% reduction from 0.208 baseline) while maintaining **93.6% BERTScore** and **67.25 MeaningBERT** semantic similarity—demonstrating reranking has 3-5× larger impact on safety than masking or infilling choices.
 
- **Comprehensive Benchmark Dataset:** Evaluated all systems on 671-sentence ParaDetox test set using 5 automatic metrics (3,355 total metric computations) plus manual qualitative analysis of 100+ failure cases categorized by severity (explicit slurs, profanity, dehumanization, stance reversal). Identified that MaRCo infilling introduces severe toxic content (e.g., "whiny cunts", "I will cut you") in 8-12% of outputs even with reranking, while LLM infilling produces milder failures.
 
- **Modular Framework Codebase:** Built reusable Python pipeline framework with 11 Jupyter notebooks (one per configuration), shared evaluation utilities (`evaluate_all.py`, toxicity/perplexity/BLEU/BERTScore modules), and pluggable components for masking/infilling/reranking. Enables rapid prototyping of new configurations—adding a new masker requires implementing single `mask(text) → masked_text` interface.
 
- **Optimal Architecture Identification:** Through systematic ablation, determined **DecompX masking + Mistral-7B infilling + Global Reranking** as best XDetox-style architecture (toxicity=0.103, BERTScore=0.932), but **T5-base + Global Reranking** provides superior overall trade-off (toxicity=0.051, BERTScore=0.936, perplexity=171.53) while being 4× faster at inference due to single-model architecture vs. ensemble pipeline.
 
### Key Outcomes
 
- **Reranking Dominance:** Global Reranking consistently outperformed DecompX reranking across all 11 configurations, reducing toxicity by 0.012-0.156 (9-75% relative reduction) while improving or maintaining fluency. For T5-base, Global Reranking dropped toxicity from 0.208→0.051 and perplexity from 235→171, demonstrating that multi-objective selection is more critical than generation quality for production safety.
 
- **Masking Strategy Impact:** DecompX masking achieved 13-33% lower toxicity than LLM masking when paired with identical infillers/rerankers (e.g., 0.132 vs 0.200 with MaRCo+DecompX reranking). DecompX's token attribution over-masks context around toxic spans, hiding subtle toxicity cues but increasing perplexity; LLM masking is more selective but occasionally misses adjectives or group references that amplify toxicity during infilling.
 
- **Infilling Model Safety:** LLM infilling (Mistral-7B) outperformed MaRCo BART in 8 of 11 configurations on toxicity metrics, reducing severe failures (explicit slurs, violent threats) by replacing insults with safety-shaped templates ("disrespectful person", "hurtful language"). However, LLM infilling still produced mild profanity ("as hell", "holy shit") and dehumanizing phrases ("piece of human waste"), indicating prompt engineering alone is insufficient for guaranteed safety.
 
- **Production Deployment Insights:** T5-base + Global Reranking provides best safety/performance trade-off for real-world deployment: 75% toxicity reduction, sub-second inference on GPU (vs. 3-5 seconds for MaRCo ensemble + LLM masking), and deterministic behavior suitable for compliance auditing. XDetox-style pipelines offer greater interpretability (explanation-driven masking) but require 3× compute budget and introduce inconsistent failures from multi-model orchestration.
 
---
 
## Tech/Methods
 
**Languages & Frameworks:** Python 3.10, PyTorch 2.0, HuggingFace Transformers 4.41.2, LangChain (LLM orchestration), Jupyter Notebooks
 
**LLMs & Models:** Mistral-7B-Instruct-v0.2 (masking/infilling), T5-base (seq2seq), MaRCo BART (Product-of-Experts), XLM-R (toxicity classifier), RoBERTa-base (DecompX explanations), LaBSE (semantic similarity embeddings), GPT-2-XL (perplexity scoring)
 
**Infrastructure & Data:** Google Colab (A100 GPU runtime), HuggingFace Model Hub, ParaDetox dataset (7,500 train / 671 test parallel examples), PyTorch DataLoaders, Amazon S3 (model checkpoint storage)
 
**Methods:** DecompX token decomposition (Modarressi et al., 2023), MaRCo Product-of-Experts detoxification (Hallinan et al., 2023), XDetox mask-and-infill pipeline (Lee et al., 2024), Multi-objective reranking (novel), Prompt engineering (few-shot masking/infilling), BERTScore semantic similarity, Nucleus sampling (top-p), Beam search decoding, Token-level attribution, Ensemble voting
 
---
 
## Repo Structure
 
```
datasci266-project/
├── Modular Framework/               # Core pipeline implementation (11 configurations)
│   ├── T5_base_Paradetox_training.ipynb        # Fine-tune T5 on ParaDetox training data
│   ├── T5_ParaDetox_Pipeline.ipynb              # T5-base (beam search, no reranking)
│   ├── T5_ParaDetox_w_DecompX-Reranking_Pipeline.ipynb  # T5 + DecompX reranking
│   ├── T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb   # T5 + Global Reranking (BEST)
│   │
│   ├── XDetox_w_DecompX-Masking-DecompX-Reranking_Pipeline.ipynb       # DecompX mask + MaRCo + DecompX rerank
│   ├── XDetox_w_DecompX-Masking-Global-Reranking_Pipeline.ipynb        # DecompX mask + MaRCo + Global rerank
│   ├── XDetox_w_DecompX-Masking_LLM-Infilling_DecompX-Reranking_Pipeline.ipynb  # DecompX mask + LLM + DecompX rerank
│   ├── XDetox_w_DecompX-Masking_LLM-Infilling_Global-Reranking_Pipeline.ipynb   # DecompX mask + LLM + Global rerank
│   │
│   ├── XDetox_w_LLM-Masking_DecompX-Reranking_Pipeline.ipynb           # LLM mask + MaRCo + DecompX rerank
│   ├── XDetox_w_LLM-Masking_Global-Reranking_Pipeline.ipynb            # LLM mask + MaRCo + Global rerank
│   ├── XDetox_w_LLM-Masking_LLM-Infilling_DecompX-Reranking_Pipeline.ipynb  # LLM mask + LLM + DecompX rerank
│   ├── XDetox_w_LLM-Masking_LLM-Infilling_Global-Reranking_Pipeline.ipynb   # LLM mask + LLM + Global rerank
│   │
│   ├── DecompX/                     # DecompX explainability framework (Modarressi et al., 2023)
│   │   ├── src/                     # Token decomposition implementation
│   │   │   ├── decompx_utils.py     # Core DecompX utilities
│   │   │   ├── modeling_bert.py     # BERT with DecompX support
│   │   │   └── modeling_roberta.py  # RoBERTa with DecompX support
│   │   ├── experiments/             # DecompX experiments and notebooks
│   │   └── README.md                # DecompX documentation
│   │
│   └── data/                        # Model outputs and evaluation results
│       └── model_outputs/           # Generated text from all 11 pipelines
│           ├── T5_baseline/
│           ├── T5_w_DecompX-Reranking/
│           ├── T5_w_Global-Reranking/
│           ├── XDetox_w_DecompX-Masking-DecompX-Reranking/
│           ├── XDetox_w_DecompX-Masking-Global-Reranking/
│           ├── XDetox_w_DecompX-Masking_LLM-Infilling_DecompX-Reranking/
│           ├── XDetox_w_DecompX-Masking_LLM-Infilling_Global-Reranking/
│           ├── XDetox_w_LLM-Masking_DecompX-Reranking/
│           ├── XDetox_w_LLM-Masking_Global-Reranking/
│           ├── XDetox_w_LLM-Masking_LLM-Infilling_DecompX-Reranking/
│           └── XDetox_w_LLM-Masking_LLM-Infilling_Global-Reranking/
│
├── Report/                          # Final research paper and documentation
│   └── An Exploration of Modern Text Detoxification Pipelines in a Modular Framework.pdf
│
├── Slides/                          # Project presentation
│   └── text_detoxification_presentation.pdf
│
└── README.md                        # Project documentation (this file)
```
 
**Note:** The `Modular Framework/` directory contains subdirectories for datasets, evaluation scripts, and rewrite utilities that are imported by the pipeline notebooks but not shown above for brevity. Each pipeline notebook is self-contained and can be run independently.
 
---
 
## Prerequisites
 
**Platforms & Services:**
 
- **[Google Colab](https://colab.research.google.com/)** - GPU runtime (A100 recommended) for model training and inference
- **[HuggingFace Account](https://huggingface.co/)** - Access to model repositories and datasets
- **[Google Drive](https://drive.google.com/)** - Storage for model checkpoints and datasets (mounted in Colab)
 
**Required Python Packages:**
 
Install via `pip` (automatically handled in Colab notebooks):
 
```bash
pip install transformers==4.41.2 tokenizers==0.19.1 datasets==2.19.0 \
    evaluate==0.4.1 sacrebleu==2.4.1 sacremoses ftfy nltk \
    bert-score sentencepiece torch pandas numpy tqdm
```
 
**Pre-trained Models (auto-downloaded from HuggingFace):**
 
- `textdetox/xlmr-large-toxicity-classifier-v2` - XLM-R toxicity classifier
- `sentence-transformers/LaBSE` - Semantic similarity embeddings
- `mistralai/Mistral-7B-Instruct-v0.2` - LLM masking/infilling
- `gpt2-xl` - Perplexity scoring
- `google/t5-base` - Seq2seq baseline (requires fine-tuning)
 
**Dataset:**
 
- **ParaDetox** dataset (Logacheva et al., 2022) - Automatically loaded via HuggingFace `datasets` library or from included `Modular Framework/datasets/` directory
 
### Required Environment Variables
 
For local runs outside Colab, create a `.env` file:
 
```env
# HuggingFace Configuration
HF_TOKEN=your_huggingface_token  # Optional, for private models
TRANSFORMERS_CACHE=/path/to/cache  # Model cache directory
 
# Google Drive (Colab only)
PROJECT_BASE=/content/drive/MyDrive/w266-Project
XDETOX_DIR=/content/drive/MyDrive/w266-Project/XDetox
T5_CHECKPOINT=/content/drive/MyDrive/w266-Project/t5-base-detox-model
 
# Weights & Biases (optional)
WANDB_DISABLED=true  # Disable experiment tracking
```
 
**Note:** All notebooks default to Colab-friendly paths (`/content/drive/MyDrive/...`) and handle environment setup automatically. No manual configuration needed for Colab execution.
 
---
 
## How to Run
 
### Quick Start (Recommended for Evaluation)
 
If you want to quickly evaluate the **best-performing model** (T5-base + Global Reranking):
 
1. **Mount Google Drive in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
 
2. **Open the best model notebook:**
   - Navigate to `Modular Framework/T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb`
   - Upload to Google Colab
 
3. **Configure paths** (update cell 2):
   ```python
   PROJECT_BASE = "/content/drive/MyDrive/w266-Project"
   T5_CHECKPOINT = "/content/drive/MyDrive/w266-Project/t5-base-detox-model"
   ```
 
4. **Run all cells:**
   - The notebook will auto-install dependencies, load models, generate detoxified outputs, and compute evaluation metrics
   - Results saved to `data/model_outputs/T5_w_Global-Reranking/`
 
---
 
### Full Pipeline (Training and Evaluation)
 
To reproduce all 11 pipelines from scratch:
 
#### **Phase 1: Train T5-Base Model**
 
**Notebook:** `Modular Framework/T5_base_Paradetox_training.ipynb`
 
This step fine-tunes T5-base on the ParaDetox training dataset (7,500 parallel toxic/non-toxic examples).
 
**Steps:**
1. Upload notebook to Google Colab (requires GPU runtime)
2. Run all cells to:
   - Load ParaDetox training data
   - Fine-tune `google/t5-base` with detoxify-prefix prompting
   - Train for 3 epochs with batch size 16, learning rate 3e-4
   - Save checkpoint to `T5_CHECKPOINT` path (e.g., `/content/drive/MyDrive/w266-Project/t5-base-detox-model`)
3. **Training time:** ~2-3 hours on A100 GPU
4. **Output:** Fine-tuned T5 model checkpoint (required for Phase 2)
 
---
 
#### **Phase 2: Run T5-Based Pipelines**
 
**Notebooks:**
- `T5_ParaDetox_Pipeline.ipynb` - T5-base (beam search, no reranking)
- `T5_ParaDetox_w_DecompX-Reranking_Pipeline.ipynb` - T5 + DecompX reranking
- `T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb` - T5 + Global Reranking (**BEST**)
 
**Steps:**
1. Ensure T5 checkpoint exists from Phase 1
2. Open desired notebook in Colab
3. Update `T5_CHECKPOINT` path in cell 2
4. Run all cells to:
   - Load fine-tuned T5 model
   - Generate 10 candidates per test sentence (671 sentences)
   - Apply reranking strategy (if applicable)
   - Compute 5 evaluation metrics (toxicity, BERTScore, MeaningBERT, BLEU-4, perplexity)
   - Save outputs to `data/model_outputs/[model_name]/`
5. **Inference time:** ~15-30 minutes per notebook on A100 GPU
 
**Key Hyperparameters:**
- `num_examples=671` - Full test set (reduce for faster testing)
- `batch_size=8` - Increase to 16 on A100 for faster inference
- `num_beams=5` (no reranking) or `C=10` candidates (with reranking)
- `max_length=128` tokens
 
---
 
#### **Phase 3: Run XDetox-Style Pipelines (DecompX Masking)**
 
**Notebooks:**
- `XDetox_w_DecompX-Masking-DecompX-Reranking_Pipeline.ipynb` - DecompX mask + MaRCo + DecompX rerank
- `XDetox_w_DecompX-Masking-Global-Reranking_Pipeline.ipynb` - DecompX mask + MaRCo + Global rerank
- `XDetox_w_DecompX-Masking_LLM-Infilling_DecompX-Reranking_Pipeline.ipynb` - DecompX mask + LLM + DecompX rerank
- `XDetox_w_DecompX-Masking_LLM-Infilling_Global-Reranking_Pipeline.ipynb` - DecompX mask + LLM + Global rerank
 
**Steps:**
1. Open desired notebook in Colab (GPU required)
2. Run all cells to:
   - Load RoBERTa toxicity classifier with DecompX
   - Compute token-level toxic importance scores
   - Mask tokens exceeding `threshold=0.20`
   - Infill with MaRCo (Product-of-Experts BART) or Mistral-7B-Instruct
   - Generate C=10 candidates per masked sentence
   - Rerank using DecompX scores or Global Reranking
   - Evaluate and save outputs
3. **Inference time:** ~45-90 minutes per notebook (slower due to multi-model ensemble)
 
**Key Hyperparameters:**
- **Masking:** `decompx_threshold=0.20` (lower = more masking)
- **MaRCo Infilling:** `α₁=4.75`, `α₂=1.5`, `temperature=2.5`, `top_k=50`
- **LLM Infilling:** `temperature=0.7`, `top_p=0.95`, `max_new_tokens=64`
 
---
 
#### **Phase 4: Run XDetox-Style Pipelines (LLM Masking)**
 
**Notebooks:**
- `XDetox_w_LLM-Masking_DecompX-Reranking_Pipeline.ipynb` - LLM mask + MaRCo + DecompX rerank
- `XDetox_w_LLM-Masking_Global-Reranking_Pipeline.ipynb` - LLM mask + MaRCo + Global rerank
- `XDetox_w_LLM-Masking_LLM-Infilling_DecompX-Reranking_Pipeline.ipynb` - LLM mask + LLM + DecompX rerank
- `XDetox_w_LLM-Masking_LLM-Infilling_Global-Reranking_Pipeline.ipynb` - LLM mask + LLM + Global rerank
 
**Steps:**
1. Open desired notebook in Colab (GPU required)
2. Run all cells to:
   - Prompt Mistral-7B-Instruct to identify toxic spans
   - Generate masked sentences with `<mask>` tokens
   - Infill with MaRCo or Mistral-7B-Instruct
   - Generate C=10 candidates per masked sentence
   - Rerank using DecompX scores or Global Reranking
   - Evaluate and save outputs
3. **Inference time:** ~60-120 minutes per notebook (LLM masking adds overhead)
 
**LLM Masking Prompt (excerpt):**
```
You are a helpful assistant trained to make toxic sentences safer by MASKING toxic words
with <mask>. Identify toxic spans, replace with single <mask> token, preserve all other
words/punctuation. Output format: [Masked sentence here.]
```
 
---
 
#### **Phase 5: Aggregate Results and Analysis**
 
After running all 11 notebooks:
 
1. **View individual results:**
   - Each pipeline saves outputs to `data/model_outputs/[pipeline_name]/[dataset]/`
   - Files: `orig.txt` (original toxic), `gen.txt` (detoxified), `gen_stats.txt` (metrics)
 
2. **Compare across pipelines:**
   - Open any notebook's evaluation cells
   - Results table shows all 5 metrics for all 11 models
   - **Key finding:** T5-base + Global Reranking has lowest toxicity (0.051) with high BERTScore (0.936)
 
3. **Manual qualitative analysis:**
   - Review `gen.txt` files for failure cases (slurs, profanity, stance reversals)
   - MaRCo pipelines show most severe toxic failures despite reranking
   - Global Reranking consistently safer than DecompX reranking
 
---
 
## Configuration
 
### Global Reranking Weights (Tunable)
 
The Global Reranking algorithm combines three weighted objectives. Default weights are optimized for ParaDetox but can be adjusted for different use cases:
 
```python
# In any pipeline notebook with Global Reranking, modify:
wT = 0.5  # Toxicity weight (higher = prioritize safety)
wS = 0.3  # Semantic similarity weight (higher = preserve meaning)
wF = 0.2  # Fluency weight (higher = prioritize naturalness)
 
# Score = wT * (1 - toxicity_prob) + wS * cosine_similarity + wF * normalized_fluency
```
 
**Recommended adjustments:**
- **High-stakes moderation (legal, children's content):** `wT=0.7, wS=0.2, wF=0.1` - Maximize safety
- **Creative writing assistance:** `wT=0.3, wS=0.4, wF=0.3` - Balance meaning + fluency
- **Translation/preservation:** `wT=0.4, wS=0.5, wF=0.1` - Prioritize semantic fidelity
 
### DecompX Masking Threshold
 
Control masking aggressiveness (lower threshold = more tokens masked):
 
```python
# In DecompX masking notebooks:
decompx_threshold = 0.20  # Default (balanced)
# decompx_threshold = 0.15  # Aggressive (over-mask, safer but less fluent)
# decompx_threshold = 0.30  # Conservative (under-mask, preserves more but riskier)
```
 
### Inference Speed vs. Quality
 
Adjust number of candidates for reranking (trade latency for diversity):
 
```python
# In notebooks with reranking:
num_candidates = 10  # Default (best quality)
# num_candidates = 5   # 2x faster, slightly lower quality
# num_candidates = 20  # 2x slower, marginal quality gains
```
 
---
 
## Notes: Limitations and Next Steps
 
### Current Limitations
 
- **Dataset Scope:** Evaluated only on English ParaDetox benchmark (671 sentences) with single reference style; results may not generalize to other domains (news, medical, legal), languages (non-English), or demographic-specific toxicity (AAVE, LGBTQ+ slang). Toxicity is culturally contextual and our models lack multilingual/multi-domain validation.
 
- **Toxicity Classifier Bias:** XLM-R toxicity classifier used for both reranking and evaluation creates circularity—systems optimized to fool the scorer may not reduce real-world harm. Classifier exhibits known biases (false positives on AAVE, identity mentions) and misses subtle toxicity (sarcasm, coded language, contextual slurs).
 
- **MaRCo Failure Modes:** MaRCo Product-of-Experts infilling introduced severe toxic content (explicit slurs, violent threats) in 8-12% of outputs even with Global Reranking, demonstrating that gradient-based detoxification alone is insufficient. MaRCo's BART base model lacks safety alignment present in instruction-tuned LLMs.
 
- **Reranking Not Learned:** Global Reranking weights (wT=0.5, wS=0.3, wF=0.2) were manually tuned rather than optimized via reinforcement learning or preference learning. Fixed weights may be suboptimal for different toxicity severities, domains, or user preferences. No gradient signal connects reranker to generator.
 
- **Compute Constraints:** Limited to Mistral-7B-Instruct (mid-tier LLM) due to Colab GPU memory; could not test larger models (Llama 3.1-70B, Claude 3.5 Sonnet, GPT-4) that likely outperform on both safety and meaning preservation. Ensemble pipelines (DecompX masking + MaRCo + reranking) require 3-5 seconds per sentence, too slow for real-time moderation.
 
- **Evaluation Limitations:** Automatic metrics (BLEU, BERTScore) correlate imperfectly with human judgments of detoxification quality. Manual analysis covered <100 examples; lacks systematic human evaluation (inter-rater reliability, user studies on acceptability). No evaluation of unintended harms (e.g., removing identity mentions, changing speaker perspective, introducing new biases).
 
### Next Steps
 
- **Multi-Domain Evaluation:** Expand benchmark to include news articles (political toxicity), Reddit/Twitter threads (conversational context), medical forums (sensitive health topics), and legal documents (precise language requirements). Test on hate speech datasets (HateXplain, SBIC) to measure performance on severe toxicity beyond mild profanity.
 
- **Human Evaluation Study:** Conduct A/B testing with 50+ human raters comparing model outputs on safety, meaning preservation, fluency, and acceptability. Measure inter-rater agreement (Krippendorff's α), collect qualitative feedback on failure modes, and correlate with automatic metrics to identify gaps in evaluation design.
 
- **Learned Reranking:** Replace hand-tuned Global Reranking weights with learned reranker trained via Direct Preference Optimization (DPO) or Reinforcement Learning from Human Feedback (RLHF) on human preference data. Fine-tune a small classifier (RoBERTa-base) to predict human judgments of detoxification quality from (toxic input, candidate output) pairs.
 
- **Upgrade to Frontier LLMs:** Re-run experiments with GPT-4o, Claude 3.5 Opus, Llama 3.3-70B-Instruct for masking/infilling to measure quality ceiling of LLM-based approaches. Test parameter-efficient fine-tuning (LoRA) of Mistral-7B on ParaDetox to create specialized detoxification model combining safety alignment with domain adaptation.
 
- **Real-Time Optimization:** Implement speculative decoding, KV-cache optimization, and quantization (4-bit) to reduce T5-base + Global Reranking latency from 1 second to <200ms per sentence for production deployment. Containerize best model (Docker + FastAPI) and deploy on AWS Lambda/GCP Cloud Run for scalable API serving.
 
- **Multilingual Extension:** Translate ParaDetox to 5 languages (Spanish, French, German, Hindi, Mandarin) via professional translators, fine-tune mT5-base on each language, and evaluate cross-lingual transfer. Test whether English-trained DecompX masking generalizes to non-English toxicity classifiers (XLM-R supports 100 languages).
 
---
 
## Credits / Data / Licenses
 
### Data Sources
 
- **ParaDetox Dataset** (Logacheva et al., 2022): 7,500 parallel toxic/non-toxic English sentence pairs for training + 671 test examples. Used under academic research license from [s-nlp/paradetox](https://github.com/s-nlp/paradetox). Licensed under CC BY-SA 4.0.
 
### Frameworks and Pre-trained Models
 
- **DecompX** (Modarressi et al., 2023): Token decomposition framework for Transformer explainability. Code adapted from [mohsenfayyaz/DecompX](https://github.com/mohsenfayyaz/DecompX). Licensed under MIT License.
  - Citation: Modarressi, A., Fayyaz, M., Aghazadeh, E., Yaghoobzadeh, Y., & Pilehvar, M. T. (2023). DecompX: Explaining Transformers Decisions by Propagating Token Decomposition. *ACL 2023*.
 
- **MaRCo Detoxification** (Hallinan et al., 2023): Product-of-Experts BART for controllable text revision. Implementation adapted from [shallinan1/MarcoDetoxification](https://github.com/shallinan1/MarcoDetoxification). Licensed under Apache 2.0.
  - Citation: Hallinan, S., Liu, A., Choi, Y., & Sap, M. (2023). Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts. *ACL 2023 Short Papers*.
 
- **XDetox** (Lee et al., 2024): Text detoxification with token-level toxicity explanations. Pipeline structure inspired by [LeeBumSeok/XDetox](https://github.com/LeeBumSeok/XDetox). Licensed under MIT License.
  - Citation: Lee, B., Kim, H., Kim, K., & Choi, Y. S. (2024). XDetox: Text Detoxification with Token-Level Toxicity Explanations. *EMNLP 2024*.
 
- **HuggingFace Transformers:** All models (T5, BART, RoBERTa, XLM-R, Mistral-7B, GPT-2) accessed via [HuggingFace Model Hub](https://huggingface.co/models). Licensed under Apache 2.0.
 
### Academic Partner
 
- **UC Berkeley School of Information**: Project conducted as W266 Natural Language Processing with Deep Learning final project (August 2024 - December 2024).
 
**Usage Terms:** This project is for academic/research purposes only. Model outputs may contain offensive content and should not be deployed in production without additional safety validation and human oversight.
 
---
 
## Team Members
 
| Name             | Email                          | LinkedIn                                           |
|------------------|--------------------------------|---------------------------------------------------|
| Benjamin He      | ben_he@berkeley.edu            | [LinkedIn](https://www.linkedin.com/in/ben-c-he/)    |
| Kent Bourgoing   | kentbourgoing@ischool.berkeley.edu | [LinkedIn](https://www.linkedin.com/in/kent-bourgoing/) |
 
---
 
**⚠️ Content Warning:** This repository contains examples of toxic language from research datasets and model outputs. Content may be offensive or disturbing. All examples are included for scientific evaluation purposes only.
