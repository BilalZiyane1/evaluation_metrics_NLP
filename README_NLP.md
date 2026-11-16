# Evaluating Language Models: Summarization, Question Answering and Text Generation

Three self-contained Jupyter / Google Colab notebooks that **implement** three NLP tasks and **evaluate them with the metrics studied in the accompanying presentation**. Every metric is first written **from scratch** (so the formula is visible and testable), then cross-checked against a reference library, and finally applied to real systems and to deliberately broken outputs to see what each metric can and cannot detect.

| Notebook | Task | Main metrics |
|---|---|---|
| `NLP_01_Summarization.ipynb` | Summarization | ROUGE-1/2/L, BERTScore, entity/number support, SummaC-style NLI, meta-evaluation |
| `NLP_02_Question_Answering.ipynb` | Question Answering | Exact Match, token F1, SQuAD 2.0 no-answer protocol, equivalence / LLM-judge, Cohen's kappa, RAGAS-style RAG metrics |
| `NLP_03_Text_Generation.ipynb` | Text generation | Perplexity, BLEU, distinct-n, self-BLEU, repetition, MAUVE, FActScore, Elo / Bradley-Terry, judge biases |

---

## 1. Quick start

**Google Colab (recommended).** Upload a notebook, choose *Runtime > Run all*. The first cell of each notebook installs the extra libraries. For the neural parts a GPU runtime (*Runtime > Change runtime type > T4 GPU*) is faster but not required.

**Local.**
```bash
pip install numpy pandas matplotlib scikit-learn scipy rouge-score sacrebleu nbformat jupyter
# optional, only for the neural sections:
pip install torch transformers bert-score mauve-text
jupyter notebook
```

**Switches.** Each notebook has `USE_MODELS = True` in its setup cell. Set it to `False` to skip every neural model and run only the from-scratch part (about 15 seconds per notebook, no internet, no GPU).

**Colab-only cells.** Cells starting with `!pip install` are tagged for Colab. Locally, install the packages yourself as shown above.

---

## 2. Design principle: two layers, so that "Run all" never fails

1. **Layer 1, offline and tested.** All metrics written from scratch, cross-checked with `rouge-score`, `sacrebleu` and `scikit-learn`, plus small from-scratch systems (Lead-k, TextRank, an n-gram language model, a retrieve-and-extract QA system). Uses only bundled data. Each notebook was executed end-to-end with **0 errors**.
2. **Layer 2, real models (Colab).** Pretrained Hugging Face models (DistilBART, `bart-large-mnli`, `distilbert` for BERTScore, `roberta-base-squad2`, `flan-t5-base`, GPT-2, real `mauve-text`). Loaded by explicit `AutoTokenizer` / `AutoModel*` code, not by high-level pipelines, and wrapped in `try/except`. If a model cannot be loaded, the cell prints a message and the notebook **falls back to the Layer 1 equivalent**, so the remaining cells still run.

> **Honest status.** Layer 1 was executed and verified. Layer 2 uses stable, standard APIs but could not be executed during development (no access to the Hugging Face Hub from the build environment). If a model cell errors on your machine, the fallback protects the rest of the notebook; report the message and fix that cell.

All toy numbers used in the presentation are **asserted in code** (for example `assert` that ROUGE-1 of the toy example is 0.833, that BLEU is 32.2, that PPL is 2.02 and 10.0), so the slides and the notebooks cannot drift apart.

---

## 3. Notebook 1: Summarization

**Question answered:** how do we know whether a summary is good, and what do the common metrics miss?

### Data
Six short **fictional** news articles (bike lanes, a planet discovery, a staircase, energy, a school, a leopard) with human-written abstractive reference summaries. An optional cell loads real CNN/DailyMail (`abisee/cnn_dailymail`) on Colab.

### Parts
| Part | Content |
|---|---|
| 1 | Data and reference summaries |
| 2 | **ROUGE from scratch**: clipped n-gram counts for ROUGE-1/2, longest common subsequence for ROUGE-L, precision / recall / F1. Cross-checked against `rouge-score`: maximum difference over 108 numbers = **0.0** |
| 3 | **Extractive systems**: Lead-k, TextRank (TF-IDF cosine similarity + PageRank by power iteration), random-k, copy-all. Bootstrap confidence intervals |
| 4 | **Abstractive model** `sshleifer/distilbart-cnn-12-6` (Colab), with TextRank as fallback |
| 5 | **BERTScore**: greedy token matching on a similarity matrix (from scratch), a hashed character-trigram "lite" embedding for offline use, and the real `bert_score` with `distilbert-base-uncased` |
| 6 | **Factual consistency**: entity/number support rate, SummaC-ZS-style sentence-level NLI (`facebook/bart-large-mnli`, with a lexical-containment stand-in offline), **hallucination injection** by swapping numbers |
| 7 | **Sanity tests**: identity, paraphrase, drop a sentence, add an irrelevant sentence, shuffle words, swap facts, shown as a heat-map |
| 8 | **Meta-evaluation**: Spearman and Kendall correlation of each metric with an assumed quality ordering |

### Key results (offline run)

*Toy example (slide 3).* "the cat sat on the mat" vs. reference variants:

| Candidate | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---|---|---|
| small edit (lay) | 0.833 | 0.600 | 0.833 |
| **WRONG fact (dog)** | **0.833** | 0.600 | 0.833 |
| paraphrase (feline) | 0.333 | 0.200 | 0.333 |

ROUGE gives the **false** sentence exactly the same score as a harmless edit, and punishes a correct paraphrase.

*Extractive systems (6 articles, mean over articles):*

| System | R1-P | R1-R | R1-F | R2-F | RL-F | avg words |
|---|---|---|---|---|---|---|
| lead-1 | 0.590 | 0.426 | 0.495 | 0.272 | 0.385 | 22.0 |
| lead-2 | 0.431 | 0.509 | 0.464 | 0.245 | 0.370 | 36.3 |
| lead-3 | 0.359 | 0.578 | 0.439 | 0.227 | 0.345 | 50.2 |
| textrank-2 | 0.478 | 0.508 | 0.490 | 0.262 | 0.389 | 33.2 |
| random-2 | 0.299 | 0.275 | 0.286 | 0.110 | 0.229 | 27.5 |
| copy-all | 0.271 | 0.812 | 0.405 | 0.216 | 0.328 | 94.2 |

`copy-all` has the highest **recall** but terrible precision: recall alone rewards length, which is why F1 (or a fixed length) is used. Bootstrap intervals (e.g. lead-1 ROUGE-1 F1 0.495 [0.425, 0.567]) overlap for the top systems, so on 6 articles they cannot be ranked reliably.

*Hallucination injection* (numbers replaced by random ones, e.g. "40 km ... 2028 ... 18 million" becomes "127 km ... 6091 ... 61 million"):

| Summary | ROUGE-1 before > after | support rate before > after |
|---|---|---|
| human reference | 1.000 > 0.892 | 1.0 > **0.222** |
| lead-2 | 0.464 > 0.393 | 1.0 > 0.469 |
| textrank-2 | 0.490 > 0.399 | 1.0 > 0.350 |

ROUGE-1 drops only about 10 to 20% while the entity/number support rate collapses. **Overlap metrics barely notice wrong facts; consistency metrics do.**

*Perturbation tests (Part 7):*

| perturbation | R1 | R2 | RL | BERT-lite | support |
|---|---|---|---|---|---|
| identity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | 0.843 | 0.720 | 0.843 | 0.870 | 1.000 |
| drop a sentence | 0.787 | 0.778 | 0.787 | 0.852 | 1.000 |
| add irrelevant sentence | 0.898 | 0.895 | 0.898 | 0.922 | 1.000 |
| **shuffle words** | **1.000** | 0.101 | 0.312 | **1.000** | 1.000 |
| **swap facts** | 0.892 | 0.792 | 0.892 | 0.903 | **0.222** |

Shuffled words keep ROUGE-1 and BERTScore at 1.0 (bag of words); only ROUGE-2 and ROUGE-L detect it. Only the support rate detects swapped facts.

*Meta-evaluation (correlation with an assumed quality ordering):*

| metric | Spearman | Kendall tau |
|---|---|---|
| ROUGE-1 | 0.096 | 0.073 |
| ROUGE-2 | 0.570 | 0.424 |
| ROUGE-L | 0.490 | 0.383 |
| BERTScore-lite | 0.114 | 0.071 |
| support rate | 0.651 | 0.558 |
| **ROUGE-2 + support** | **0.848** | **0.696** |

Combining a fluency/overlap metric with a consistency metric correlates better than either alone.

### Notes and caveats
* The six articles and their reference summaries are **fictional and tiny**; results illustrate behaviour, not benchmark performance.
* The offline NLI stand-in is lexical containment. It scores a reworded human reference *lower* than copied text, which is exactly why real NLI models are used. The notebook says so in the output.
* In the BERTScore comparison, the real model is probed once; if it works it is used for all systems, otherwise the lite scorer is used for all, so scores in one table always come from the same scorer.

---

## 4. Notebook 2: Question Answering

**Question answered:** how is an answer judged correct, and why does exact match fail for modern, fluent LLM answers?

### Data
Four fictional passages (a bridge, a photosynthesis passage, a railway, a lake) and **27 questions**: 19 answerable and 8 **unanswerable** (SQuAD 2.0 style; gold answer list empty), with several accepted gold answers where appropriate.

### Parts
| Part | Content |
|---|---|
| 1 | Data in SQuAD format |
| 2 | **SQuAD metrics from scratch**: answer normalisation (lowercase, strip punctuation, drop articles a/an/the, collapse spaces), **Exact Match**, **token F1**, maximum over several gold answers; the unanswerable rule (correct only if you predict nothing) |
| 3 | **A QA system from scratch**: IDF-weighted sentence retrieval (crude prefix stemmer) + question-type-driven span extraction (who, when, how many, ...); returns an answer and a **no-answer score** |
| 4 | **SQuAD 2.0 protocol**: HasAns / NoAns breakdown, threshold sweep on the no-answer score, `best_f1`, comparison with "always answer" and "always abstain" baselines |
| 5 | **Real model** `deepset/roberta-base-squad2` with manual span decoding (best start/end pair, null score from the first token), fallback to the from-scratch system |
| 6 | **Why EM fails for LLM answers**: 18 verbose, LLM-style answers with human correctness labels; compared metrics: EM, F1 >= 0.5, *contains*, rule-based **equivalence**, optional `google/flan-t5-base` judge; agreement and **Cohen's kappa** (from scratch and via scikit-learn) |
| 7 | **RAGAS-style metrics** for retrieval-augmented QA (lexical proxies): faithfulness, answer relevance, context precision, context recall |
| 8 | Bootstrap confidence intervals and summary |

### Key results (offline run)

*Toy example (slide 6):* "the Eiffel Tower" vs gold "Eiffel Tower" gives EM = 1 (article removed); "Obama" vs "Barack Obama" gives EM = 0, F1 = 0.667; "Lyon" vs "Paris" gives 0 / 0. Predicting nothing on an unanswerable question scores 1.0; predicting "Paris" scores 0.0.

*SQuAD 2.0 protocol on the 27 questions (from-scratch system):*

| System | EM = F1 | HasAns F1 | NoAns accuracy | abstained |
|---|---|---|---|---|
| always answer | 0.630 | 0.842 | 0.125 | 3.7% |
| always abstain | 0.296 | 0.000 | 1.000 | 100% |
| **best threshold** (tau = 0.347) | **0.815** | 0.737 | 1.000 | 44.4% |

Choosing the no-answer threshold trades answer quality for the ability to abstain. The best F1 is found by sweeping tau, exactly as in the official SQuAD 2.0 evaluation. Reporting `HasAns` and `NoAns` separately exposes what a single number hides.

*Exact match vs. human judgement on 18 verbose answers* (e.g. "The Harbour Bridge was designed by the engineer Lucia Ferreira." for gold "Lucia Ferreira"):

| Metric | estimated accuracy | agreement with human | Cohen's kappa |
|---|---|---|---|
| Exact Match | 0.000 | 0.222 | 0.000 |
| F1 >= 0.5 | 0.222 | 0.444 | 0.151 |
| contains gold | 0.500 | 0.722 | 0.444 |
| **equivalence rules** | 0.722 | **0.944** | **0.852** |
| human (reference) | 0.778 | 1.000 | 1.000 |

**EM says the system is 0% correct; humans say 78%.** This reproduces, in miniature, the finding of Kamalloo et al. (2023) on Natural Questions (InstructGPT: 12.6% EM vs. 71.4% human-judged correct). Kappa (agreement beyond chance) shows how much better the rule-based equivalence check is than EM. Our from-scratch kappa matches `sklearn.metrics.cohen_kappa_score`.

*RAG diagnostics (three scenarios on the same question):*

| Scenario | faithfulness | answer relevance | context precision | context recall |
|---|---|---|---|---|
| A: good context, faithful answer | 1.000 | 0.345 | 0.5 | 1.0 |
| B: good context, **hallucinated** answer | **0.375** | 0.236 | 0.5 | 1.0 |
| C: wrong context, correct answer from memory | 0.250 | 0.345 | **0.0** | **0.0** |

Scenario B stays on topic but invents facts: only faithfulness drops. Scenario C answers correctly but not from the retrieved evidence: context precision and recall are zero, so the answer is **not grounded**. Each metric catches a different failure; none replaces the others.

### Notes and caveats
* The **human labels in Part 6 were written by the author** as an illustration. They are not a real annotation study.
* The RAGAS-style metrics are **lexical proxies**, not the LLM-based RAGAS implementation; they demonstrate the logic of each metric.
* With 27 questions the bootstrap 95% interval for the best system is wide (about 0.67 to 0.96): far too few to rank similar systems. Real benchmarks use thousands.

---

## 5. Notebook 3: Text Generation

**Question answered:** how do we evaluate open-ended text for which there is no single correct output?

### Data
1,301 clean English prose sentences (19,819 words) extracted from **Python's own built-in documentation** (`pydoc_data.topics`, available in every Python installation, so no download). The text is technical but genuine human-written English. Split 70/10/20 into train (910), validation (130) and test (261) sentences.

### Parts
| Part | Content |
|---|---|
| 1 | Corpus extraction and splits |
| 2 | **Perplexity**: formula, toy checks, an **interpolated unigram/bigram/trigram language model** with weights tuned on the validation set, learning curve, and the **tokenizer caveat** (per word vs per character) |
| 3 | **Decoding strategies**: greedy, temperature (0.7, 1, 1.5), top-k, top-p (nucleus); 261 generated texts per setting |
| 4 | **Diversity and repetition**: distinct-1/2, self-BLEU, repetition rate, perplexity of the output, quality-vs-diversity plot |
| 5 | **BLEU from scratch** (clipped precision, brevity penalty, sacreBLEU-style smoothing), cross-checked with sacreBLEU; why BLEU is a weak proxy for open-ended text |
| 6 | **MAUVE-style score from scratch**: TF-IDF > SVD > k-means quantisation > divergence frontier (c = 5) > area under the curve |
| 7 | **GPT-2 on Colab**: word-level perplexity, generation with five decodings, same diversity metrics, MAUVE-lite and the **real MAUVE** (`mauve-text`) |
| 8 | **FActScore-style evaluation**: atomic facts checked against a knowledge base |
| 9 | **Pairwise evaluation**: Elo, Bradley-Terry, bootstrap intervals, simulated **LLM-judge biases** (position, verbosity), **length-controlled** logistic regression |

### Key results (offline run)

*Perplexity.* Toy values from slide 9: 2.02 and 10.00; a uniform model over 50,000 words has PPL = 50,000 (perplexity of a guesser equals the vocabulary size).

| Language model | Test perplexity |
|---|---|
| unigram only | 211.9 |
| bigram-ish (0.2 / 0.8 / 0.0) | 96.2 |
| **tuned interpolation** (0.4 / 0.4 / 0.2) | **78.7** |

*Learning curve* (fixed vocabulary): test perplexity falls from **215.7** with 10% of the training data to 159.5, 111.4, 88.7 and **78.7** with 100%. The vocabulary is held fixed on purpose: if it grew with the data, perplexities would not be comparable, because a tiny vocabulary turns most words into `<unk>`, which is easy to predict.

*Tokenizer caveat.* The same model on the same text has perplexity **78.7 per word** but **2.19 per character**: the total log-likelihood is identical, only the unit changed. Compare models only at the same unit (per word is used when comparing the n-gram model with GPT-2).

*Decoding strategies:*

| Setting | avg len | distinct-1 | distinct-2 | self-BLEU | repeated trigrams | PPL of output |
|---|---|---|---|---|---|---|
| greedy | 15.7 | 0.075 | 0.134 | 83.8 | 35.5% | 27.3 |
| T = 0.7 | 11.2 | 0.211 | 0.638 | 33.3 | 0.03% | 25.4 |
| pure sampling (T = 1) | 13.5 | 0.220 | 0.808 | 18.3 | 0% | 80.7 |
| top-k (k = 10) | 9.1 | 0.193 | 0.535 | 45.1 | 0.06% | 22.4 |
| top-p (p = 0.9) | 13.1 | 0.198 | 0.766 | 22.8 | 0.02% | 54.7 |
| T = 1.5 | 17.9 | 0.209 | 0.949 | 11.3 | 0% | 495.9 |
| **human text** | 14.7 | 0.245 | 0.762 | 18.5 | 0.2% | 78.3 |

Greedy decoding gets a low perplexity (the model loves its own favourite words) but degenerates into loops ("the the the ..."): distinct-2 of 0.13 and 35% repeated trigrams. T = 1.5 is very diverse but the model finds it absurd (perplexity about 496). **Human text sits in between.** Low perplexity of the output does *not* mean good text (Holtzman et al., 2020).

*BLEU.* The toy example reproduces the slide (p1 = 5/6, p2 = 3/5, brevity penalty 0.846, BLEU = **32.16**, identical to sacreBLEU). Against the single human continuation, every decoding scores only 0.55 to 5.0 BLEU: BLEU rewards copying one reference, which is the wrong objective for open-ended generation.

*MAUVE-lite* (TF-IDF features; 1 = same distribution as human text):

| Setting | MAUVE-lite |
|---|---|
| T = 0.7 | 0.871 |
| top-p (p = 0.9) | 0.871 |
| T = 1.5 | 0.823 |
| pure sampling (T = 1) | 0.746 |
| top-k (k = 10) | 0.716 |
| **greedy** | **0.128** |

Sanity checks: two halves of human text 0.870 (should be near 1); "the the the ..." 0.004 (should be near 0). **Honest limitation:** because TF-IDF is a bag of words, MAUVE-lite gives **word-shuffled human text a perfect 1.000** and barely penalises T = 1.5 word salad. The real MAUVE uses GPT-2 embeddings, which are sensitive to word order and fluency (the reason the notebook includes the real one in Part 7).

*FActScore (slide 10).* A fictional biography with five atomic facts, three supported by the knowledge base: **3/5 = 60%**. A shorter biography with two supported facts scores 100%, which shows why the real FActScore adds a length penalty and why it measures precision only (not recall).

*Elo and Bradley-Terry.* 1200 vs 1000 gives a 76.0% win probability; after A beats B (both 1000, K = 32) the ratings become 1016 and 984. From 3,000 simulated votes between four models of true ratings 1125 / 1025 / 975 / 875 (centred):

| model | true | online Elo | Bradley-Terry |
|---|---|---|---|
| A | 1125 | 1135 | 1118 |
| B | 1025 | 1038 | 1023 |
| C | 975 | 949 | 973 |
| D | 875 | 878 | 886 |

Bradley-Terry recovers the ratings more closely; online Elo depends on vote order. Bootstrap intervals shrink as votes go from 100 to 3,000.

*Judge biases (simulation).* A simulated judge with a position bonus and a verbosity bonus agrees with true quality 89.5% of the time in a single order and 92.1% when both orders are averaged; the same winner survives a swap of positions only 83.8% of the time; for pairs of **equal** quality the longer answer wins **69.8%** of the time (50% would mean no length bias).

*Length-controlled evaluation.* Three models: good (skill 0.3 centred), verbose (0.0, answers 2.2 times longer), bad (-0.3). The raw judge ranking is **verbose > good > bad** (the verbose model wins by writing more); adding a length term to the Bradley-Terry regression gives **good > verbose > bad**, matching the true skills (estimated length coefficient 1.16 vs. true 1.2). This is the idea behind length-controlled AlpacaEval.

### Notes and caveats
* The GPT-2 part (Part 7) runs only on Colab with internet; offline it prints a message and is skipped.
* The Elo / judge-bias experiments are **simulations** with known ground truth, designed to show how the method behaves, not measurements of a real LLM judge.
* The FActScore example uses a hand-written knowledge base and hand-written atomic facts to show the scoring logic; the real system decomposes text with an LLM and retrieves evidence from Wikipedia.

---

## 6. Metric cheat-sheet (all formulas used)

| Metric | Formula / idea | Blind spot demonstrated |
|---|---|---|
| ROUGE-N | clipped n-gram overlap; P = overlap / candidate n-grams, R = overlap / reference n-grams, F1 = 2PR/(P+R) | same score for a true and a false sentence; penalises paraphrase |
| ROUGE-L | longest common subsequence | order-aware but still lexical |
| BERTScore | greedy cosine matching of contextual token embeddings; P, R, F1 | bag-of-embeddings: shuffled words still score 1.0 (lite version) |
| Support rate | share of reference/summary entities and numbers found in the source | needs anchors; blind to fluency |
| SummaC-ZS | sentence-level NLI entailment against the source, max over source sentences, mean over summary sentences | needs a good NLI model |
| Exact Match | 1 if normalised prediction equals a normalised gold answer | 0% for fluent sentence answers |
| Token F1 | harmonic mean of token precision and recall of prediction vs gold | still low for verbose answers |
| SQuAD 2.0 | unanswerable correct only if the system abstains; threshold on no-answer score | one number hides HasAns / NoAns split |
| Cohen's kappa | (p_o - p_e) / (1 - p_e): agreement beyond chance | needs human labels |
| Faithfulness (RAGAS) | share of answer claims supported by the context | lexical proxy here |
| Perplexity | exp(-(1/N) sum ln p(w_t given w_<t)) | not comparable across tokenizers; low is not always good |
| BLEU | BP * exp(sum_n w_n ln p_n), BP = min(1, exp(1 - r/c)) | one-reference objective |
| distinct-n | unique n-grams / all n-grams | says nothing about truth |
| self-BLEU | BLEU of each output against the other outputs | diversity only |
| MAUVE | area under the divergence frontier between quantised human and model distributions | depends on the embedding |
| FActScore | supported atomic facts / atomic facts | rewards saying less; precision only |
| Elo | P(A beats B) = 1 / (1 + 10^(-(R_A - R_B)/400)); update K (S - E) | order dependent; needs many votes |
| Bradley-Terry | maximum-likelihood ratings from all pairwise outcomes | judge biases leak into ratings |

---

## 7. Repository layout (suggested)

```
.
├── README.md                          <- this file
├── requirements.txt
├── notebooks/
│   ├── NLP_01_Summarization.ipynb
│   ├── NLP_02_Question_Answering.ipynb
│   └── NLP_03_Text_Generation.ipynb
└── presentation/
    ├── Evaluating_Language_Models_NLP.pptx
    ├── Speech_Script_NLP.pdf
    └── Metrics_Explained_NLP.pdf
```

**requirements.txt**
```
numpy
pandas
matplotlib
scikit-learn
scipy
rouge-score
sacrebleu
nbformat
jupyter
# optional, for the neural sections (Colab already has torch / transformers):
torch
transformers
bert-score
mauve-text
```

---

## 8. Models used (Colab sections)

| Task | Model | Purpose |
|---|---|---|
| Summarization | `sshleifer/distilbart-cnn-12-6` | abstractive summaries |
| Summarization | `distilbert-base-uncased` (via `bert-score`) | real BERTScore |
| Summarization | `facebook/bart-large-mnli` | NLI for SummaC-style consistency |
| QA | `deepset/roberta-base-squad2` | extractive QA with no-answer |
| QA | `google/flan-t5-base` | small LLM judge (optional) |
| Generation | `gpt2` | perplexity, generation, MAUVE features |

The first run downloads a few gigabytes. Models are not committed to the repository.

---

## 9. Troubleshooting

* **"Model not available here (ModuleNotFoundError / OSError)".** No `transformers` or no internet. The notebook continues with the from-scratch fallback; install the packages or run on Colab to use the real model.
* **Out of memory on GPU.** Restart the runtime, or set `USE_MODELS = False`.
* **`mauve-text` fails.** It is optional; MAUVE-lite from scratch is always computed.
* **Numbers differ slightly from this README.** Neural outputs depend on hardware and library versions. The offline results use fixed seeds and should be reproducible.

---

## 10. Suggested exercises

Each notebook ends with its own exercises. Examples:
1. Summarization: add a **Lead-3 + TextRank hybrid** and test whether it beats both on ROUGE-2 + support; add a human-ranked set and recompute the correlations.
2. QA: add a second answer style (very short vs. verbose) and compare EM, F1 and kappa; implement an **F1 threshold sweep** for the judge.
3. Generation: add a **4-gram** order; compare `min_count` 1, 2, 5; add top-p 0.5 and 0.95 to the MAUVE table; replace the simulated judge by a real LLM judge and measure its position consistency.

---

## 11. References

* Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
* Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR.
* Laban et al. (2022). *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL.
* Rajpurkar et al. (2016, 2018). *SQuAD* and *Know What You Don't Know: Unanswerable Questions for SQuAD.*
* Kamalloo et al. (2023). *Evaluating Open-Domain Question Answering in the Era of Large Language Models.* ACL.
* Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.*
* Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL.
* Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration.* ICLR.
* Pillutla et al. (2021). *MAUVE: Measuring the Gap Between Neural Text and Human Text.* NeurIPS.
* Min et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP.
* Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.
* Chiang et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.*
* Dubois et al. (2024). *Length-Controlled AlpacaEval.*
* Liu et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.*

---

## 12. Data and licence notes

* All articles, passages, questions and the biography are **fictional**, written for this project.
* Text generation uses sentences from Python's bundled documentation (Python Software Foundation licence), loaded at run time and not redistributed.
* Optional real datasets (CNN/DailyMail) are downloaded from the Hugging Face Hub under their own licences.
* Human labels in the QA and summarization demonstrations were written by the author and are illustrative.
