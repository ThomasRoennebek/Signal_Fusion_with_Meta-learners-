# Master Thesis: Methodology & Final Testing Framework

**Core Research Question:** *Can weak supervision provide a robust foundational representation for complex procurement contracts, and does Meta-Learning optimize the adaptation of this foundation to local departmental nuances better than standard fine-tuning under extreme few-shot constraints?*

---

## Stage 1 Summarized: Representation Learning & Weak Supervision

### The Method
Procurement data often contains vast amounts of unlabeled contracts, making traditional supervised deep learning impossible. Stage 1 tested the hypothesis that **Weak Supervision (via Snorkel)** could successfully distill domain heuristics into a massive, noisy, but structurally sound training dataset. 

We trained standard tabular baselines (XGBoost, Elastic Net) and Neural Networks (MLP) across three simulated conditions:
1.  **`A_weak_only`**: Pre-training strictly on the massive dataset of weak labels.
2.  **`B_gold_only`**: Training strictly on the tiny amount of available manual expert labels.
3.  **`C_hybrid`**: Jointly training on weak and gold labels simultaneously.

### The Conclusion
1.  **Gold-Only neural networks are unstable.** Due to the tiny sample size and high class imbalance, standard network architectures severely overfit or collapse (highlighted mathematically by catastrophic failures in Batch Normalization statistics).
2.  **Weak Supervision is a massive success.** Given a properly widened architecture with normalization, the `MLP_A_weak_only` explicitly tied the Gradient Boosted baseline (XGBoost) in nearly every top-line metric (0.9375 AUROC, 0.956 F1, 0.31 Logloss). 
3.  **Joint Training (`C_hybrid`) breaks few-shot simulation.** Mixing datasets randomly from scratch allows data leakage and does not represent real-world "transfer learning."

**Final Verdict for Stage 1:** Weak supervision successfully learned the global structure of a contract entirely unsupervised. `MLP_A_weak_only` is validated as a mathematically sound, highly competitive baseline feature extractor perfectly suited to act as the initialization state for Meta-Learning.

---

## Stage 2 Finalized Framework: Few-Shot Meta-Learning

### The Method 
We transition from *learning the global representation* to *adapting it to local departments*. 

Since we only possess a severely limited number of gold expert labels per department, Stage 2 tests the optimization mechanics of "Few-Shot Adaptation". We formally load the saved, frozen weights of `MLP_A_weak_only` and hand the algorithms a highly restricted **Support Set** (e.g., 5-10 gold labels).

### The Experimental Testing Setup (The Bulletproof Comparison)
To isolate and prove the value of Meta-Learning, all models must start from the exact same weakly-supervised foundation and use the exact same tiny Support Set of gold labels. 

The final thesis tables will compare the models against the hold-out **Gold Test Set**:
1.  **Baseline 1: Zero-Shot Transfer**: Evaluating `MLP_A_weak_only` directly against the Gold Test Set with absolutely no adaptation.
2.  **Baseline 2: Sequential Fine-Tuning**: Taking the foundational network, freezing its early layers, and using standard backpropagation (e.g., Adam/SGD) on the Support Set. This represents the "standard machine learning" approach.
3.  **The Challenger: Meta-Learning**: Taking the foundational network and utilizing episodic Meta-Learning algorithms (e.g., MAML, FOMAML, ANIL) to optimize adaptation on the Support Set.

### Overall Thesis Conclusion Hypothesis
By structuring the experiment this way, your thesis will ultimately conclude:
1.  That Weak Supervision successfully eliminates the need for massive manual annotation infrastructure in corporate procurement pipelines.
2.  That **Meta-Learning mathematically and practically outperforms standard Fine-Tuning** when adjusting these general foundations to department-specific logic. This proves that specialized meta-algorithms are strictly necessary when operating under extreme corporate data scarcity.
