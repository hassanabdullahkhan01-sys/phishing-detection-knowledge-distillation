# Robust Phishing Detection via Knowledge Distillation and Character-Level Transformer Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassanabdullahkhan01-sys/phishing-detection-knowledge-distillation/blob/main/phishing_detection_dissertation.ipynb)

MSc Data Science dissertation, London South Bank University (2026).

**In short:** BERT detects phishing and smishing (SMS phishing) messages very accurately, but at 110M parameters and 437.9 MB it is too heavy to run privately on a phone. This project distils it into a DistilBERT student that **matches the teacher's accuracy while being 39% smaller and 56% faster**, then adds a character-level branch so the model can still read obfuscated text like `cl1ck h3r3 t0 l0g1n`.

## Results

| Metric | BERT-base teacher | DistilBERT student (distilled) |
|---|---|---|
| Test accuracy | 96.87% | **96.90%** |
| Parameters | 110M | **66.9M** |
| Model size | 437.9 MB | **267.8 MB** (−38.8%) |
| GPU inference latency | 3.33 ms | **1.47 ms** (55.8% faster) |

**Mobile hardware simulation (CPU only, student model):**

| Setting | Latency per message | Target |
|---|---|---|
| 2 threads (mid-range phone) | 74.87 ms | < 100 ms ✅ |
| 1 thread (entry-level phone) | 113.10 ms | < 100 ms ❌ (13 ms over) |
| Peak inference RAM | 0.05 MB | < 500 MB ✅ |

## Approach

1. **Data.** Merged the UCI SMS Spam Collection with a Hugging Face phishing dataset into **81,187 samples**, then split them 70/15/15 into train, validation and test sets.
2. **Teacher.** Fine-tuned BERT-base for 3 epochs (learning rate 2e-5, batch size 32) on an NVIDIA A100.
3. **Knowledge distillation.** Trained a DistilBERT student on both hard labels and the teacher's softened probabilities:

   `L_total = α · KL(σ(z_s/T) ‖ σ(z_t/T)) · T² + (1 − α) · CE`, where α = 0.5 and T = 2.0

4. **Character-level model (CharLevelDistilBert).** WordPiece tokenizers break on obfuscated words. This branch adds a character-level embedding and 2× Conv1D (64 filters) to DistilBERT's sub-word embeddings (concatenated to 832 dimensions, then projected to 768). It is trained with 15% character noise: visual substitutions, adjacent swaps and deletions.
5. **Adversarial testing.** Tested with TextAttack (DeepWordBug) and a leetspeak substitution attack on 100 phishing samples.
6. **Mobile simulation.** Disabled the GPU and limited PyTorch to 1 or 2 CPU threads, then profiled memory with `tracemalloc`.

## Adversarial robustness

| Model | DeepWordBug attack success | Leetspeak | Clean val. accuracy |
|---|---|---|---|
| DistilBERT student (standard KD) | 25.0% | More vulnerable | ~97.0% |
| CharLevelDistilBert | 47.0% | **More robust** ✅ | 96.12% |

The character-level model correctly classifies obfuscated messages such as `ph1sh1ng` and `g3t y0ur cr3d1t c4rd`. However, it was **more** vulnerable to DeepWordBug. That attack changed about 42% of characters on average, nearly three times the 15% noise the model was trained on, so this is a real limitation rather than a solved problem.

## Limitations and next steps

- **Simulated hardware only.** Thread-limited CPU runs don't capture thermal throttling or OS scheduling on real phones.
- **No quantization yet.** INT8 quantization (about 267.8 MB down to about 67 MB) and ONNX/TFLite export should bring single-thread latency under 100 ms.
- **Curriculum adversarial training.** Gradually raising noise from 15% to about 42% should close the DeepWordBug gap.
- **English only.** A multilingual backbone and an extended character vocabulary would cover Urdu, Arabic, Hindi and Mandarin smishing.

## How to run

1. Click **Open in Colab** above. A GPU runtime is recommended; the results above used an A100.
2. Run **Block 1 (Environment Setup)**, then choose **Runtime → Restart session**.
3. Run the remaining blocks in order:
   - Blocks 2–3: data loading and tokenization
   - Blocks 4–6: teacher training and benchmark
   - Blocks 7–9b: distillation, student benchmark and mobile simulation
   - Blocks 10–12: adversarial evaluation and the character-level model
   - Block 13: charts
4. The notebook saves models to Google Drive under `MyDrive/Hassan_Dissertation/`. Change these paths if your Drive is laid out differently.

Trained model weights aren't included because they exceed GitHub's 100 MB file limit.

**Built with:** Python, PyTorch, Hugging Face Transformers, TextAttack, Google Colab.

## Author
Hassan Abdullah Khan

**Hassan Abdullah Khan** · [LinkedIn](https://www.linkedin.com/in/hassan-khan-78512021a/)
