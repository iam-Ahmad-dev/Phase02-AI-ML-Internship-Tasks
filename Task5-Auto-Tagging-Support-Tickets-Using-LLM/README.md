
# 🏷️ Task 5 — Auto Tagging Support Tickets Using LLM

Automatically classify support tickets into categories using prompt engineering
with Mistral-7B-Instruct. Compares zero-shot vs few-shot performance and outputs
the top 3 most probable tags per ticket.

---

## ✅ Task Requirements Coverage

| Requirement | Status |
|---|---|
| Use prompt engineering with an LLM | ✅ Mistral-7B-Instruct via HuggingFace InferenceClient |
| Compare zero-shot vs few-shot performance | ✅ Both run on every ticket, side-by-side comparison |
| Apply few-shot learning techniques | ✅ 5 labeled examples injected into few-shot prompt |
| Output top 3 most probable tags per ticket | ✅ JSON-parsed top-3 tags per ticket, ordered by relevance |

---

## 📁 Project Structure

```
task5_autotagging/
├── auto_tag.py           # Main script — run this
├── tagging_results.json  # Output file (created after running)
└── README.md             # This file
```

---

## 📦 Dataset

**Built-in support ticket dataset** — 20 realistic tickets covering common
support scenarios. No download needed.

Available tags (10 categories):
`billing` · `technical_issue` · `account_access` · `shipping_delivery` ·
`refund_return` · `product_defect` · `feature_request` · `password_reset` ·
`subscription` · `general_inquiry`

Each ticket has ground-truth labels for evaluation.

---

## ⚙️ Setup

### Step 1 — Get a Free HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → name it (e.g. "AutoTag")
3. Under **Inference**, tick ✅ **"Make calls to Inference Providers"**
4. Save and copy the token (starts with `hf_`)

### Step 2 — Install Dependencies

```bash
pip install huggingface-hub
```

That's the only dependency needed.

### Step 3 — Set Your Token

Open `auto_tag.py` and replace line 22:
```python
HF_TOKEN = "hf_your_token_here"   # <-- replace this
```
with your actual token:
```python
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

---

## ▶️ How to Run

```bash
python auto_tag.py
```

Runtime: ~3-5 minutes for all 20 tickets (2 API calls each with delay to avoid rate limits).

---

## 📊 What the Script Does

### Zero-Shot Prompt
No examples given — the model classifies purely from the tag definitions:
```
You are a support ticket classification system.
Available tags: billing, technical_issue, ...
Support ticket: "I was charged twice last week."
Response (JSON only):
```

### Few-Shot Prompt
5 labeled examples shown before the target ticket:
```
Ticket: "I was double charged on my credit card this month."
Response: {"tags": ["billing", "refund_return"]}

...4 more examples...

Now classify this ticket:
Ticket: "I was charged twice last week."
Response (JSON only):
```

### Output Format
```json
{"tags": ["billing", "refund_return", "subscription"]}
```

### Evaluation Metrics
- **Top-1 Accuracy**: predicted[0] matches any ground-truth tag
- **Top-3 Accuracy**: any of the 3 predicted tags matches ground truth

---

## 📈 Expected Results

| Metric | Zero-Shot | Few-Shot |
|---|---|---|
| Top-1 Accuracy | ~60–70% | ~70–80% |
| Top-3 Accuracy | ~80–90% | ~85–95% |

Few-shot typically outperforms zero-shot because the examples teach the model
the expected tag vocabulary and output format, reducing parsing errors.

---

## 📂 Output File

After running, `tagging_results.json` is created containing:
- Model name and configuration
- Per-ticket predictions for both zero-shot and few-shot
- Accuracy metrics for both approaches
- Full comparison table

---

## 🧠 Skills Gained

- Prompt engineering (zero-shot and few-shot techniques)
- LLM-based text classification with structured JSON output
- Robust JSON parsing with fallback handling
- Multi-label classification and top-k tag ranking
- Evaluation metrics for multi-label tasks (Top-1, Top-3 accuracy)

---

## 📌 References

- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [HuggingFace InferenceClient](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)
- [Few-shot prompting guide](https://www.promptingguide.ai/techniques/fewshot)
