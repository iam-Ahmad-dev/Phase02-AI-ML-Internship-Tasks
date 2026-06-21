# 🏠 Task 3 — Multimodal ML: Housing Price Prediction (Images + Tabular Data)

A multimodal deep learning pipeline that combines a **CNN image branch** and a
**Dense tabular branch** to predict California housing prices, built with
TensorFlow/Keras.

---

## ✅ Task Requirements Coverage

| Requirement | Status |
|---|---|
| Combine image + tabular data in one model | ✅ CNN branch + Dense branch, concatenated |
| CNN for image feature extraction | ✅ 3-layer Conv2D → MaxPool → Flatten → Dense(128) |
| Dense layers for tabular features | ✅ Dense(64) → Dense(32) |
| Regression output (price prediction) | ✅ Final Dense(1), optimized with MSE |
| Evaluate and compare vs baseline | ✅ MAE + RMSE, multimodal vs tabular-only |

---

## 📁 Project Structure

```
task3_multimodal/
├── Task3_Multimodal_Housing_Price_Prediction.ipynb   # Main notebook (run on Colab)
└── README.md                                         # This file
```

---

## 📦 Dataset

**California Housing Dataset** — built into scikit-learn, no download needed.

- 20,640 samples, 8 numeric features
- Target: median house value (in $100,000s)
- Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

**Synthetic images** are auto-generated for each sample (64×64 RGB). The image
brightness correlates with price, giving the CNN a real signal to learn — while
keeping the demo fully self-contained without needing a real image dataset.

---

## ⚙️ Setup (Google Colab — Recommended)

No local setup needed. Upload the notebook to Colab and run:

1. **Runtime → Change runtime type → T4 GPU → Save**
2. **Runtime → Run all**

The first cell installs all dependencies automatically.

### Local Setup (VS Code / Jupyter)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install tensorflow scikit-learn matplotlib numpy pandas jupyter
jupyter notebook Task3_Multimodal_Housing_Price_Prediction.ipynb
```

---

## 🏗️ Model Architecture

```
Image Input (64×64×3)          Tabular Input (8 features)
       │                                │
  Conv2D(32) + MaxPool             Dense(64, relu)
  Conv2D(64) + MaxPool             Dense(32, relu)
  Conv2D(128) + MaxPool                │
  Flatten → Dense(128)           [32-dim vector]
       │                                │
       └──────── Concatenate ───────────┘
                     │
               Dense(64, relu)
               Dropout(0.2)
               Dense(32, relu)
               Dense(1)  ← predicted price
```

---

## 🔧 Key Fix Applied (SyntaxError)

The original notebook had `generate_house_image()` split across **3 separate cells**:
- Cell 1: `def generate_house_image(price, size=(64, 64)):` — just the `def` line
- Cell 2: Markdown cell containing the docstring
- Cell 3: The function body

Python requires the `def` line and its body to be in the **same cell**. Splitting
them across cells causes `SyntaxError: incomplete input`. All three parts have
been merged into one clean code cell, along with `build_image_branch()` and
`build_tabular_branch()` which are now also in a single cell.

---

## 📊 Expected Results

| Model | MAE | RMSE |
|---|---|---|
| Multimodal (CNN + Tabular) | ~0.55–0.65 | ~0.75–0.90 |
| Tabular-only baseline | ~0.60–0.70 | ~0.80–0.95 |

The multimodal model should outperform the tabular-only baseline, demonstrating
that even synthetic image features add predictive signal.

Results may vary slightly each run due to random image generation.

---

## 💡 Real-World Extension

To use this with real house images:
1. Replace `generate_house_image()` with actual image loading (e.g. from a Zillow/Kaggle dataset)
2. Swap the 3-layer CNN with a pretrained model like `ResNet50` or `EfficientNet` for richer feature extraction
3. Add data augmentation (flips, crops) to regularize the image branch

---

## 🧠 Skills Gained

- Multimodal deep learning (combining heterogeneous data types)
- CNN architecture design for image feature extraction
- Keras Functional API for multi-input models
- Regression evaluation: MAE, RMSE
- Baseline comparison methodology

---

## 📌 References

- [California Housing Dataset — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Keras Functional API](https://keras.io/guides/functional_api/)
- [TensorFlow Conv2D docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
