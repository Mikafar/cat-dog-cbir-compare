# Cat vs Dog CBIR — 3 Encoding Model Comparison

Same task as `cat_dog_project`, but compares **3 pre-trained encoders**:

| Model    | Feature dim | Train acc (320) | Unseen acc (20) |
|----------|-------------|-----------------|-----------------|
| ResNet50 | 2048        | 99.38%          | 100% (20/20)    |
| VGG16    | 4096        | 99.06%          | 100% (20/20)    |
| DenseNet121 | 1024     | 98.44%          | 100% (20/20)    |

## Website features

1. **Tab 1 — Query**: upload (or 🎲 sample) a cat/dog image; for each of the 3 encoders it shows
   the **top-5 most similar / top-5 most dissimilar** database images plus a KNN (k=5, cosine)
   classification with confidence.
2. **Tab 2 — Unseen test**: evaluates all **20 unseen images** (10 cats + 10 dogs) with all 3
   models (60 classifications), shows per-image predictions, and visualizes **failed cases**.

## Pipeline

- Dataset: 160 cat + 160 dog images (`dataset/cat`, `dataset/dog`), 20 unseen
  (`dataset/unseen_cat`, `dataset/unseen_dog`), all ResNet-verified as real pet photos.
- `encode_and_compare.py`: encodes every image with ResNet50 / VGG16 / DenseNet121
  (pre-classifier embeddings, L2-normalized), trains a cosine KNN per model, evaluates on the
  20 unseen images, and saves `features_<Model>.pkl` + `compare_results.pkl`.
- `app.py`: the Streamlit comparison website.

## Run locally (for others)

Requires only Python 3.9+ — one command sets up everything:

```bash
git clone https://github.com/Mikafar/cat-dog-cbir-compare.git
cd cat-dog-cbir-compare
python start_site.py                 # creates .venv, installs deps, starts site on :8502
python start_site.py 8503            # optionally pick a different port
```

(On Windows, use `py start_site.py`.) First run downloads torch + model weights.

## Developer run

```bash
cd ~/PycharmProjects/cat_dog_project_compare
.venv/bin/streamlit run app.py          # website on http://localhost:8502
.venv/bin/python encode_and_compare.py  # rebuild features + KNN + evaluation
```

Note: on this machine run torch downloads with
`SSL_CERT_FILE=<venv>/lib/python3.10/site-packages/certifi/cacert.pem` if you hit
`ssl.SSLCertVerificationError`.

## Deploy to Streamlit Community Cloud

1. Push repo `Mikafar/cat-dog-cbir-compare` to GitHub (already done).
2. Go to https://share.streamlit.io -> **Create app** -> "Yup, I have an app".
3. Select repo `Mikafar/cat-dog-cbir-compare`, branch `main`, main file `app.py`.
4. Optional: set a custom subdomain (e.g. `cat-dog-cbir-compare`).
5. Deploy. If memory limits hit while loading the 3 encoders, enable
   **High performance** mode under App Settings -> Resources.

## Compare

Use `compare_results.pkl` (dict keyed by model name) for further analysis:
`results[name]["acc"]`, `["train_acc"]`, `["y_pred"]`, `["test_labels"]`, `["knn"]`, etc.