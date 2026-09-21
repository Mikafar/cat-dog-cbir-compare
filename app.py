import pickle
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

st.set_page_config(page_title="貓狗 CBIR — 3 種編碼模型比較", layout="wide")

MODELS = ["ResNet50", "VGG16", "DenseNet121"]
DIM = {"ResNet50": 2048, "VGG16": 4096, "DenseNet121": 1024}


# --------------------------------------------------
# 載入模型與特徵資料庫
# --------------------------------------------------
@st.cache_resource
def load_encoders():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    def build(name):
        net = getattr(models, name.lower())(weights="DEFAULT")
        if name.lower().startswith("resnet"):
            enc = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten())
        elif name.lower().startswith("vgg"):
            enc = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten(),
                                      *list(net.classifier.children())[:-1])
        elif name.lower().startswith("densenet"):
            enc = torch.nn.Sequential(*list(net.children())[:-1],
                                      torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten())
        else:
            enc = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten())
        enc = enc.to(device)
        enc.eval()
        return enc

    return {name: build(name) for name in MODELS}, device


@st.cache_data
def load_results():
    with open("compare_results.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_distance_results():
    return {
        "summary": pickle.load(open("distance_compare_results.pkl", "rb")),
        "margins": pickle.load(open("distance_margins.pkl", "rb")),
    }


encoders, device = load_encoders()
results = load_results()
dres = load_distance_results()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_single_feature(image, enc):
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = enc(tensor).squeeze().cpu().numpy()
    return feat / np.linalg.norm(feat)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🐱🐶 貓狗 CBIR — 三種編碼模型比較")
st.write("使用 3 個 pre-trained 編碼器 (ResNet50 / VGG16 / DenseNet121) 將圖片編碼成特徵向量，"
         "並以 Cosine KNN (k=5) 做分類。")

best = max(results, key=lambda n: (results[n]['acc'], results[n]['train_acc']))
st.markdown("#### 📋 各模型比較")
rows = "| 模型 | 特徵維度 | 訓練集準確率 | 未見 20 張準確率 |\n|---|---|---|---|"
for name in MODELS:
    r = results[name]
    star = " ⭐" if name == best else ""
    rows += f"\n| **{name}**{star} | {r['dim']} | {r['train_acc']:.2%} | {r['acc']:.2%} ({int((r['y_pred']==r['test_labels']).sum())}/{len(r['test_labels'])}) |"
st.markdown(rows)
st.info(f"**最佳模型**: {best} — 未見圖片準確率最高且訓練集最準確。")

tab1, tab2, tab3 = st.tabs(["🔍 查詢圖片：Top-5 相似/不相似 (三模型比較)",
                            "📊 20 張未見圖片測試與失敗案例 (三模型比較)",
                            "📏 距離計算比較 (Cosine / Euclidean / Manhattan)"])

# --------------------------------------------------
# TAB 1: query image across 3 models
# --------------------------------------------------
with tab1:
    c_s, c_u = st.columns([1, 3])
    with c_s:
        if st.button("🎲 使用一張範例查詢圖片"):
            import glob
            st.session_state['sample_query'] = Image.open(
                np.random.choice(sorted(glob.glob("dataset/*/*"))))
            if 'uploaded_query' in st.session_state:
                del st.session_state['uploaded_query']
    with c_u:
        uploaded = st.file_uploader("上傳一張貓或狗的圖片...", type=["jpg", "jpeg", "png"],
                                    key="q_up")
    if uploaded is not None:
        st.session_state['uploaded_query'] = Image.open(uploaded)
        if 'sample_query' in st.session_state:
            del st.session_state['sample_query']

    query_img = st.session_state.get('uploaded_query') or st.session_state.get('sample_query')

    if query_img is not None:
        is_sample = 'sample_query' in st.session_state and 'uploaded_query' not in st.session_state
        qc, rc = st.columns([1, 2])
        with qc:
            st.image(query_img, caption="範例查詢 (Sample)" if is_sample else "上傳的 Query Image",
                     width="stretch")

        for name in MODELS:
            st.divider()
            st.subheader(f"🔷 {name} (dim={DIM[name]})")
            feat = extract_single_feature(query_img, encoders[name])

            pred = results[name]['knn'].predict([feat])[0]
            proba = results[name]['knn'].predict_proba([feat])[0]
            label_text = "🐱 貓" if pred == 0 else "🐶 狗"
            st.success(f"**KNN 分類 ({name})**：{label_text}  (貓 {proba[0]*100:.1f}% | 狗 {proba[1]*100:.1f}%)")

            sims = cosine_similarity([feat], results[name]['train_features'])[0]
            sorted_idx = np.argsort(sims)[::-1]
            top5 = sorted_idx[:5]
            bot5 = sorted_idx[-5:][::-1]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔥 Top 5 最相似**")
                cols = st.columns(5)
                for i, idx in enumerate(top5):
                    with cols[i]:
                        st.image(results[name]['train_paths'][idx], width="stretch")
                        st.caption(f"{sims[idx]:.3f}\n{'貓' if results[name]['train_labels'][idx]==0 else '狗'}")
            with c2:
                st.markdown("**❄️ Top 5 最不相似**")
                cols = st.columns(5)
                for i, idx in enumerate(bot5):
                    with cols[i]:
                        st.image(results[name]['train_paths'][idx], width="stretch")
                        st.caption(f"{sims[idx]:.3f}\n{'貓' if results[name]['train_labels'][idx]==0 else '狗'}")

# --------------------------------------------------
# TAB 2: unseen test across 3 models
# --------------------------------------------------
with tab2:
    rng_state = st.session_state.get("rng", None)
    all_test = results['ResNet50']['test_paths']
    cat_idx = [i for i, (_, lb) in enumerate(all_test) if lb == 0]
    dog_idx = [i for i, (_, lb) in enumerate(all_test) if lb == 1]

    if rng_state is None or 'sel' not in st.session_state:
        rng_state = np.random.RandomState()
        st.session_state['rng'] = rng_state
    asample = st.session_state.get('sel', None)
    if asample is None:
        sel = sorted(rng_state.choice(cat_idx, 5, replace=False).tolist() +
                     rng_state.choice(dog_idx, 5, replace=False).tolist())
        st.session_state['sel'] = sel

    top = st.columns([2, 1])
    with top[0]:
        st.subheader("🎯 未見圖片測試（抽樣 10 張：5 貓 + 5 狗）— 三模型比較")
    with top[1]:
        if st.button("🎲 重新抽 10 張"):
            sel = sorted(rng_state.choice(cat_idx, 5, replace=False).tolist() +
                         rng_state.choice(dog_idx, 5, replace=False).tolist())
            st.session_state['sel'] = sel

    test_paths = [all_test[i] for i in st.session_state['sel']]

    st.markdown("#### ✅ 各模型預測結果")
    cols = st.columns(5)
    for col_i, (path, true_label) in enumerate(test_paths):
        img = Image.open(path).convert('RGB')
        preds = []
        for name in MODELS:
            feat = extract_single_feature(img, encoders[name])
            preds.append(int(results[name]['knn'].predict([feat])[0]))
        with cols[col_i % 5]:
            st.image(img, width="stretch")
            st.caption(f"真實: {'貓' if true_label==0 else '狗'}")
            for name, p in zip(MODELS, preds):
                ok = "✅" if p == true_label else "❌"
                st.caption(f"{name}: {'貓' if p==0 else '狗'} {ok}")
        if col_i % 5 == 4:
            st.divider()

    st.divider()
    st.subheader("⚠️ 失敗案例 (Failed Cases)")
    all_correct = True
    for path, true_label in test_paths:
        img = Image.open(path).convert('RGB')
        for name in MODELS:
            feat = extract_single_feature(img, encoders[name])
            if int(results[name]['knn'].predict([feat])[0]) != true_label:
                all_correct = False
                st.error(f"{name} 錯判 {os.path.basename(path)} "
                         f"(真實: {'貓' if true_label==0 else '狗'})")
    if all_correct:
        st.balloons()
        st.success("🎉 抽出的 10 張 × 3 模型 = 30 次分類皆正確！")
    else:
        st.markdown("""
        **失敗原因分析：**
        1. **背景佔比過高**：背景物件干擾特徵。
        2. **極端視角**：特寫／遠景失去判別訊號。
        3. **相似外型**：長毛小型犬與貓咪特徵重疊。
        """)

# --------------------------------------------------
# TAB 3: distance metric comparison
# --------------------------------------------------
with tab3:
    st.subheader("📏 三種距離計算的比較 (kNN = 5, 特徵皆已 L2 正規化)")
    st.info("""
    **背景說明**：因為所有特徵向量都做了 L2 正規化 (長度=1)，在數學上
    `Euclidean²(v,u) = 2 - 2·cos(v,u)`，所以 **Euclidean 與 Cosine 的排序
    完全等價** —— 這就是為什麼兩者結果幾乎完全相同。
    """)

    dm = dres['summary']
    st.markdown("#### 🔢 Accuracy 比較")
    hdr = "| 編碼器 | 距離 | LOOCV 準確率 (320) | 未見 20 張準確率 |"
    lines = [hdr, "|---|---|---|---|"]
    for enc in MODELS:
        for m, v in dm[enc].items():
            lines.append(f"| {enc} | **{m}** | {v['loocv']:.2%} | {v['unseen']:.2%} |")
    lines.append("")
    st.markdown("\n".join(lines))

    st.divider()
    st.markdown("#### 📐 Margin 分析 (未見圖片：最近同類距離 vs 最近異類距離)")

    marg = dres['margins']
    rows = "| 編碼器 | 距離 | Mean margin | 有風險(<0) | Mean d_same | Mean d_opp |\n|---|---|---|---|---|---|"
    for k, v in marg.items():
        enc, m = k
        if enc not in MODELS:
            continue
        rows += (f"\n| {enc} | **{m}** | {v['margin'].mean():+.4f} | {int((v['margin'] < 0).sum())}/20 "
                 f"| {v['d_same'].mean():.4f} | {v['d_opp'].mean():.4f} |")
    st.markdown(rows)

    st.divider()
    st.markdown("#### ⚖️ 優缺點分析 (Strength & Weakness)")

    st.markdown("""
    **1️⃣ Cosine Similarity 🟢**
    - **優點**：對「整體亮暗／對比度縮放」不敏感，只看特徵的方向；是高維特徵
      檢索 (CBIR) 最常用的指標。
    - **缺點**：忽略向量的大小 (magnitude)，若兩個影像只有「強度」差異會被視為相同；
      在高維空間中所有向量彼此趨近 (距離集中效應)，解析度較低。

    **2️⃣ Euclidean (L2) 🟡**
    - **優點**：保留完整的幾何距離資訊，直觀好理解 (真正的空間直線距離)。
    - **缺點**：對整體 scale 敏感 (亮度/對比會放大距離)；在 L2 正規化後與 Cosine
      完全等價，因此 **沒有額外資訊**——我們的數據也證實了兩者準確率相同。

    **3️⃣ Manhattan (L1) 🟠**
    - **優點**：逐維取絕對值差，對單一維度的極端離群值較不敏感 (robust)；
      在高維數據中常比 L2 更穩定。
    - **缺點**：距離值隨維度累加而變大 (尺度與 Cosine/L2 無法直接比較)；對所有維度
      一視同仁，容易受大量低訊號維度 (雜訊) 影響，導致準確率略降。

    **📌 結論**
    - 三者在此任務中差距很小 (99–99.4%)，代表 **特徵本身的品質才是主導因素**。
    - Cosine 與 L2 等價 → 選一個即可 (通常選 Cosine，方便當 similarity)。
    - Manhattan 在 VGG16 上略勝 (99.38%)，但在 ResNet50/DenseNet121 略輸，
      顯示它對「不同特徵分布」很敏感，不一定保證更好。
    """)