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


encoders, device = load_encoders()
results = load_results()

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

tab1, tab2 = st.tabs(["🔍 查詢圖片：Top-5 相似/不相似 (三模型比較)",
                      "📊 20 張未見圖片測試與失敗案例 (三模型比較)"])

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
    st.subheader("20 張未見圖片測試 — 三模型準確率比較")
    test_paths = results['ResNet50']['test_paths']

    st.markdown("#### ✅ 各模型預測結果")
    conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    cols = st.columns(5)
    for i, (path, true_label) in enumerate(test_paths):
        img = Image.open(path).convert('RGB')
        preds = []
        for name in MODELS:
            feat = extract_single_feature(img, encoders[name])
            preds.append(int(results[name]['knn'].predict([feat])[0]))
        with cols[i % 5]:
            st.image(img, width="stretch")
            st.caption(f"真實: {'貓' if true_label==0 else '狗'}")
            for name, p in zip(MODELS, preds):
                ok = "✅" if p == true_label else "❌"
                st.caption(f"{name}: {'貓' if p==0 else '狗'} {ok}")
        if i == 4:
            st.divider()

    st.divider()
    st.subheader("⚠️ 失敗案例 (Failed Cases)")
    all_correct = True
    for i, (path, true_label) in enumerate(test_paths):
        img = Image.open(path).convert('RGB')
        for name in MODELS:
            feat = extract_single_feature(img, encoders[name])
            if int(results[name]['knn'].predict([feat])[0]) != true_label:
                all_correct = False
                st.error(f"{name} 錯判 {os.path.basename(path)} "
                         f"(真實: {'貓' if true_label==0 else '狗'})")
    if all_correct:
        st.balloons()
        st.success("🎉 全部 20 張 × 3 模型 = 60 次分類皆正確！")
    else:
        st.markdown("""
        **失敗原因分析：**
        1. **背景佔比過高**：背景物件干擾特徵。
        2. **極端視角**：特寫／遠景失去判別訊號。
        3. **相似外型**：長毛小型犬與貓咪特徵重疊。
        """)