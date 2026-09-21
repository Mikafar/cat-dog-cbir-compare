import glob
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODELS = {
    "ResNet50":   ("resnet50",    2048),
    "VGG16":      ("vgg16",       4096),
    "DenseNet121":("densenet121", 1024),
}

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_encoder(model_name, feat_dim):
    fn = getattr(models, model_name)
    net = fn(weights="DEFAULT")

    if model_name.startswith("resnet"):
        # drop fc, keep conv+avgpool -> (2048,)
        encoder = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten())
    elif model_name.startswith("vgg"):
        # keep conv features + avgpool + classifier fc layers (except final 1000-way) -> (4096,)
        encoder = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten(),
                                      *list(net.classifier.children())[:-1])
    elif model_name.startswith("densenet"):
        # drop classifier, global avg pool conv features -> (1024,)
        encoder = torch.nn.Sequential(*list(net.children())[:-1],
                                      torch.nn.AdaptiveAvgPool2d((1, 1)),
                                      torch.nn.Flatten())
    else:
        encoder = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten())

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def extract(fn, net, path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = net(t).squeeze().cpu().numpy()
    return f / np.linalg.norm(f)


def collect(net, paths):
    feats = []
    for p in paths:
        f = extract(None, net, p)
        feats.append(f)
    return np.array(feats)


def load_paths(cls, suffix=""):
    return sorted(glob.glob(f"dataset/{cls}/{suffix}*"))


def main():
    train_paths = (load_paths("cat") + load_paths("dog"))
    train_labels = np.array([0] * 160 + [1] * 160)

    test_paths = [(p, 0) for p in load_paths("unseen_cat")] + [(p, 1) for p in load_paths("unseen_dog")]
    test_labels = np.array([l for _, l in test_paths])

    results = {}
    for name, (mname, dim) in MODELS.items():
        print(f"\n=== Encoding with {name} (dim={dim}) ===")
        net = build_encoder(mname, dim)

        X_train = collect(net, train_paths)
        print(f"train features: {X_train.shape}")

        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train, train_labels)

        X_test = collect(net, [p for p, _ in test_paths])
        y_pred = knn.predict(X_test)
        acc = accuracy_score(test_labels, y_pred)

        # top-1 / top-3 score report (train acc for sanity on the DB)
        train_pred = knn.predict(X_train)
        train_acc = accuracy_score(train_labels, train_pred)

        print(f"{name}: unseen acc = {acc:.4f} ({sum(y_pred == test_labels)}/{len(test_labels)}), "
              f"train acc = {train_acc:.4f}")
        print(confusion_matrix(test_labels, y_pred))

        results[name] = {
            "dim": dim,
            "train_features": X_train,
            "train_labels": train_labels,
            "train_paths": train_paths,
            "test_features": X_test,
            "test_labels": test_labels,
            "test_paths": test_paths,
            "y_pred": y_pred,
            "knn": knn,
            "acc": acc,
            "train_acc": train_acc,
        }

        with open(f"features_{name}.pkl", "wb") as f:
            pickle.dump(results[name], f)
        print("saved features_%s.pkl" % name)

    print("\n================ SUMMARY ================")
    for name, r in results.items():
        print(f"{name:12s} dim={r['dim']:5d}  train_acc={r['train_acc']:.4f}  unseen_acc={r['acc']:.4f} ({int((r['y_pred']==r['test_labels']).sum())}/{len(r['test_labels'])})")

    with open("compare_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("saved compare_results.pkl")


if __name__ == "__main__":
    main()