from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

RANDOM_STATE = 42
BATCH_SIZE = 64
VALID_SIZE = 0.2

path= "tested.csv"

data= pd.read_csv(path)
print(data.head())
print(data.isnull().sum()*100/data.shape[0])

def seed_everything(seed=RANDOM_STATE):
    import random, os
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

seed_everything()

assert "Survived" in data.columns, "No encuentro la columna objetivo 'Survived' en el CSV."

if {"SibSp", "Parch"}.issubset(data.columns):
    data["FamilySize"] = data["SibSp"].fillna(0) + data["Parch"].fillna(0) + 1
    data["IsAlone"]   = (data["FamilySize"] == 1).astype(int)
else:
    if "FamilySize" not in data.columns:
        data["FamilySize"] = np.nan
    if "IsAlone" not in data.columns:
        data["IsAlone"] = np.nan

cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in data.columns]

target_col = "Survived"
X = data.drop(columns=[target_col] + cols_to_drop, errors="ignore").copy()
y = data[target_col].astype(int).copy()

candidate_numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
candidate_categ   = ["Sex", "Embarked", "Pclass"]

numeric_features = [c for c in candidate_numeric if c in X.columns]
categorical_features = [c for c in candidate_categ if c in X.columns]

X_train_df, X_valid_df, y_train, y_valid = train_test_split(
    X, y, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

X_train_np = preprocessor.fit_transform(X_train_df)
X_valid_np = preprocessor.transform(X_valid_df)

feature_names = []
if len(numeric_features) > 0:
    feature_names += numeric_features
if len(categorical_features) > 0:
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names += ohe_names

print(f"Shape train: {X_train_np.shape}, valid: {X_valid_np.shape}")
print("Ejemplo de features:", feature_names[:10])

X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
X_valid_t = torch.tensor(X_valid_np, dtype=torch.float32)

y_train_bce = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_valid_bce = torch.tensor(y_valid.values.reshape(-1, 1), dtype=torch.float32)

class_counts = np.bincount(y_train.values)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train.values]

train_sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(sample_weights),
    replacement=True
)

train_ds = TensorDataset(X_train_t, y_train_bce)
valid_ds = TensorDataset(X_valid_t, y_valid_bce)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Listo el pipeline ✅")
print(f"Clases (train): {dict(zip(range(len(class_counts)), class_counts))}")

X_train_sq_t = X_train_t ** 2
X_valid_sq_t = X_valid_t ** 2

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x):
        return self.linear(x)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_n = 0.0, 0
    logits_list, ys_list = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        logits_list.append(logits.cpu())
        ys_list.append(yb.cpu())
    avg_loss = total_loss / max(total_n, 1)
    logits_all = torch.cat(logits_list, dim=0).numpy().ravel()
    y_all = torch.cat(ys_list, dim=0).numpy().ravel()
    y_pred = (1/(1+np.exp(-logits_all)) >= 0.5).astype(int)
    acc = accuracy_score(y_all, y_pred)
    prec = precision_score(y_all, y_pred, zero_division=0)
    rec = recall_score(y_all, y_pred, zero_division=0)
    return avg_loss, {"accuracy": acc, "precision": prec, "recall": rec}

def train_model_logreg(
    X_dim,
    train_loader,
    valid_loader,
    epochs=25,
    lr=1e-3,
    weight_decay=0.0,
    pos_weight=None,
    verbose=True,
    title="LogReg"
):
    model = LogisticRegressionTorch(X_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "valid_loss": [], "valid_acc": [], "valid_prec": [], "valid_rec": []}

    for ep in range(1, epochs+1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * yb.size(0)
            n += yb.size(0)
        train_loss = running / max(n, 1)

        valid_loss, metrics = eval_epoch(model, valid_loader, criterion)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(metrics["accuracy"])
        history["valid_prec"].append(metrics["precision"])
        history["valid_rec"].append(metrics["recall"])

        if verbose and (ep % 5 == 0 or ep == 1 or ep == epochs):
            print(f"[{title}] Epoch {ep:02d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | "
                  f"acc={metrics['accuracy']:.3f} | prec={metrics['precision']:.3f} | rec={metrics['recall']:.3f}")

    return model, history

def plot_history(history, title="Training Curves"):
    epochs = range(1, len(history["train_loss"])+1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title); plt.legend(); plt.grid(True)
    plt.show()

neg = (y_train_bce.numpy().ravel() == 0).sum()
pos = (y_train_bce.numpy().ravel() == 1).sum()
pos_weight = torch.tensor([neg / max(pos,1)], dtype=torch.float32).to(device)

X_dim = X_train_t.shape[1]
model_lr, hist_lr = train_model_logreg(
    X_dim,
    train_loader,
    valid_loader,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    pos_weight=pos_weight,
    title="LogReg (baseline)"
)

plot_history(hist_lr, title="LogReg (baseline) - Loss")
print(f"Valid (baseline) -> Acc: {hist_lr['valid_acc'][-1]:.3f} | "
      f"Prec: {hist_lr['valid_prec'][-1]:.3f} | Rec: {hist_lr['valid_rec'][-1]:.3f}")

train_sq_ds = TensorDataset(X_train_sq_t, y_train_bce)
valid_sq_ds = TensorDataset(X_valid_sq_t, y_valid_bce)

train_sq_loader = DataLoader(train_sq_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_sq_loader = DataLoader(valid_sq_ds, batch_size=BATCH_SIZE, shuffle=False)

X_dim_sq = X_train_sq_t.shape[1]
model_lr_sq, hist_lr_sq = train_model_logreg(
    X_dim_sq,
    train_sq_loader,
    valid_sq_loader,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    pos_weight=pos_weight,
    title="LogReg (x^2)"
)

plot_history(hist_lr_sq, title="LogReg (x^2) - Loss")
print(f"Valid (x^2) -> Acc: {hist_lr_sq['valid_acc'][-1]:.3f} | "
      f"Prec: {hist_lr_sq['valid_prec'][-1]:.3f} | Rec: {hist_lr_sq['valid_rec'][-1]:.3f}")

import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class MLP(nn.Module):
    def __init__(self, in_features, hidden=[32, 32], out_features=1, activation="relu",
                 batchnorm=False, dropout=0.0):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "leakyrelu": lambda: nn.LeakyReLU(0.1),
            "tanh": nn.Tanh,
        }
        act = acts[activation.lower()] if isinstance(activation, str) else activation

        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def eval_epoch_bce(model, loader, criterion):
    model.eval()
    total_loss, total_n = 0.0, 0
    logits_list, ys_list = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        logits_list.append(logits.cpu())
        ys_list.append(yb.cpu())
    avg_loss = total_loss / max(total_n, 1)
    logits_all = torch.cat(logits_list, dim=0).numpy().ravel()
    y_all = torch.cat(ys_list, dim=0).numpy().ravel()
    y_pred = (1/(1+np.exp(-logits_all)) >= 0.5).astype(int)
    acc = accuracy_score(y_all, y_pred)
    prec = precision_score(y_all, y_pred, zero_division=0)
    rec = recall_score(y_all, y_pred, zero_division=0)
    return avg_loss, {"accuracy": acc, "precision": prec, "recall": rec, "logits": logits_all, "y": y_all}

@torch.no_grad()
def eval_epoch_ce(model, loader, criterion):
    model.eval()
    total_loss, total_n = 0.0, 0
    logits_list, ys_list = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        logits_list.append(logits.cpu())
        ys_list.append(yb.cpu())
    avg_loss = total_loss / max(total_n, 1)
    logits_all = torch.cat(logits_list, dim=0)
    y_all = torch.cat(ys_list, dim=0).numpy().ravel()
    y_pred = torch.argmax(logits_all, dim=1).numpy().ravel()
    acc = accuracy_score(y_all, y_pred)
    prec = precision_score(y_all, y_pred, zero_division=0)
    rec = recall_score(y_all, y_pred, zero_division=0)
    return avg_loss, {"accuracy": acc, "precision": prec, "recall": rec,
                      "logits": logits_all.numpy(), "y": y_all, "y_pred": y_pred}

def plot_history(history, title="Training Curves"):
    epochs = range(1, len(history["train_loss"])+1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title); plt.legend(); plt.grid(True)
    plt.show()

def train_model_mlp_bce(
    X_dim,
    train_loader,
    valid_loader,
    hidden=[32,32],
    activation="relu",
    batchnorm=False,
    dropout=0.0,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    pos_weight=None,
    title="MLP (BCE)"
):
    model = MLP(X_dim, hidden=hidden, out_features=1, activation=activation,
                batchnorm=batchnorm, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "valid_loss": [], "valid_acc": [], "valid_prec": [], "valid_rec": []}

    for ep in range(1, epochs+1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * yb.size(0); n += yb.size(0)
        train_loss = running / max(n, 1)

        valid_loss, metrics = eval_epoch_bce(model, valid_loader, criterion)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(metrics["accuracy"])
        history["valid_prec"].append(metrics["precision"])
        history["valid_rec"].append(metrics["recall"])

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[{title}] Ep {ep:02d}/{epochs} | "
                  f"train={train_loss:.4f} | valid={valid_loss:.4f} | "
                  f"acc={metrics['accuracy']:.3f} | prec={metrics['precision']:.3f} | rec={metrics['recall']:.3f}")

    return model, history

def train_model_mlp_ce(
    X_dim,
    train_loader,
    valid_loader,
    hidden=[32,32],
    activation="relu",
    batchnorm=False,
    dropout=0.0,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    class_weights=None,
    title="MLP (CE)"
):
    model = MLP(X_dim, hidden=hidden, out_features=2, activation=activation,
                batchnorm=batchnorm, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "valid_loss": [], "valid_acc": [], "valid_prec": [], "valid_rec": []}

    for ep in range(1, epochs+1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * yb.size(0); n += yb.size(0)
        train_loss = running / max(n, 1)

        valid_loss, metrics = eval_epoch_ce(model, valid_loader, criterion)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(metrics["accuracy"])
        history["valid_prec"].append(metrics["precision"])
        history["valid_rec"].append(metrics["recall"])

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[{title}] Ep {ep:02d}/{epochs} | "
                  f"train={train_loss:.4f} | valid={valid_loss:.4f} | "
                  f"acc={metrics['accuracy']:.3f} | prec={metrics['precision']:.3f} | rec={metrics['recall']:.3f}")

    return model, history

y_train_ce = torch.tensor(y_train.values, dtype=torch.long)
y_valid_ce = torch.tensor(y_valid.values, dtype=torch.long)

train_ce_ds = TensorDataset(X_train_t, y_train_ce)
valid_ce_ds = TensorDataset(X_valid_t, y_valid_ce)

train_ce_loader = DataLoader(train_ce_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_ce_loader = DataLoader(valid_ce_ds, batch_size=BATCH_SIZE, shuffle=False)

neg = (y_train.values == 0).sum()
pos = (y_train.values == 1).sum()
pos_weight = torch.tensor([neg / max(pos,1)], dtype=torch.float32).to(device)
ce_class_weights = torch.tensor([1.0/neg, 1.0/max(pos,1)], dtype=torch.float32).to(device)

X_dim = X_train_t.shape[1]
mlp_v1, hist_mlp_v1 = train_model_mlp_bce(
    X_dim,
    train_loader,
    valid_loader,
    hidden=[32,32],
    activation="relu",
    batchnorm=False,
    dropout=0.0,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    pos_weight=pos_weight,
    title="MLP v1 (BCE)"
)
plot_history(hist_mlp_v1, "MLP v1 (BCE) - Loss")
print(f"Valid (MLP v1) -> Acc: {hist_mlp_v1['valid_acc'][-1]:.3f} | "
      f"Prec: {hist_mlp_v1['valid_prec'][-1]:.3f} | Rec: {hist_mlp_v1['valid_rec'][-1]:.3f}")

mlp_v2, hist_mlp_v2 = train_model_mlp_ce(
    X_dim,
    train_ce_loader,
    valid_ce_loader,
    hidden=[32,32],
    activation="relu",
    batchnorm=False,
    dropout=0.0,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    class_weights=ce_class_weights,
    title="MLP v2 (CE)"
)
plot_history(hist_mlp_v2, "MLP v2 (CE) - Loss")
print(f"Valid (MLP v2) -> Acc: {hist_mlp_v2['valid_acc'][-1]:.3f} | "
      f"Prec: {hist_mlp_v2['valid_prec'][-1]:.3f} | Rec: {hist_mlp_v2['valid_rec'][-1]:.3f}")

mlp_t1, hist_mlp_t1 = train_model_mlp_bce(
    X_dim, train_loader, valid_loader,
    hidden=[64, 32],
    activation="relu",
    batchnorm=False,
    dropout=0.20,
    epochs=25, lr=1e-3, weight_decay=1e-4,
    pos_weight=pos_weight,
    title="MLP (tuning #1: 64-32 + dropout0.2)"
)
plot_history(hist_mlp_t1, "MLP tuning #1 - Loss")
print(f"Valid (Tuning #1) -> Acc: {hist_mlp_t1['valid_acc'][-1]:.3f} | "
      f"Prec: {hist_mlp_t1['valid_prec'][-1]:.3f} | Rec: {hist_mlp_t1['valid_rec'][-1]:.3f}")

mlp_t2, hist_mlp_t2 = train_model_mlp_bce(
    X_dim, train_loader, valid_loader,
    hidden=[32, 32],
    activation="leakyrelu",
    batchnorm=True,
    dropout=0.0,
    epochs=25, lr=1e-3, weight_decay=1e-4,
    pos_weight=pos_weight,
    title="MLP (tuning #2: BN + LeakyReLU)"
)
plot_history(hist_mlp_t2, "MLP tuning #2 - Loss")
print(f"Valid (Tuning #2) -> Acc: {hist_mlp_t2['valid_acc'][-1]:.3f} | "
      f"Prec: {hist_mlp_t2['valid_prec'][-1]:.3f} | Rec: {hist_mlp_t2['valid_rec'][-1]:.3f}")

def show_examples_bce(model, X_valid_t, y_valid_bce, k=5, title="BCE Examples"):
    model.eval()
    with torch.no_grad():
        logits = model(X_valid_t.to(device)).cpu().numpy().ravel()
        probs = 1/(1+np.exp(-logits))
    y_true = y_valid_bce.numpy().ravel().astype(int)
    idx = np.random.choice(len(y_true), size=min(k, len(y_true)), replace=False)
    print(f"\n{title}")
    for i in idx:
        print(f"idx={i:03d} | prob_survive={probs[i]:.3f} | pred={int(probs[i]>=0.5)} | true={y_true[i]}")

def show_examples_ce(model, X_valid_t, y_valid_ce, k=5, title="CE Examples"):
    model.eval()
    with torch.no_grad():
        logits = model(X_valid_t.to(device)).cpu().numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_true = y_valid_ce.numpy().ravel().astype(int)
    idx = np.random.choice(len(y_true), size=min(k, len(y_true)), replace=False)
    print(f"\n{title}")
    for i in idx:
        print(f"idx={i:03d} | prob_survive={probs[i,1]:.3f} | pred={probs[i].argmax()} | true={y_true[i]}")

show_examples_bce(mlp_v1, X_valid_t, y_valid_bce, k=5, title="MLP v1 (BCE) - 5 ejemplos")
show_examples_ce(mlp_v2, X_valid_t, y_valid_ce, k=5, title="MLP v2 (CE) - 5 ejemplos")
show_examples_bce(mlp_t1, X_valid_t, y_valid_bce, k=5, title="MLP Tuning #1 (BCE) - 5 ejemplos")
show_examples_bce(mlp_t2, X_valid_t, y_valid_bce, k=5, title="MLP Tuning #2 (BCE) - 5 ejemplos")

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def bce_probs_and_labels(model, loader):
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy().ravel()
        p = 1/(1+np.exp(-logits))
        probs.append(p)
        ys.append(yb.numpy().ravel())
    y = np.concatenate(ys).astype(int)
    p = np.concatenate(probs)
    return p, y

@torch.no_grad()
def ce_probs_and_labels(model, loader):
    model.eval()
    ys, probs1 = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
        probs1.append(p)
        ys.append(yb.numpy().ravel())
    y = np.concatenate(ys).astype(int)
    p1 = np.concatenate(probs1)
    return p1, y

def metrics_from_probs(y_true, prob_pos, use_argmax=False):
    if use_argmax:
        y_pred = (prob_pos >= 0.5).astype(int)
    else:
        y_pred = (prob_pos >= 0.5).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, prob_pos),
    }, y_pred

def save_conf_mat(y_true, y_pred, title, fname_png):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    plt.figure()
    disp.plot(colorbar=False)
    plt.title(title)
    plt.savefig(os.path.join(FIG_DIR, fname_png), bbox_inches="tight", dpi=160)
    plt.close()

def save_history_png(history, title, fname_png):
    eps = range(1, len(history["train_loss"])+1)
    plt.figure()
    plt.plot(eps, history["train_loss"], label="Train Loss")
    plt.plot(eps, history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, fname_png), bbox_inches="tight", dpi=160)
    plt.close()

save_history_png(hist_lr,       "LogReg (baseline) - Loss", "loss_logreg_baseline.png")
save_history_png(hist_lr_sq,    "LogReg (x^2) - Loss",      "loss_logreg_x2.png")
save_history_png(hist_mlp_v1,   "MLP v1 (BCE) - Loss",      "loss_mlp_v1.png")
save_history_png(hist_mlp_v2,   "MLP v2 (CE) - Loss",       "loss_mlp_v2.png")
save_history_png(hist_mlp_t1,   "MLP tuning #1 - Loss",     "loss_mlp_t1.png")
save_history_png(hist_mlp_t2,   "MLP tuning #2 - Loss",     "loss_mlp_t2.png")

rows = []

p, y = bce_probs_and_labels(model_lr, valid_loader)
m, yhat = metrics_from_probs(y, p)
save_conf_mat(y, yhat, "LogReg (baseline) - Confusion Matrix", "cm_logreg_baseline.png")
rows.append({
    "model": "LogReg (baseline)",
    "params": num_params(model_lr),
    **{k: round(v, 4) for k, v in m.items()}
})

p, y = bce_probs_and_labels(model_lr_sq, valid_loader)
m, yhat = metrics_from_probs(y, p)
save_conf_mat(y, yhat, "LogReg (x^2) - Confusion Matrix", "cm_logreg_x2.png")
rows.append({
    "model": "LogReg (x^2)",
    "params": num_params(model_lr_sq),
    **{k: round(v, 4) for k, v in m.items()}
})

p, y = bce_probs_and_labels(mlp_v1, valid_loader)
m, yhat = metrics_from_probs(y, p)
save_conf_mat(y, yhat, "MLP v1 (BCE) - Confusion Matrix", "cm_mlp_v1.png")
rows.append({
    "model": "MLP v1 (BCE)",
    "params": num_params(mlp_v1),
    **{k: round(v, 4) for k, v in m.items()}
})

p1, y = ce_probs_and_labels(mlp_v2, valid_ce_loader)
m, yhat = metrics_from_probs(y, p1, use_argmax=True)
save_conf_mat(y, yhat, "MLP v2 (CE) - Confusion Matrix", "cm_mlp_v2.png")
rows.append({
    "model": "MLP v2 (CE)",
    "params": num_params(mlp_v2),
    **{k: round(v, 4) for k, v in m.items()}
})

p, y = bce_probs_and_labels(mlp_t1, valid_loader)
m, yhat = metrics_from_probs(y, p)
save_conf_mat(y, yhat, "MLP tuning #1 (BCE) - Confusion Matrix", "cm_mlp_t1.png")
rows.append({
    "model": "MLP tuning #1 (BCE)",
    "params": num_params(mlp_t1),
    **{k: round(v, 4) for k, v in m.items()}
})

p, y = bce_probs_and_labels(mlp_t2, valid_loader)
m, yhat = metrics_from_probs(y, p)
save_conf_mat(y, yhat, "MLP tuning #2 (BCE) - Confusion Matrix", "cm_mlp_t2.png")
rows.append({
    "model": "MLP tuning #2 (BCE)",
    "params": num_params(mlp_t2),
    **{k: round(v, 4) for k, v in m.items()}
})

df_results = pd.DataFrame(rows).sort_values(["accuracy","roc_auc","f1"], ascending=False)
print("\nResumen de métricas (valid):")
print(df_results.to_string(index=False))

csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nCSV guardado en: {csv_path}")

print(f"Figuras guardadas en: {FIG_DIR}")
