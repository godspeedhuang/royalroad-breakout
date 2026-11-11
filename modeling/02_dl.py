"""Deep learning：在預訓練文字向量上訓練 MLP（PyTorch）。

文字編碼器（text-embedding-3-small）是凍結的，這裡訓練的是接在向量後面的分類網路。
選模流程和 01_ml.py 一樣：2017–2021 訓練、2022 選超參數與 epoch 數，
選好後用 2017–2022 重新訓練同樣的 epoch 數，在 2023 與 2024 打分數（3 個 seed 平均）。

輸出：out/preds_dl.parquet、out/selection_dl.json

ponytail: 編碼器沒有 fine-tune。要再往上，下一步是在前三章原文上 fine-tune 小型文字模型
（chapters.parquet 有原文），成本與時間都高一個量級，先確認文字有訊號再做。
"""
import itertools
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from features import OUT, block, load, splits

MAX_EPOCHS, PATIENCE, BATCH, SEEDS = 40, 5, 256, 3
GRID = list(itertools.product([0.2, 0.5], [1e-4, 1e-2]))  # (dropout, weight_decay)


def train(Z, y, drop, wd, epochs, seed=0, Zva=None, yva=None):
    """有驗證集：每個 epoch 算驗證 AUC，早停，回傳 (最佳 AUC, 最佳 epoch)。沒有驗證集：跑滿 epochs，回傳模型。"""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.ReLU(), nn.Dropout(drop),
                        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
    best, best_ep = 0.0, 0
    for ep in range(1, epochs + 1):
        net.train()
        for idx in torch.randperm(len(Zt)).split(BATCH):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(Zt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
        if Zva is not None:
            auc = roc_auc_score(yva, predict(net, Zva))
            if auc > best:
                best, best_ep = auc, ep
            elif ep - best_ep >= PATIENCE:
                break
    return (best, best_ep) if Zva is not None else net


@torch.no_grad()
def predict(net, Z):
    net.eval()
    return net(torch.from_numpy(Z)).squeeze(1).numpy()


if __name__ == "__main__":
    b, X = load()
    sp = splits(b)
    y, q = b.top10.values, b.quarter.values
    tr, va = sp["train"], sp["val"]
    trva, ev = tr | va, sp["test"] | sp["recent"]
    preds = b.loc[ev, ["fiction_id"]].copy()
    sel = {}
    for names in ["T", "SAT"]:
        raw = block(X, names)
        Z = StandardScaler().fit(raw[tr]).transform(raw).astype(np.float32)
        val = {p: train(Z[tr], y[tr], *p, MAX_EPOCHS, Zva=Z[va], yva=y[va]) for p in GRID}
        p = max(val, key=lambda k: val[k][0])
        epochs = val[p][1]
        Z = StandardScaler().fit(raw[trva]).transform(raw).astype(np.float32)
        preds[f"mlp_{names}"] = np.mean([predict(train(Z[trva], y[trva], *p, epochs, seed=s), Z[ev]) for s in range(SEEDS)], axis=0)
        sel[f"mlp_{names}"] = {"dropout": p[0], "weight_decay": p[1], "epochs": epochs, "val": {"auc": val[p][0]},
                               "val_auc_by_param": {str(k): [round(v[0], 4), v[1]] for k, v in val.items()}}
        print(f"mlp_{names:5s} dropout={p[0]} wd={p[1]} epochs={epochs} val AUC {val[p][0]:.3f}", flush=True)
    preds.to_parquet(OUT / "preds_dl.parquet", index=False)
    (OUT / "selection_dl.json").write_text(json.dumps(sel, indent=1))
