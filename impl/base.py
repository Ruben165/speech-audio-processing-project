"""Shared helpers for the speech & audio projects (paths, data, features, training, plots).

Layout (relative to the repository root):
    data/project_<x>/case_<y>/            downloaded / cached data
    result/project_<x>/case_<y>/          plots and metrics
    result/project_<x>/case_<y>/model/    best model (or a reference file for pretrained ones)
"""

import io
import json
import os
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audioread.exceptions import NoBackendError
from soundfile import LibsndfileError
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
joblib.parallel_config(
    backend="threading"
)  # loky worker processes crash on Windows + numba/librosa

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


# ----------------------------------------------------------------------------- paths
class Case:
    """Resolves the data/result/model folders of one case and saves its artifacts."""

    def __init__(self, project, case):
        self.name = f"project_{project}/case_{case}"
        self.data = ROOT / "data" / self.name
        self.result = ROOT / "result" / self.name
        self.model = self.result / "model"
        for d in (self.data, self.model):
            d.mkdir(parents=True, exist_ok=True)
        # Keep downloads inside the case data folder. HuggingFace reads HF_HOME only at import time,
        # so HF calls should also pass cache_dir=case.data explicitly.
        os.environ["KAGGLEHUB_CACHE"] = str(self.data / "kaggle_cache")
        os.environ["HF_HOME"] = str(self.data / "hf_cache")
        os.environ["TORCH_HOME"] = str(self.data / "torch_cache")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KAGGLEHUB_VERBOSITY"] = "error"

    def save_fig(self, name, show=True):
        plt.tight_layout()
        plt.savefig(self.result / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.show() if show else plt.close()

    def save_json(self, name, obj, model=False):
        path = (self.model if model else self.result) / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path


# ----------------------------------------------------------------------------- data access
def download_gdrive_zip(file_id, dest, keep_zip=False):
    """Download a Google Drive zip into `dest` and extract it (skipped if `dest` is not empty)."""
    import gdown

    dest = Path(dest)
    done = dest / ".downloaded"
    if done.exists():
        return dest
    zip_path = dest / "download.zip"
    gdown.download(id=file_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    if not keep_zip:
        zip_path.unlink()
    done.touch()
    return dest


def _retry_429(fn, tries=6):
    """Call fn(); on a Kaggle 'HTTP 429 Too Many Requests' wait 2, 4, 8... seconds and try again."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" not in str(e) or i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def fetch_kaggle_files(handle, files, dest, workers=4):
    """Download single files of a public Kaggle dataset (no API key needed). Returns local paths.

    The dataset version is looked up once and reused: an unversioned handle makes every file
    download call the Kaggle metadata API again, which gets rate limited (HTTP 429)."""
    import kagglehub

    dest = Path(dest)
    todo = [f for f in files if not (dest / f).exists()]
    bar = tqdm(
        total=len(files), desc="download"
    )  # created first so it keeps the real stderr
    bar.update(len(files) - len(todo))
    # kagglehub prints a progress bar per file; hide those
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        if todo and "/versions/" not in handle:
            _retry_429(
                lambda: kagglehub.dataset_download(
                    handle, path=todo[0], output_dir=str(dest)
                )
            )
            bar.update(1)
            todo = todo[1:]
            owner, name = handle.split("/")[:2]
            versions = [
                p.name
                for p in (dest / ".complete" / "datasets" / owner / name).glob("*")
                if p.name.isdigit()
            ]
            if versions:
                handle = f"{handle}/versions/{max(versions, key=int)}"

        def one(f):
            _retry_429(
                lambda: kagglehub.dataset_download(handle, path=f, output_dir=str(dest))
            )
            bar.update(1)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
    bar.close()
    return [dest / f for f in files]


def load_audio(src, sr=22050):
    """Load audio from a path or raw bytes as mono float32 at `sr`. Falls back to PyAV (webm/opus)."""
    import librosa
    import soundfile as sf

    try:
        if isinstance(src, (bytes, bytearray)):
            y, orig = sf.read(io.BytesIO(src), dtype="float32", always_2d=True)
            y = (
                librosa.resample(y.mean(1), orig_sr=orig, target_sr=sr)
                if orig != sr
                else y.mean(1)
            )
        else:
            y, _ = librosa.load(src, sr=sr)
        return y
    except (LibsndfileError, NoBackendError, Exception):
        import av

        with av.open(
            io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src)
        ) as c:
            rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
            chunks = [
                f.to_ndarray().ravel()
                for fr in c.decode(audio=0)
                for f in rs.resample(fr)
            ]

        return np.concatenate(chunks).astype("float32")


# ----------------------------------------------------------------------------- features
def clip_features(y, sr, n_mfcc=13):
    """Clip-level handcrafted features (mean/std statistics) as a flat dict."""
    import librosa
    import pywt
    from scipy.stats import kurtosis, skew

    f = {}
    for name, v in {
        "rms": librosa.feature.rms(y=y),
        "zcr": librosa.feature.zero_crossing_rate(y),
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "flatness": librosa.feature.spectral_flatness(y=y),
    }.items():
        f[f"{name}_mean"], f[f"{name}_std"] = v.mean(), v.std()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i, (m, s) in enumerate(zip(mfcc.mean(1), mfcc.std(1)), 1):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_std"] = m, s

    f0 = librosa.yin(y, fmin=65, fmax=2000, sr=sr)
    f["f0_mean"], f["f0_std"] = f0.mean(), f0.std()

    e = np.array([np.sum(c**2) for c in pywt.wavedec(y, "db4", level=5)])
    p = e / (e.sum() + 1e-12)
    f["wavelet_entropy"] = -np.sum(p * np.log(p + 1e-12))
    f["skew"], f["kurtosis"] = skew(y), kurtosis(y)

    return f


def build_features(sources, sr=22050, augment=None, augment_mask=None, **kw):
    """Feature DataFrame for a list of paths/bytes (parallel). `augment(y, sr) -> [y2, ...]` adds augmented
    copies (only where `augment_mask[i]` is True, if given). The `source` column maps each row to its
    original index, so it can be used as CV group to avoid train/validation leakage."""
    from joblib import Parallel, delayed

    def one(i, src):
        y = load_audio(src, sr)
        extra = (
            augment(y, sr)
            if augment and (augment_mask is None or augment_mask[i])
            else []
        )
        return [(i, clip_features(v, sr, **kw)) for v in [y] + extra]

    # Threads: worker processes crashed on Windows with numba-compiled librosa code.
    jobs = Parallel(n_jobs=-1, prefer="threads", return_as="generator")(
        delayed(one)(i, s) for i, s in enumerate(sources)
    )

    rows = [r for out in tqdm(jobs, total=len(sources), desc="features") for r in out]
    df = pd.DataFrame([r[1] for r in rows])
    df["source"] = [r[0] for r in rows]
    return df


def augment_clip(y, sr, seed=SEED, n_pitch=2):
    """Deterministic waveform augmentations: pitch shift, time stretch, noise, gain."""
    import librosa

    rng = np.random.default_rng(seed)
    out = [
        librosa.effects.pitch_shift(y, sr=sr, n_steps=float(s))
        for s in rng.choice([-3, -2, -1, 1, 2, 3], n_pitch, replace=False)
    ]
    out.append(librosa.effects.time_stretch(y, rate=float(rng.uniform(0.85, 1.15))))
    out.append(
        y + rng.normal(0, 0.005 * (np.abs(y).max() + 1e-9), y.shape).astype("float32")
    )
    out.append(y * float(rng.uniform(0.5, 1.5)))
    return out


# ----------------------------------------------------------------------------- classic ML
def model_zoo():
    """(estimator, param_grid) pairs; grids are small and valid for current scikit-learn."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [0.1, 1, 10]},
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=SEED),
            {"C": [1, 10, 100], "gamma": ["scale", 0.01]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 3]},
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=SEED, n_jobs=-1
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, n_jobs=-1, verbosity=0),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.2],
            },
        ),
    }


def plot_confusion(y_true, y_pred, labels, title, ax=None):
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", colorbar=False, ax=ax
    )
    (ax or plt.gca()).set_title(title)


def compare_models(
    case,
    X_train,
    y_train,
    X_test,
    y_test,
    labels,
    groups=None,
    models=None,
    scoring="f1_macro",
):
    """Tune every model with grouped CV on train, score on test, save the best pipeline to case.model.

    Best = highest CV score; ties (within 0.005) go to the faster predictor.
    Returns (results DataFrame, best fitted pipeline).
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedGroupKFold,
        StratifiedKFold,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cv = (
        StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        if groups is not None
        else StratifiedKFold(5, shuffle=True, random_state=SEED)
    )
    rows, fitted = [], {}
    for name, (est, grid) in (models or model_zoo()).items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
        search = GridSearchCV(
            pipe,
            {f"clf__{k}": v for k, v in grid.items()},
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        search.fit(X_train, y_train, groups=groups)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = search.predict(X_test)
        ms = (time.perf_counter() - t0) / len(X_test) * 1000
        fitted[name] = search.best_estimator_
        rows.append(
            {
                "model": name,
                "cv_score": search.best_score_,
                "test_acc": accuracy_score(y_test, pred),
                "test_bal_acc": balanced_accuracy_score(y_test, pred),
                "test_f1_macro": f1_score(y_test, pred, average="macro"),
                "fit_s": fit_s,
                "predict_ms": ms,
                "params": search.best_params_,
            }
        )

    res = pd.DataFrame(rows).set_index("model")
    top = res[res.cv_score >= res.cv_score.max() - 0.005]
    best = top.predict_ms.idxmin()
    res["best"] = res.index == best

    _, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    res[["cv_score", "test_acc", "test_f1_macro"]].plot.bar(
        ax=ax[0], rot=30, ylim=(0, 1)
    )
    ax[0].set_title("Model comparison")
    plot_confusion(
        y_test, fitted[best].predict(X_test), labels, f"{best} (test)", ax[1]
    )
    case.save_fig("model_comparison")

    pred = fitted[best].predict(X_test)
    print(classification_report(y_test, pred, target_names=labels))
    res.drop(columns="params").to_csv(case.result / "metrics.csv")
    joblib.dump(
        {
            "pipeline": fitted[best],
            "features": list(X_train.columns),
            "labels": list(labels),
            "name": best,
        },
        case.model / "model.joblib",
    )

    case.save_json(
        "best_model",
        {
            "name": best,
            "params": res.loc[best, "params"],
            **res.loc[best].drop(["params", "best"]).to_dict(),
        },
        model=True,
    )
    return res.drop(columns="params"), fitted[best]


def predict_clip(case, path, sr=22050):
    """Inference with the model saved by `compare_models`: returns (label, {label: probability})."""
    bundle = joblib.load(case.model / "model.joblib")
    x = pd.DataFrame([clip_features(load_audio(path, sr), sr)])[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(x)[0]
    labels = bundle["labels"]
    return labels[int(proba.argmax())], {
        l: round(float(p), 3) for l, p in zip(labels, proba)
    }


# ----------------------------------------------------------------------------- deep learning
def train_classifier(
    model,
    train_loader,
    val_loader,
    case=None,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-2,
    patience=5,
    device=None,
    verbose=True,
):
    """Generic PyTorch loop: AdamW + ReduceLROnPlateau + early stopping on val loss.
    Restores the best weights, returns (model, history DataFrame)."""
    import copy

    import torch
    import torch.nn.functional as F

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad, hist = float("inf"), None, 0, []

    def run(loader, train):
        model.train(train)
        loss_sum = correct = n = 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = F.cross_entropy(out, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                loss_sum += loss.item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                n += len(y)
        return loss_sum / n, correct / n

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tl, ta = run(train_loader, True)
        vl, va = run(val_loader, False)
        sched.step(vl)
        hist.append(
            {
                "epoch": ep,
                "train_loss": tl,
                "val_loss": vl,
                "train_acc": ta,
                "val_acc": va,
                "sec": time.perf_counter() - t0,
            }
        )

        if verbose:
            print(
                f"epoch {ep:02d} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f} | {hist[-1]['sec']:.0f}s"
            )

        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def plot_history(hist, case, name="training_curves"):
    _, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    hist.plot(x="epoch", y=["train_loss", "val_loss"], ax=ax[0], title="Loss")
    hist.plot(x="epoch", y=["train_acc", "val_acc"], ax=ax[1], title="Accuracy")
    case.save_fig(name)


def predict_torch(model, x, device=None):
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x.to(device)), 1).cpu().numpy()
