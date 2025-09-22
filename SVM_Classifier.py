from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import glob, os
from sklearn.metrics import classification_report, confusion_matrix

fs=44100

def main():
    train_paths_and_labels = [
        ("good", "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good"),
        ("broken", "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken"),
        ("heavy_load", "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload"),
    ]
    test_paths_and_labels = [
        ("good", "IDMT-ISA-ELECTRIC-ENGINE/test_cut/engine1_good"),
        ("broken", "IDMT-ISA-ELECTRIC-ENGINE/test_cut/engine2_broken"),
        ("heavy_load", "IDMT-ISA-ELECTRIC-ENGINE/test_cut/engine3_heavyload"),
    ]

    print("Loading data...")
    X_train, y_train = load_dataset(train_paths_and_labels)
    print(f"Loaded {len(y_train)} samples. Feature dim: {X_train.shape[1]}")
    print(y_train[0])  # good, broken, heavyload
    print(X_train[0])  # mean, var, std, rms, ptp

    X_test, y_test = load_dataset(test_paths_and_labels)
    print(f"Loaded {len(y_train)} samples. Feature dim: {X_train.shape[1]}")
    print(y_test[0])  # good, broken, heavyload
    print(X_test[0])  # mean, var, std, rms, ptp

    print("Training...")
    best_params_, best_score_, best_estimator_ = grid_search(X_train, y_train)
    print("Best params: ...")
    print(best_params_)
    print("Best score: ...")
    print(best_score_)
    print("Best estimator: ...")
    print(best_estimator_)

    # ---------- Evaluation on external test set ----------
    best_model = best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print("Test accuracy: %.4f" % test_acc)

    y_pred = best_model.predict(X_test)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

## load wav files
def get_wav(folder_path):
    import wave

    with wave.open(folder_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array

## Extract features
def features(folder_path):
    def safe_rms(x):
        if x.size == 0:
            return np.nan
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # replace bad values
        return np.sqrt(np.mean(x.astype(np.float64) ** 2))

    audio_array = get_wav(folder_path)
    if audio_array.size == 0 or len(audio_array) == 0:
        return np.nan

    mean_val = np.mean(audio_array)
    var_val = np.var(audio_array)
    std_val = np.std(audio_array)
    rms_val = safe_rms(audio_array)
    ptp_val = np.ptp(audio_array)  # max - min
    # 2) Zero-crossing rate (proxy for dominant freq)
    zcr_val = float(((audio_array[:-1] * audio_array[1:]) < 0).mean())

    # Next step , include frequency information
    return np.array([mean_val, rms_val, zcr_val ], dtype=float)

## Load datasets
# ---------- Dataset loader ----------
def load_dataset(paths_and_labels, pattern="*.wav"):
    X, y = [], []
    for label, path in paths_and_labels:
        for fp in glob.glob(os.path.join(path, pattern)):
            feats = features(fp)
            if feats is not None and np.all(np.isfinite(feats)) and feats is not np.nan:
                X.append(feats)
                y.append(label)
    return np.vstack(X), np.array(y)

## Grid Search
def grid_search(X_train, y_train):
    # ---------- Grid search SVM ----------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced"))
    ])

    grid = {
        "svc__C": [1, 10, 100],
        "svc__gamma": ["scale", 1e-3, 1e-4]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1)


    gs.fit(X_train, y_train)

    return gs.best_params_, gs.best_score_, gs.best_estimator_

