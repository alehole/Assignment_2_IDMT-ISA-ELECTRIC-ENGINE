from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import features as ft

fs=44100

def main(X_train, y_train,X_test, y_test):
    print("Loading data...")
    #X_train, y_train = ft.load_dataset(train_paths_and_labels)
    print(f"Loaded {len(y_train)} samples. Feature dim: {X_train.shape[1]}")
    print(y_train[0])  # good, broken, heavyload
    print(X_train[0])  # mean, var, std, rms, ptp

   # X_test, y_test = ft.load_dataset(test_paths_and_labels)
    print(f"Loaded {len(y_train)} samples. Feature dim: {X_train.shape[1]}")
    print(y_test[0])  # good, broken, heavyload
    print(X_test[0])  # mean, var, std, rms, ptp

    print("Training:")
    best_params_, best_score_, best_estimator_ = grid_search(X_train, y_train)
    print("Best params:")
    print(best_params_)
    print("Best score:")
    print(best_score_)
    print("Best estimator:")
    print(best_estimator_)

    # ---------- Evaluation on external test set ----------
    best_model = best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print("Test accuracy: %.4f" % test_acc)

    y_pred = best_model.predict(X_test)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

## Grid Search
def grid_search(X_train, y_train):
    # ---------- Grid search SVM ----------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced"))
    ])

    grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale",1, 0.1,  1e-3, 1e-4]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    return gs.best_params_, gs.best_score_, gs.best_estimator_
