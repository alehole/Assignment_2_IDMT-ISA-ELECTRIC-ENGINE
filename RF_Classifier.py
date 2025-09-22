from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def main(X_train, y_train,X_test, y_test):


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

def grid_search(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    grid = {
        "n_estimators": [100, 200, 500],  # number of trees
        "max_depth": [None, 10, 20, 30],  # tree depth (None = unlimited)
        "min_samples_split": [2, 5, 10],  # min samples needed to split a node
        "min_samples_leaf": [1, 2, 4],  # min samples per leaf
        "max_features": ["sqrt", "log2"],  # features considered per split
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs_rf = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    print("Training RandomForest...")
    gs_rf.fit(X_train, y_train)

    return gs_rf.best_params_, gs_rf.best_score_, gs_rf.best_estimator_