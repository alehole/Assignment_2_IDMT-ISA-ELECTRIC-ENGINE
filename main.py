import RF_Classifier
import data_exp,SVM_Classifier
import features as ft

if __name__ == '__main__':
    #data_exploration()
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
    X_train, y_train = ft.load_dataset(train_paths_and_labels)
    X_test, y_test = ft.load_dataset(test_paths_and_labels)
    print(f"Loaded {len(y_train)} samples. Feature dim: {X_train.shape[1]}")


    #SVM_Classifier.main(X_train, y_train,X_test, y_test)
    RF_Classifier.main(X_train, y_train,X_test, y_test)
