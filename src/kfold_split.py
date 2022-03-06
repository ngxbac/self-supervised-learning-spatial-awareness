import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("/data/train.csv")
targets = df[["healthy", "multiple_diseases", "rust", "scab"]].values
targets = targets.argmax(axis=1)
df["target"] = targets
folds = np.zeros_like(targets)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2411)
for fold, (train_idx, valid_idx) in enumerate(kf.split(df, targets)):
    folds[valid_idx] = fold
df["fold"] = folds
df.to_csv("/data/train_kfold.csv", index=False)