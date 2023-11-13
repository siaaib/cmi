import pandas as pd
import yaml
from sklearn.model_selection import KFold

NFOLDS = 5
RANDOM_STATE = 42
df = pd.read_csv("/home/siavash/random/data/train_events.csv")
ids = df.series_id.unique()
skf = KFold(n_splits=NFOLDS, random_state=RANDOM_STATE, shuffle=True)
for fold_num, (train_index, valid_index) in enumerate(skf.split(ids), start=0):
    train, valid = ids[train_index], ids[valid_index]
    fold_data = {
        "train_series_ids": list(
            set([str(x) for x in train.tolist()])
            - set(
                [
                    "31011ade7c0a",
                    "a596ad0b82aa",
                    "390b487231ce",
                    "c7b1283bb7eb",
                    "2fc653ca75c7",
                    "a3e59c2ce3f6",
                    "0f9e60a8e56d",
                    "89c7daa72eee",
                    "c5d08fc3e040",
                    "e11b9d69f856",
                ]
            )
        ),
        "valid_series_ids": [str(x) for x in valid.tolist()],
    }

    yaml_filename = f"/home/siavash/random/dsseg/run/conf/split/fold_{fold_num}.yaml"

    with open(yaml_filename, "w") as file:
        yaml.dump(fold_data, file)

    print(f"Saved fold {fold_num} to {yaml_filename}")
