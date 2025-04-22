from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "MongoDB/airbnb_embeddings"
FILENAME = "train.csv"

import pandas as pd

df = pd.read_json("hf://datasets/MongoDB/airbnb_embeddings/airbnb_embeddings.json", lines=True)
df.to_csv("airbnb.csv", index=False)
print(df.columns)
