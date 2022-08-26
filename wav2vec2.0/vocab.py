import pandas as pd
import json

"""
제공된 labels.csv를 labels.json으로 변환
빈도 수가 낮은 것은 삭제
"""

labels = pd.read_csv("labels.csv", index_col=0)
labels = labels.loc[(labels["freq"] > 50) | (labels["freq"] == 0)]
label_dict = {c:f for c, f in zip(labels["char"], labels["freq"])}
labels.to_csv("./labels1.csv")
with open("./labels.json", "w", encoding="UTF-8") as f:
    json.dump(label_dict, f, ensure_ascii=False)

