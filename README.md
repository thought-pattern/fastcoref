# fastcoref (Tapestry Fork)

Fork of [shon-otmazgin/fastcoref](https://github.com/shon-otmazgin/fastcoref) v2.1.6, updated for compatibility with transformers 5.x.

## Changes from upstream

- Added `_tied_weights_keys` and `all_tied_weights_keys` class attributes to `FCorefModel` and `LingMessModel` for transformers 5.x compatibility (`mark_tied_weights_as_initialized` expects this attribute during `from_pretrained`)
- Added `CorefResult.get_resolved_text()` to produce text with pronouns replaced by their canonical antecedents

## Usage

```python
from fastcoref import FCoref

model = FCoref()

preds = model.predict(texts='John went to the store. He bought milk.')
preds.get_clusters()
# [['John', 'He']]

preds.get_resolved_text()
# 'John went to the store. John bought milk.'
```

For the more accurate LingMess model:

```python
from fastcoref import LingMessCoref

model = LingMessCoref()
```

## Citation

```
@inproceedings{Otmazgin2022FcorefFA,
  title={F-coref: Fast, Accurate and Easy to Use Coreference Resolution},
  author={Shon Otmazgin and Arie Cattan and Yoav Goldberg},
  booktitle={AACL},
  year={2022}
}
```

[F-coref: Fast, Accurate and Easy to Use Coreference Resolution](https://aclanthology.org/2022.aacl-demo.6) (Otmazgin et al., AACL-IJCNLP 2022)
