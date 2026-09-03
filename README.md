# fastcoref (Tapestry Fork)

Inference-only fork of [shon-otmazgin/fastcoref](https://github.com/shon-otmazgin/fastcoref) v2.1.6, using the current Transformers 5 runtime. The package contains the model and inference utilities used by Tapestry; obsolete research training entrypoints and generated package metadata are not maintained as parallel source representations.

## Changes from upstream

- Added `CorefResult.get_resolved_text()` to produce text with pronouns replaced by their canonical antecedents

## Usage

```python
from fastcoref import FCoref
from spacy import load

nlp = load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
model = FCoref(nlp=nlp)

preds = model.predict(texts='John went to the store. He bought milk.')
preds.get_clusters()
# [['John', 'He']]

preds.get_resolved_text()
# 'John went to the store. John bought milk.'
```

For the more accurate LingMess model:

```python
from fastcoref import LingMessCoref
from spacy import load

nlp = load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
model = LingMessCoref(nlp=nlp)
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
