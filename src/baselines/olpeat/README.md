# OLPEAT Overview
Scripts for One-Language-Pretrained-Embeddings-Alias-Table.

The calculation has two steps:

## Embedding

### DaMuEL
In `at_embeddings.py`. 
Produces the embeddings but saves them as toks,embs pairs.
Each toks is also embeded only once.
We cannot train on the result but we can use it as mapping later.

### Mewsli-9
Standard embeddings from `utils.embeddings.py`.
We don't care about duplicities because the data is small.
Thus, we just embed Mewsli-9 to emb(mention), qid dataset.

## OLPEAT recall
Currently in `find_recall.py`.
A training data is constructed from the DaMuEL and the previous step.
The idea is to take the underlying DaMuEL tokens and count ocurrencies of 
each tokenized result. This allows us to only put the top R most occurring
QIDs per the same array of tokens to the index.