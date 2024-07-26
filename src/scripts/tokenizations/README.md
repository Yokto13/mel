# Tokenization

Scripts for data tokenization.

## at (alias tables)

These tokenize only labels/names (what exactly can be seen in data_processors.tokens).
Usefull for alias tables, like OLPEAT.

## finetuning

Tokenize not only mentions but also their surrounding context. 
The result looks like: [left context] [M] [mention] [M] [right context].
[M] is a special token denoting the mention.