# Tokenization

Scripts for data tokenization.

`tokenize_mentions.sh` tokenizes for OLPEAT, `tokenize_context.sh` for finetuning.
Both scripts are simple and only thing they do is that they run the tokenization for both the damuel and mewsli.

If you need to change the model rewrite the name of the model config that is currently hard coded in the scripts.
Additionally, it is highly advisable to check the config corresponding to the tokenization to make sure that it contains sensible values. 