# mel-reborn

## Results

### OLPEAT
| Language | Recall@1 | Recall@10 | Recall@100 |
|----------|----------|-----------|------------|
| Arabic   |   90.0   |    93.2   |     94.3   |
| German   |          |           |            |
| English  |          |           |            |
| Spanish  |          |           |            |
| Persian  |          |           |            |
| Japanese |          |           |            |
| Serbian  |          |           |            |
| Tamil    |          |           |            |
| Turkish  |          |           |            |



For OLPEAT explanation click [here](https://arxiv.org/pdf/2406.16892#section.6.4).

## History

This repository is based on my [thesis](https://arxiv.org/abs/2406.16892), however it implements plateau of different improvements which were not part of the original code.
Most noticably:
- We now support multilingual entity linking in the style of [Entity Linkin in 100 Languages](https://aclanthology.org/2020.emnlp-main.630/).
- The code is significantly faster and more memory efficient.

Other improvements are:
- Completly rewritten tokenization pipeline. The previous version was pretty much impossible to extend or change and also inefficient
- Support for different models 

Soon to be added:
- gin