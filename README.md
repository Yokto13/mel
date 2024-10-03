# mel-reborn

## Results

### OLPEAT
| Language | Recall@1 | Recall@10 | Recall@100 |
|----------|----------|-----------|------------|
| Arabic   |   90.1   |   93.3    |    94.4    |
| German   |   89.1   |   93.3    |    94.3    |
| English  |   80.3   |   88.9    |    91.4    |
| Spanish  |   84.5   |   91.2    |    92.5    |
| Persian  |   83.0   |   89.3    |    91.4    |
| Japanese |   85.7   |   93.7    |    94.9    |
| Serbian  |   91.2   |   95.3    |    96.4    |
| Tamil    |   91.4   |   95.5    |    97.3    |
| Turkish  |   86.4   |   93.9    |    94.6    |



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