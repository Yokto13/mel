# mel-reborn

## Results

### OLPEAT
| Language | Recall@1 | Recall@10 | Recall@100 |
|----------|----------|-----------|------------|
| Arabic   |   90.0   |   93.2    |    94.3    |
| German   |   88.9   |   93.2    |    94.2    |
| English  |   80.2   |   88.9    |    91.3    |
| Spanish  |   84.5   |   91.1    |    92.4    |
| Persian  |   82.8   |   89.2    |    91.2    |
| Japanese |   85.3   |   93.2    |    94.4    |
| Serbian  |   91.2   |   95.3    |    96.3    |
| Tamil    |   91.3   |   95.5    |    97.2    |
| Turkish  |   86.2   |   93.7    |    94.5    |



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