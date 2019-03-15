# BERT

**update!**  I got no idea why this repo suddenly gets some public attention but thanks. I just made an update to make it better.

a simple yet complete implementation of the popular BERT model.

Experiments show this code could achieve close performance to that of Google.

## Advanced Features

- Distributed training
- Lazy file reader

## Requirement
- python==3.6
- torch==1.0.0

## Quick Guide

- This code is very simple, it should explain itself.

- Distributed training settings
  - --world_size #total number of gpus
  - --gpus #gpus on this machine
  - --MASTER_ADDR #master node IP
  - --MASTER_PORT #master node port
  - --start_rank # range from 0 to world_size-1, the index of the first gpu on this machine
  - --backend # 'nccl' or 'gloo', nccl is generally better but may not work on some machines

- Exemplar use of a trained model

  - See `example_use.sh` and `example_use.py` for more details.

  - Preprocessing Guide

    ```python
    from google_bert import BasicTokenizer
    tokenizer = BasicTokenizer()
    x = "BERT在多个自然语言处理任务中表现优越。"
    char_level_tokens = tokenizer.tokenize(x)
    
    # if processing at word level
    # We assume a word segmenter "word_segmenter" in hand
    word_level_tokens = word_segmenter.segment(x)
    #Note you may need to add speical tokens (e.g., [CLS], [SEP]) by yourself.
    ```
## Contact
[Deng Cai](https://jcyk.github.io)