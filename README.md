# Disentangled Deep MHP

The implementation of our ICDM-21 submission "Disentangled Deep Multivariate Hawkes Process".

## Requirements

python == 3.6.2

numpy == 1.17.4

scipy == 1.3.1

torch == 1.6.0

## Dataset

Two commercial datasets are available [here](https://drive.google.com/drive/folders/1wK7nNyBZ9v3P2lm0w4gtbqPjk1H3NRrl?usp=sharing):

- ML-1M
- Taobao

The format of data is given as:

```
entity_id type_id timestamp
```

Two social datasets provided by previous works can be downloaded [here](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U):

- Retweets
- StackOverflow

## Running the Code Command


For running ML-1M:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py --model d2mhp --undebug --data_dir ml-1m --dis_k 4 --mi_loss 0.5
```

For running Taobao:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py --model d2mhp --undebug --data_dir taobao_item --dis_k 6 --mi_loss 0.5
```

For running Retweets:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py --model d2mhp --undebug --data_dir retweet --dis_k 2 --mi_loss 1
```


For running StackOverflow:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py --model d2mhp --undebug --data_dir so --dis_k 4 --mi_loss 1
```