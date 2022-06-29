# PRNN-Pytorch

```bash
pip install -r requirements.txt
```

## Usage

```bash
Usage: run.py [OPTIONS]

Options:
  --task [yelp2|yelp5|toxic]  [default: yelp5]
  --b INTEGER                 [default: 128]
  --d INTEGER                 [default: 96]
  --num_layers INTEGER        [default: 2]
  --batch_size INTEGER        [default: 512]
  --dropout FLOAT             [default: 0.5]
  --lr FLOAT                  [default: 0.001]
  --rnn_type [LSTM|GRU|QRNN]  [default: GRU]
  --help                      Show this message and exit.
```

Datasets

-   yelp2(polarity): it will be downloaded w/ datasets(huggingface)
-   yelp5: [json file](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification?select=yelp_reviews.json) should be downloaded to into `data/`
-   toxic: [dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) should be downloaded and unzipped to into `data/`
-

### Example: Yelp Polarity

    python -W ignore run.py --task yelp2 --b 128 --d 64 --num-layers 4

## Credits

[tensorflow](https://github.com/tensorflow/models/tree/master/research/sequence_projection/prado)

Powered by [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [grid.ai](https://www.grid.ai/)
