## Recommmender system playground

https://github.com/HarshdeepGupta/recommender_pytorch
specifically, https://github.com/HarshdeepGupta/recommender_pytorch/blob/master/MLP.py

https://github.com/hexiangnan/neural_collaborative_filtering

https://medium.com/towards-data-science/recommender-systems-using-deep-learning-in-pytorch-from-scratch-f661b8f391d7

paper - https://arxiv.org/abs/1708.05031

### Retrieving datasets

```
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
```

```
wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Toys_and_Games_5.json.gz --no-check-certificate
gzip -d Toys_and_Games_5.json.gz
```

Multitask paper:
https://daiwk.github.io/assets/youtube-multitask.pdf

Single task paper:
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

Dataset options:
- https://www.kaggle.com/competitions/avazu-ctr-prediction/data?select=train.gz
- https://www.kaggle.com/datasets/mrkmakr/criteo-dataset