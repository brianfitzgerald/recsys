## Recommmender system playground

https://github.com/HarshdeepGupta/recommender_pytorch

```
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
```

Multitask paper:
https://daiwk.github.io/assets/youtube-multitask.pdf

Single task paper:
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

Project outline:

- Estimate user rating for a piece of content
- Use embeddings generated to generate a KNN candidate selection phase
- Simulate user data, i.e. a set of watched videos, and estimate based on past watch history
- MoE gating for prediction of different genres
- DeepFM implementation

Dataset options:
- https://www.kaggle.com/competitions/avazu-ctr-prediction/data?select=train.gz
- https://www.kaggle.com/datasets/mrkmakr/criteo-dataset