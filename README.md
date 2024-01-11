# Recommender Systems Challenge 2023

[![Open](https://img.shields.io/badge/Open-Kaggle-blue.svg)](https://www.kaggle.com/competitions/recommender-system-2023-challenge-polimi)

This repository contains the code for the Recommender System Challenge 2023, made on [Kaggle](https://www.kaggle.com/competitions/recommender-system-2023-challenge-polimi).

The challenge involves recommending to users 10 books that are most likely to interact with.

## Recommender

The final recommender used in the challenge was an hybrid of:
- SLIM Elastic Net
- RP3beta

obtained merging the two similarity matrices using a weighted sum.

We trained all the models using both kaggle and colab notebooks.

## Evaluation

The evaluation metric used for the competition is MAP@10
- Public leaderboard score: 0.14042 (18th/63)
- Private leaderboard score: 0.13984 (20th/63)

## Credits

For this challenge it was used the code taken from the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), that provides recommenders and utility code.

## Team

[Simone Scevaroli](https://github.com/simonescevaroli) & [Elisa Composta](https://github.com/elisacomposta)
