# Movie Recommender System
It is the final Recommender Systems course project.

The **main goal** of this project is to apply techniques and topics covered in the course on one of the standard recommender systems dataset.

## Installation


## Structure
This project has the following structure:
```
    |- artifacts
    |- data
    |- experiments
    └- src
        └- models
```

Each version of used model with parameters, weights etc. inside *artifacts* directory. It can be used to reproduce the behavior.

The dataset is saved inside the *data* directory.

All the notebooks with experiments with its descriptions and results are placed inside *experiments* directory.
Each notebook named in the following form *<experiment_date>_<descriptive_name>*.

There are all the implementation code of models, metrics, utils, etc. inside the *src* directory.

## Dataset


## Offline Evaluation Framework
To do the offline evaluation of the recommendations quality we will use our dataset as the "ground truth".

For evaluation purpose we use the combination of predictive and ranking metrics.
* *ML metrics:*
  * root mean squared error
* *predictive metrics*
  * mean average precision
* *ranking metrics*
  * mean reciprocal rank
  * normalized discounted cumulative gain

## Baseline model
We use a popularity-based recommender as a baseline. So, we recommend TOP-10 popular movies to all users.

It is quite easy from both the implementation and interpretation perspective. There is no problem to make the recommendations for new users.

However, such type of recommendations are deprived of the personalization. It also neglects the niche items.
