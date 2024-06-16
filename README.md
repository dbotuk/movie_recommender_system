# Movie Recommender System
It is the final Recommender Systems course project.

The **main goal** of this project is to apply techniques and topics covered in the course on one of the standard recommender systems dataset.

## Installation
Follow these steps to set up the project locally:

1. **Clone the repository** 

  Clone this repository to your local machine using the following command:
  ```sh
   git clone https://github.com/dbotuk/movie_recommender_system.git
```
2. **Navigate to the project directory**

  Change your working directory to the project directory:
  ```sh
  cd movie_recommender_system
  ```

3. **Create a virtual environment (optional but recommended)**

  Create and activate a virtual environment:
  ```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. **Install dependencies**

Install the required packages using the **requirements.txt** file:
```sh
pip install -r requirements.txt
```

5. **Open notebook**
After installing the dependencies, you can open any notebook and just enjoy =)

## Structure
This project has the following structure:
```
    |- artifacts
    |- data
    |- experiments
    |- scripts
    └- src
        └- models
```

Each version of used model with parameters, weights etc. inside *artifacts* directory. It can be used to reproduce the behavior.

The dataset is saved inside the *data* directory.

All the notebooks with experiments with its descriptions and results are placed inside *experiments* directory.
Each notebook named in the following form *<experiment_date>_<descriptive_name>*.

Inside *scripts* directory you can find some additional scripts, like additional info scrapping.

There are all the implementation code of models, metrics, utils, etc. inside the *src* directory.

## Dataset
For this project we are using MovieLens dataset, which contains 3 files *movies.dat* (contains information about each movie), *ratings.dat* (contains information about users' ratings of movies and its timestamps), *users.dat* (contains the information about users).

We also scrapped additional informaton about movie using IMDb API, which is saved inside *movies_extended.csv* file.

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
