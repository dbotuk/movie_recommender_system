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

## PageRank implementation results
We've implemented Personalized Page Rank algorithm to test its capabilities in solving recommendation problem. 

The conclusion is that PageRank is not suitable for this problem as a standalone solution. This is because PageRank primarily considers the link structure (i.e., the user-movie interactions) without understanding the content or features of the movies or users. Our dataset contains information about movies (genres) and users (demographics, occupation) which are not leveraged by the algorithm.

Another problem is that the dataset is sparse, meaning most users have rated only a small subset of movies (this fact was fixed during the exploratory data analysis). PageRank relies on the link structure, and sparse data can lead to a poor representation of user preferences and weak connections in the bipartite graph.

The last but not the least problem is that PageRank tends to favor nodes with more connections (by its design), which in the context of movies can lead to popular movies being recommended repeatedly, even if they are not relevant to a specific user's preferences.

# Content-based filtering
We've implemented a content-based filtering approach for movie recommendations.

The main algorithm can be splitted to the following steps:
1. Take the dataset with movie information
2. Vectorize the data using TF-IDF approach
3. Calculate similarity matrix based on the previous results.
4. Take the training dataset of ratings.
4. Do the dot product of training dataset with similarity matrix.
5. Evaluate the results on test dataset.

For this approach we scrapped additional data about movies, but it didn't provide significant improvement.

In general such approach performed better than the baseline, but not so significantly.