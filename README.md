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

## Content-based filtering
We've implemented a content-based filtering approach for movie recommendations.

The main algorithm can be splitted to the following steps:
1. Take the dataset with movie information
2. Vectorize the data using TF-IDF approach
3. Calculate similarity matrix based on the previous results.
4. Take the training dataset of ratings.
4. Do the dot product of training dataset with similarity matrix.
5. Evaluate the results on test dataset.

For this approach we scrapped additional data about movies, but it didn't provide significant improvement.

In general such an approach performed better than the baseline, but not so significantly.


## Item-to-item/user-to-user collaborative filtering
In addition to other methods, we implemented item-to-item and user-to-user collaborative filtering approaches for movie recommendations.

These techniques demonstrated performance that was broadly consistent with each other. Both item-to-item and user-to-user collaborative
filtering methods produced lower RMSE compared to other models, indicating improved accuracy in predicting individual ratings. Despite the lower RMSE,
these approaches did not show significant improvements in ranking-based metrics such as MAP, MRR, and NDCG when compared to the baseline model. This suggests that while the predictions were more accurate on average, the ranking and relevance of the recommendations were not substantially enhanced. The collaborative filtering approaches are relatively straightforward to implement, as they primarily rely on the existing rating matrix and do not necessitate extensive additional data or feature engineering. This simplicity can be advantageous in scenarios where data is sparse or feature extraction is complex.

One of the notable limitations is the computational intensity associated with these methods. Constructing the similarity matrices and generating predictions becomes computationally expensive as the dataset size increases. Particularly for datasets with more than 1,000 samples, the time and resources required to compute all ratings and maintain the similarity matrix can be prohibitive.


## Multi-armed bandit approach 
We tried to implement a recommender system using multi-armed bandit (MAB) algorithms, specifically focusing on the LinUCB algorithm. The main goal was to create a recommendation system that balances exploration and exploitation to optimize movie recommendations for users.

The model's choice was done because of the training speed and its performance compared to naive non-contextual Epsilon Greedy and UCBBandit algorithms (we've tested them also). 
LinUCB algorithm implementation is based on the contextualbandits Python library. Overall approach was chosen for its ability to handle contextual information, which is crucial for personalized recommendations.

The metrics achieved are the next:
* LinUCB MAP: 0.19589655639717443
* LinUCB MRR: 0.2297836
* LinUCB NDCG: 0.16184333990828204

A MAP of indicates the mean precision of the algorithm when averaged over all queries. A value of  ~0.195 is poor and suggests that, on average, the precision of the recommended items is about 20%.
An MRR of 0.23 reflects low precision telling us that relevant items are, on average, ranked fairly low in the recommendation list (~8-9 place).
NDCG evaluates the quality of the ranking by considering the position of relevant items. An NDCG of 0.16 suggests that the overall ranking of relevant items is not effective

Despite many experiments conducted, the final metrics are not high and showing results similar to PageRank which works much faster compared to MAB approach. It suggestes that MAB approach is not suitable for our use case (little context and lack of powerful computational resources to analyze 'arms' for bandits). 


## A/B testing framework
#### A/B test goal:
Our goal is to choose the best recommendation algorithm out of set different models. The appropriate randomization unit for our use case is a user. We will test different algoritms work on all active users using our online platform.  

#### Assumptions about our online system characteristics:
* We have an online content platform with 3952 movies available and with more than 300 genres combinations.  
* We recommend movies on the main and on the search pages. We believe that the better movies are sorted, the better service we provide as a content platform.   
* Users can browse, rate, and receive personalized movie recommendations.
* We can track user interactions, including clicks, ratings, and time spent on movie pages.
* We have a big enough users base in order to conduct statistically significant A/B tests.
* We can randomly assign users to control and treatment groups.

#### Metrics selection:
The primary goal of our recommender system is to find relevant movies that users will enjoy. So, the main metric should reflect this goal and be connected with the recommender system quality. However, we are interested in overall users engagement level, so we need to add a conversion rate to the list. In result, the metrics list is:  
1. Mean average precision - measures the quality of the ranking of recommended items.
2. Conversion rate - the percentage of recommended movies that are watched or rated.

We don't want to include more metrics in the list because the more metrics we include, the harder to control test quality and make the final decision.

#### Statistical testing approach
We will use a two-tailed hypothesis test to compare the performance of recommender algorithms acroos two groups (control vs. treatment). Users in the control will receive recommendations from the base algorithm and the treatment group users will receive recommendations from another algorithm. 
The null hypothesis (H0): there is no significant difference in the chosen metric between the control and treatment groups. The alternative hypothesis (Ha): there is a significant difference in the chosen metric, favoring the treatment group. 

#### Experiment design
**Control/treatment split**
In order to achieve statistically significant results, the split will be done at random with the same population sizes in each test group (a 50/50 randomized split between control and treatment groups). However, out users base is not gender balanced according to the exploratory data analysis. Therefore, we need to control for gender balance in each group as well as occupation structure.  

**Sample size calculation based on the effect size, statistical power and significance**
We need a sufficient sample size to detect a meaningful difference between the groups with a desired level of confidence. We will use industry standard significance level (5%) and power of test (80%).
Also, we want to achieve minimum detectable difference in the primary metric. Empirical experience gives us +0.3 in the mean average precision and +5% to the conversion rate (our practical decision boundaries). 

The example formula of the sample size considering comments above is the next:
“n\=(dZ1−α/2​+Z1−β​​)2×2×σ2”

,where 𝑍1−𝛼/2 and 𝑍1−𝛽 are the critical values for the significance level and power, respectively, and 𝜎 is the standard deviation of the primary metric.

**Experiment scenario key points**
Each test should inlude the next points in its scenario: 
1) Ramp-up plan:
- Each test will start with dozens of users (5% of the target population). The number of users participated will be gradually increased to the target population size in order to limit potential effects from bugs / technical problems connected with the test (if they occur).

2) Account on seasonality: 
- The test should not be conducted during the holidays or other specific period when users naturally sticks to specific films higher than to others (e.g. Harry Potter movies during winter holidays). 

3) Sanity check: 
- We need to check if no time lags or other technical problems occur in test groups.


#### Decision-making methodology
1. We run the experiment for a predetermined duration and until the required sample size is reached.
2. Conduct a statistical test: if p-value < 0.05 for primary metrics, reject the null hypothesis.
3. We consider the practical significance of the observed differences described above (+0.3 in the mean average precision and +5% to the conversion rate).
4. Make a decision based on statistical significance, practical importance and financial resources needed to fully implement new recommendation model in production.


#### A/B test example 1:
**Testing a new collaborative filtering algorithm**
Objective: Compare a new collaborative filtering algorithm (treatment) against the current content-based algorithm (control).

The metric:
1. Conversion rate to a view or rating. 

Desired improvement in CTR: +5% increase. Significance level: 5%. Power of test: 80%.

Sample size (example, can be calculated thorugh fact metrics by using the formula above): 8,350 per group. So, the total sample size: 16,700 users.

Experiment Duration: assuming 5,000 daily active users, we run the experiment for at least 4 days.

After running the experiment, we observe:
1. Control Group: 840 views out of 8,350 recommendations (10.06% CTR)
2. Treatment Group: 980 views out of 8,350 recommendations (11.74% CTR)

Performing a two-sample z-test for proportions:
z = (p1 - p0) / sqrt(p * (1-p) * (2/n))
z ≈ 3.89

The p-value for this z-score is < 0.0001, which is less than our significance level of 5%.
The decision: we reject the null hypothesis. The new collaborative filtering algorithm shows a statistically significant improvement in CTR. However, the practical importance described above is 5% of increase in CTR. Additionally, we will need to gain new portion of investments in order to implement a collaborative filtering algorithm on the full production scale. So, the overall effect is nor significant enough and we decided to reject the new algorithm.


#### A/B test example 2:
**Testing a New Personalized Ranking Algorithm**
Objective: Compare a new personalized ranking algorithm (treatment) against the current ranking algorithm (control).

The metric: Mean average precision (MAP).

Desired improvement in MAP: 15% relative increase. Significance level: 5%. Power of test: 90% (increased from 0.8 to ensure higher confidence in detecting smaller effects).

Let's the sample size awill be the same as in the previous example due to the same assumptions. 

Duration: assuming 5,000 daily active users, we run the experiment for at least 4 days. However, to account for potential variability and to capture longer-term effects, we'll extend the experiment to 2 weeks.


After running the experiment for 2 weeks, we observe:
1. Control group: MAP: μ0 = 0.281, σ0 = 0.079
2. Treatment roup: MAP: μ1 = 0.301, σ1 = 0.082

Performing a two-sample t-test:
t = (μ1 - μ0) / sqrt(σ1^2/n1 + σ0^2/n0) ≈ 7.53
p-value < 0.0001
The improvement in MAP is statistically significant.
Relative improvement: (0.301 - 0.281) / 0.281 * 100 = 7.12%


Final decision: based on the statistically significant and practically important improvements in our primary metric (MAP), we decide to implement the new personalized ranking algorithm.









