<!-- # Predicting Competitive Game Outcome Based on Jungler Statistics in League of Legends -->

by Lukas Fullner (lfullner@ucsd.edu) and Justin Lu (jzlu@ucsd.edu)

## Introduction: Framing the Problem

This study examines different stats related to the `Jungle` player in professional League of Legends matches, and predicts the victor of a game based on those stats.

In the game League of Legends, teams of 5 players simultaneously play in a square-shaped arena. Each team starts in their base on opposing corners, while their objective is to ultimately breach the other team's base. This arena is broken down into 3 "lanes" connecting the two bases: top, middle (mid), and bottom (bot), which describe the region of the arena it is located in. While 4 out of 5 players on a team will play in these lanes, the final player, called the "jungler", plays in the space between them, called the "jungle".

This role entails they harvest the monsters in the jungle for money and level-up experience, while occasionally visiting lanes to assist the player(s) there. To assist them with this, each jungler is also equipped with a special ability that helps them take down monsters more efficiently. In addition to regular monsters which a jungler can harvest on their own, there are also "epic monsters," which are harder to harvest monsters which grant a team-wide bonus. Since the junglers have a special ability, they are also responsible for securing these team-wide bonuses by getting the final strike on an epic monster. These epic monsters are known as: barons, heralds, elemental drakes/dragons, and elder drakes/dragons.

With everything in mind, we can acknowledge that junglers have many choices they can make that will influence how the game plays out due to their large area of play, many responsibilities, and potential to affect the lanes where other players are. As a result, many League of Legends players feel junglers and their jungle have an outsized impact on a game, with some believing it to be too much. Thus, this report will present a model that predicts a game's result using statistics related to a jungler.

To conclude, we will use the statistics related to the junglers to predict the winner of a professional League of Legends game.

> Our initial foray into this dataset can be found at [justinlu.ca/lol-meta-health](https://justinlu.ca/lol-meta-health), where we looked into the per-patch meta health of competitive League of Legends matches

---

## Baseline Model

The initial dataset contains 12 rows per game, representing data about the various players in the game (every 1st-10th row), as well as data on the whole team (every 11th and 12th row). Since we are only using stats related to the overall game as our datapoints, we need to compress the blocks of 12 rows into one row per game. We notate the two teams in the game as team 0 and team 1. The data is organized as follows:

> - `gameid`: The index is set to id of the game, not a feature but useful for organization
> - `team_X_barons`: The number of barons the team took
> - `team_X_heralds`: The number of heralds the team took
> - `team_X_elders`: The number of elder drakes/dragons the team took
> - `team_X_elementaldrakes`: The number of elemental drakes/dragons the team took
> - `first_baron`: Which team took the baron first
> - `first_dragon`: Which team took the first dragon
> - `first_herald`: Which team took the herald first
> - `team_X_monsterkills`: How many monsters the team killed
> - `team_X_minionkills`: How many minions the team killed (includes monster kills)
> - `result`: Which team ultimately won

Note we have the data split between team 0 and 1 while not actually accounting for who those teams are.  Our objective here is to train a classifier that could be given the stats of two arbitrary teams and predict who wins, and so we abstract the teams into binary for classifier performance.


<div class="table-wrapper" markdown="block">

|                              |   team_0_barons |   team_1_barons |   team_0_elders |   team_1_elders |   team_0_elementaldrakes |   team_1_elementaldrakes |   first_baron |   first_dragon |   first_herald |   team_0_heralds |   team_1_heralds |   team_0_minionkills |   team_1_minionkills |   team_0_monsterkills |   team_1_monsterkills |   result |
|:-----------------------------|----------------:|----------------:|----------------:|----------------:|-------------------------:|-------------------------:|--------------:|---------------:|---------------:|-----------------:|-----------------:|---------------------:|---------------------:|----------------------:|----------------------:|---------:|
| ('ESPORTSTMNT01_3286841', 0) |               0 |               1 |               0 |               0 |                        0 |                        3 |             1 |              1 |              1 |                1 |                1 |                  631 |                  721 |                   132 |                   171 |        1 |
| ('ESPORTSTMNT01_3286882', 0) |               2 |               0 |               0 |               0 |                        4 |                        2 |             0 |              0 |              1 |                1 |                1 |                  899 |                  895 |                   237 |                   190 |        0 |
| ('ESPORTSTMNT01_3286910', 0) |               0 |               1 |               0 |               0 |                        1 |                        4 |             1 |              1 |              1 |                1 |                1 |                  992 |                 1027 |                   226 |                   231 |        1 |
| ('ESPORTSTMNT01_3286935', 0) |               0 |               2 |               0 |               0 |                        3 |                        2 |             1 |              0 |              1 |                0 |                2 |                  828 |                  818 |                   136 |                   238 |        1 |
| ('ESPORTSTMNT01_3287811', 0) |               1 |               0 |               0 |               0 |                        2 |                        2 |             0 |              1 |              0 |                2 |                0 |                  846 |                  861 |                   195 |                   187 |        0 |

</div>

Here we are going to create our baseline classifier, using a random forest classifier with hand-picked hyperparameters (`max_depth`=10, `n_estimators`=100). The final model will use a grid search with cross-validation to optimize them, but here we are just trying to get an estimate on how good this style of classifier is. We will be using the first 4 features (8 columns) of the cleaned dataset, `team_X_barons`, `team_X_heralds`, `team_X_elders`, `team_X_elementaldrakes` as our features. Each is an int representing the number of that epic monster that was taken, and our assumption is that the more of the epic monsters a team takes the more likely it is to win. 

Using this method, we achieved an accuracy of: **90.30%**, which is a fairly high score. Note that while the pipeline has no transformers, this is since a lot of the feature engineering we planned to do already occured in the data cleaning step, where we took the entire dataframe and compressed it into a format that we can regress on.  Next we will add the rest of the extracted values, and perform a hyperparameter search.  

---

## Final Model

Here we add more features, as more data about the game should allow us to make better classifications. We will present the features to the classifier directly, as the scale of a feature in a random forest model does not impact the behavior of the model. We will, however, engineer a polynomial feature by non-linearly combining several of the features we are already using. The goal here is to create an extra feature that captures the interplay of the different game events we are studying. For example, capturing any of the epic monsters yields a buff to the team that captured it, ex. the dragon providing player stat buffs and the heralds giving you a new, strong helper. These will have an impact on the other features, and could lead to a snowballing effect that will determine the result we are predicting.

Thus, we use a pipeline and `SplineTransformer` to combine a 2nd-degree polynomial for each stat, regardless of team and, using the baseline model, perform a GridSearchCV to find the best hyperparameters, which was `{'higher-order__unifier__degree': 2, 'higher-order__unifier__n_knots': 5}`. Then, we fed this into a random forest classifier, which we also used a GridSearchCV to find the optimal hyperparameters for. These came out to be `{'forest__max_depth': 14, 'forest__n_estimators': 3000}`, which means our final model achieved a **91.96%** testing accuracy, just under a 2% increase from the baseline.


---

## Fairness Analysis

Since shorter games by nature take less time, we feel that our model might have had an easier time predicting the winner of a game for shorter games. Therefore, we will perform a fairness analysis by seeing whether our model had equally good predictions for games of shorter and longer lengths each. We define the threshold for what makes a game shorter or longer we define as the median game length, which in this case, comes out to be 1839 seconds, or 30 minutes and 39 seconds.

We will judge how good our model is by using the difference of group means. In this case, this means calculating the average precision of our model for shorter games and subtracting the average precision for longer games to get the signed difference.

For this analysis, we will perform a permutation test using a 0.05 significance level. Our hypotheses for this are:

> - `null`: Our model is equally fair; the mean precision of shorter games = mean precision of longer games
> - `alternate`: Our model is unfair; the mean precision of shorter games > mean precision of longer games

<iframe src="assets/fairness.html" width=800 height=600 frameBorder=0></iframe>

Using 1000 simulations, we get the above distribution along with a p-value of **0.0**. This is far below our significance level of 0.05, leading us to the conclusion of rejecting the null hypothesis that our model was equally fair. We can see the observed average precision difference is far higher than expected if it was fair, which could potentially mean that our model does better at predicting the outcome of shorter games compared to longer games.