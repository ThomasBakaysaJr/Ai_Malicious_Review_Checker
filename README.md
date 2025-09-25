# Ai_Malicious_Review_Checker
Thomas T Bakaysa, Jr
Final project for Ai enhanced security. Goal is to create a model that can determine whether a review is spam / fake by 
utilizing a bootstrap model to pseudo-label a review dataset, validate those and then attempt to create a more
powerful model that could reliably lable reviews as spam / fake based on a vareity of attributes.

The bootstrap model would use a less detailed dataset. Unfortunately the dataset used for the bootstrap model were reviews 
from a wide variety of sources, with fake reviews being computer generated. This lead to a bootstrap model that was 
trained on different data than Yelp reviews would normally contain, as Yelp is for business while the Fake dataset was
from a variety of online retail spaces.

In the end, I pivoted to trying to predict what star rating a customer would give based on their personal information,
previous review data, the current review message that they are posting. Ending with a LightGBM model that had about 70%
accuracy. Due to time and class constraints, this was the first and only functional model made from the Yelp Dataset.

## Datasets used
[Yelp dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data)
[Fake Reviews](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data)

## Overview for Original Idea:
1. pre-process the yelp dataset, discarding irrelevant categories while normalizing and encoding everything else
2. I will be using the business, user and review json files.
3. Because the yelp dataset does not contain target labels on whether the review is fake or not, I will be using the Fake Reviews data set
to train an initial model, which i will call the support model.
4. That support model will then be used to create pseudo labels on a subset of the the yelp dataset, which i will manually check to see how
accurate the predictions are.
5. When I'm happen the performance of the support model, I will use it to create psuedo labels for the entirety of the yelp dataset.
6. Train a new model on the yelp dataset, integrating the various features that I had cleaned earlier.


## Support Models (bootstrap model)
Model expects this as the structure of the hstack
X = hstack([x_text, rate_feature])
where x_text is cleaned text (no punctuation, lower case) that that has been vectorized.
where rate_feature is the ratings / 5.0 and shaped into a 2d arrary
