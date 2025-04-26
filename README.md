# Ai_Malicious_Review_Checker
Thomas T Bakaysa, Jr
Final project for Ai enhanced security. Goal is to create a model that can determine whether a review is spam / fake.

## Datasets used
[Yelp dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data)
[Fake Reviews](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data)

## Overview:
1. pre-process the yelp dataset, discarding irrelevant categories while normalizing and encoding everything else
2. I will be using the business, user and review json files.
3. Because the yelp dataset does not contain target labels on whether the review is fake or not, I will be using the Fake Reviews data set
to train an initial model, which i will call the support model.
4. That support model will then be used to create pseudo labels on a subset of the the yelp dataset, which i will manually check to see how
accurate the predictions are.
5. When I'm happen the performance of the support model, I will use it to create psuedo labels for the entirety of the yelp dataset.
6. Train a new model on the yelp dataset, integrating the various features that I had cleaned earlier.
