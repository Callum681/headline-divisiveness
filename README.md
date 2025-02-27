# Headline Divisiveness Prediction  

This project uses **BERT embeddings** and **XGBoost regression** to analyze news headlines and predict their divisiveness score. It processes a dataset of headlines, extracts embeddings using a pre-trained BERT model, trains an XGBoost model, and applies it to new data for predictions.

The model was used as part of a project exploring the effect of the homophily of a social network on the divisiveness of the news which spreads on that network. 

The specific use case for this project was to take headlines which have been manually rated in how divisive they are on a scale from 1 to 10, and use this data to impute the divisiveness scores for unlabelled headlines. However, the tool is entirely generalizable to any use case where the user is interested in projection of sentence-level text data into a scalar representation. e.g. if the user has access to a series of headlines and their manually-labelled numerical political leaning scores, the model could also be used to generate a political-leaning classifier. 

A gradient boosted tree is used for its flexibility. The hyperparameters used are the result of tuning by the author, but retuning may be advisable for different use cases.
