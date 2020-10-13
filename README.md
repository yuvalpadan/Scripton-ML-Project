# Scripton-ML-Project
Characterizing aspects of violent content in films from the language used in scripts only, using various Machine Learning models.

## Description
Violent content in movies can affect the viewer's perception of the society.
The purpose of the project is to characterize aspects of violent content in films from the language used in scripts only.

## Data collection / data processing
Dataset - The data is 280 movies screenplays that have a viewing classification, determined by British Board of Film Classification (bbfc). 
I extracted the html/ txt scripts from https://www.dailyscript.com/movie.html

Features - bag of words, using the tf-idf transformation.
 
Labels - films' viewing classification, according to bbfc: 
-U (Suitable for all), PG(Parental guidance), 12+, 15+, 18+.

I used 280 screenplays for training and for testing (Leave-one-out cross-validation).

## Machine Learning Setup
- Preprocessing the data.
- Labeling and categorizing.
- Build KNN and linear regression models
- Find correlated words
- Testing results.
