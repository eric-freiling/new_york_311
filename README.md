# new_york_311
Analysis of the 311 service requests from NYC Open Data:

    https://nycopendata.socrata.com/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9

This projects assumes there is a path '../data/new_york_311' with the downloaded csv or a sample of the data.

## Challenge 1 is addressed in:
* Notebooks/EDA.ipynb

## Challenge 2 is addressed in :
* Notebooks/basic_random_forest.ipynb

## Extras are contained in: 
* Notebooks/basic_random_forest_time_elapsed.ipynb
    * A model trained on predicting the time between created date and closed date
        
* Notebooks/fastai.ipynb
    * A failed attempt to use fastai's default tabular framework
        
* Notebooks/visualzation.ipynb
    * A proof of concept of a geographical visualzation

## Timeine
* Challenge 1:
    * Dec 14th: spent a couple hours with 2 questions
* Challenge 2:
    * Dec: spent a whole day dealing with online GPU servers
        * spent a couple hours trying to get fastai working
        * Salamnder.ai is my usual and was failing. Not supproted that well
        * Tried a new provider, Paperspace, but was getting memory issues
        * Decided to abandon using all the data and took a sample
        * Got Kaggle's free kernels to run fastai on a sample of the data
    * Jan: Just short of a day
        * A half day on Random Forest predicting complaint type and time elapsed 
        * A couple hours on visualzation
