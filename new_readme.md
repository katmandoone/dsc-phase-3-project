# Predicting ESRB Ratings for Video Games

## Introduction

The purpose of this project is to predict game ratings for PlayStation4, Xbox One, and Nintendo Switch games. The game data is gathered directly from the esrb.org website. By looking at the descriptors given to a game, I hope to be able to use a classification model to predict the given rating. I believe that this model will help to determine whether game ratings are 'fair,' or if there is some degree of ambiguity when a consumer consults the rating tag before deciding on a game purchase.

![image](./images/esrb.gif)

> Looking at the image above, we can see an example of and ESRB rating and its descriptors.

## Obtaining Data

The data for this project was obtained from esrb.org using the Selenium library. I originally attempted to gather the data with BeautifulSoup as I was more familiar with that library, but I ran into some issues with the ESRB site requiring client-side loading of assets with Java. BeautifulSoup was not equipped to handle this, but Selenium allows for client-side web scraping.

The process for scraping the data is detailed in the data_gathering.ipynb notebook within this repository.

After scraping the data, everything was compiled into a pandas DataFrame

![image](./images/original_df_snip.png)

## Scrubbing the Data

There really wasn't much scrubbing involved with this data. I just needed to pull the values out of the lists in the descriptors column so that I could one-hot encode each descriptor as a categorical variable.

```
descriptors_ohe = pd.get_dummies(df.descriptors.apply(pd.Series).stack()).sum(level=0)
df_ohe = pd.concat([df.drop(columns=['descriptors']), descriptors_ohe], axis=1)
```

