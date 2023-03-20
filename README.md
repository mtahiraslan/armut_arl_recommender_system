
# Armut Association Rule Based Recommender System

![img](https://thingsolver.com/wp-content/uploads/Screen-Shot-2021-01-13-at-2.04.50-PM.png)

## What is Recommendation System?

A recommendation system is an artificial intelligence or AI algorithm, usually associated with machine learning, that uses Big Data to suggest or recommend additional products to consumers. These can be based on various criteria, including past purchases, search history, demographic information, and other factors. Recommender systems are highly useful as they help users discover products and services they might otherwise have not found on their own.

## Business Problem

Armut, which is Turkey's largest online service platform, brings together service providers and those who want to receive services. It enables easy access to services such as cleaning, renovation, and transportation with just a few touches on a computer or smartphone. An association rule learning-based product recommendation system is desired to be created using the dataset containing users who received services and the categories of services they received.

![img2](https://cdn.armut.com/images/og_image_url.jpg)

## Features of Dataset

- Total Number of Variables: 4
- Total Number of Rows: 162.523
- CSV File Size: 5MB

## The story of the dataset

The dataset consists of the services customers receive and the categories of these services. Date and time of each service received contains information.

| Variable Name | Description |
|----------------|----------------|
| **UserId** | Customer ID |
| **ServiceId** | They are anonymized services belonging to each category. (Example: Upholstery washing service under Cleaning category) A ServiceId can be found under different categories and represents different services under different categories. (Example: The service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while the service with CategoryId 2 and ServiceId 4 is furniture assembly) |
| **CategoryId** | They are anonymized categories. (Example: Cleaning, transportation, renovation category) |
| **CreateDate** | The date the service was purchased |

## Methods and libraries used in the project

- pandas, numpy, mlxtend.frequent_patterns, warnings
- apriori, association_rules

## Requirements.txt

- Please review the 'requirements.txt' file for required libraries.


