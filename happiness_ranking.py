# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 20:56:53 2020

@author: Deepak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r".spyder-py3\Happiness_rank_2020.csv")
data.info()
data.head()

plt.figure(figsize=(10,10))
plt.bar(data["Country name"], data["Ladder score"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("score")
plt.show()

plt.bar(data["Country name"], data["Logged GDP per capita"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("Gdp per capita")
plt.show()

plt.figure(figsize=(10,10))
plt.bar(data["Country name"], data["Freedom to make life choices"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("Freedom to make life choices")
plt.show()

plt.figure(figsize=(10,10))
plt.bar(data["Country name"], data["Generosity"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("Generosity")
plt.show()

plt.figure(figsize=(10,10))
plt.bar(data["Country name"], data["Healthy life expectancy"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("Health life expectation")
plt.show()

plt.figure(figsize=(10,10))
plt.bar(data["Country name"], data["Perceptions of corruption"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("corruption")
plt.show()

plt.figure(figsize=(10,10))
plt.bar(data["Country name"], data["Social support"])
plt.title("Happiness Ranking among Countries")
plt.xlabel("Country")
plt.ylabel("social support")
plt.show()
