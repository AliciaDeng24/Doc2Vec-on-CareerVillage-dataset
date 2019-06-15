import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
%matplotlib inline

from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords
import random
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Lambda, ActivityRegularization
from keras.optimizers import Adam
