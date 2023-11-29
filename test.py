import streamlit as st
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from fastai.data.all import *
from fastai.vision.all import *
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from openai import OpenAI
import pathlib
import os

posix_backup = pathlib.PosixPath
path = 'model.pkl'
try:
    pathlib.PosixPath = pathlib.WindowsPath
    learn = load_learner(path)
finally:
    pathlib.PosixPath = posix_backup

img = 'dog.jpg'
img2 = 'dog2.jpg'

print(learn)
output = learn.predict(img)[0]
output2 = learn.predict(img2)[0]
print(output)
print(output2)