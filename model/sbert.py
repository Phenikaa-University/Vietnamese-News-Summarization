import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from summarizer.sbert import SBertSummarizer

from worker import extractive_summary
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, pipeline, LlamaForCausalLM


df_dataset = pd.read_csv("dataset/process/mews-mds.csv")

print("====== Extractive Summarization ======")
extractive_model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
df_test = df_dataset[:100]
df_test["extractive_summary"] = df_test["document"].apply(lambda x: extractive_summary(x, model=extractive_model, ratio=0.2))
df_test.to_csv("dataset/final/news-mds-extractive.csv", index=False)