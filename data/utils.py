import os
import numpy as np
import pandas as pd
import re

# def read_txt(path):
#     with open(path, "r") as f:
#         return "||||".join(line.rstrip() for line in f if line.rstrip())

def read_txt_sum(path):
    with open(path, "r") as f:
        return f.read().strip()
    
def read_txt(path):
    with open(path, "r") as f:
        text = f.read()
        # Split the text by "Content:" and get the second part
        content = text.split("Content:", 1)[-1]
        return content.strip()
def create_dataframe(original_dir, summary_dir):
    data = []
    for i in range(1, 101):
        cluster = f"Cluster_{i:03d}"
        original_cluster_dir = os.path.join(original_dir, cluster, "original")
        summary_file = os.path.join(summary_dir, cluster, "1.gold.txt")
        
        document = " ||||| ".join(read_txt(os.path.join(original_cluster_dir, file)) for file in os.listdir(original_cluster_dir) if file.endswith(".txt"))
        summary = read_txt_sum(summary_file)
        
        data.append({"document": document, "summary": summary})
    
    return pd.DataFrame(data)

def save_data(df, path):
    df = df[ pd.notnull(df["document"]) ]
    df = df[ pd.notnull(df["summary"]) ]
    df["summary"] = df["summary"].apply(lambda x: utils_preprocess_text(x))
    df["document"] = df["document"].apply(lambda x: utils_preprocess_text(x))
    df.to_csv(path, index=False)

def utils_preprocess_text(txt, lst_regex=None, lower=True):
    '''
    Preprocess a string.
    :parameter
        :param txt: string - name of column containing text
        :param lst_resgex: list - list of regex to remove
        :param punkt: bool - if True removes punctuations and characters
        :param lower: bool - if True convert lowercase
        :param lst_stopwords: list - list of stopwords to remove
    :return
        cleaned text
    '''
    ## Regex (in case, before cleaning)
    if lst_regex is not None: 
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)

    ## Clean 
    ### Remove special characters
    txt = re.sub(r'[\@\#\$\%\^\&\*]+', '', str(txt))
    # ### separate sentences with '. '
    # txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    # ### remove punctuations and characters
    # txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
            
    ## Tokenize (convert from string to list)
    lst_txt = txt.split()
            
    ## Back to string
    txt = " ".join(lst_txt)
    return txt

def main():
    data_dir = "dataset/raw/News-mds"
    original_dir = os.path.join(data_dir, "original")
    summary_dir = os.path.join(data_dir, "summary")
    df = create_dataframe(original_dir, summary_dir)
    save_data(df, "dataset/process/mews-mds.csv")

if __name__ == "__main__":
    main()