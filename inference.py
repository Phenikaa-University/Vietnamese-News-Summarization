import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from summarizer.sbert import SBertSummarizer

from utils import extractive_summary
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, pipeline, LlamaForCausalLM


df_dataset = pd.read_csv("dataset/process/mews-mds.csv")

print("====== Extractive Summarization ======")
extractive_model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
df_test = df_dataset[:100]
df_test["extractive_summary"] = df_test["document"].apply(lambda x: extractive_summary(x, model=extractive_model, ratio=0.2))
df_test.to_csv("dataset/final/news-mds-extractive.csv", index=False)
print("====== Abstractive Summarization ======")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path="LR-AI-Labs/vbd-llama2-7B-50b-chat"
SYS_PROMPT = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực giúp người dùng. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn. "\
    "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
#     load_in_8bit=True
)
def generate_summary(txt):
    summary = []
    for chunk in tqdm(txt.split("\n")):
        input_prompt = f"{SYS_PROMPT} USER: Tóm tắt lại văn bản với ngữ cảnh sau: {chunk} ASSISTANT:"
        input_ids = tokenizer(input_prompt, return_tensors="pt")
        outputs = model.generate(
            inputs=input_ids["input_ids"].to(device),
            attention_mask=input_ids["attention_mask"].to(device),
            do_sample=True,
            temperature=0.7,
            top_k=50, 
            top_p=0.9,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[-1].strip()
        summary.append(response)
    return " ".join(summary)
print("====== Generating summaries ======")
# Abstractive summary
df_test["abstractive_summary"] = df_test["extractive_summary"].apply(lambda x: generate_summary(x))

print("====== Saving to file ======")
df_test.to_csv("dataset/final/vims_summary_100_2.csv", index=False)