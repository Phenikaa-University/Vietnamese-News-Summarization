from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, pipeline, LlamaForCausalLM

def extractive_summary(txt, model, sentences=None, ratio=None):
    all_docs = txt.split("|||||")[:-1]
    for i, doc in enumerate(all_docs):
        doc = doc.replace("\n", " ")
        doc = " ".join(doc.split())
        all_docs[i] = doc
    results_extract_bert = []
    if sentences:
        for body in all_docs:
            results_extract_bert.append(model(body, num_sentences=5))
    elif ratio:
        for body in all_docs:
            results_extract_bert.append(model(body, ratio=ratio))
    # Back to string
    txt = "\n".join(results_extract_bert)
    return txt

def generate_summary(txt, model_path="LR-AI-Labs/vbd-llama2-7B-50b-chat", device="cuda"):
    SYS_PROMPT = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực giúp người dùng. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn. "\
    "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
    #     load_in_8bit=True
    )
    model.eval()
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

def calculate_rouge_scores(df):
    # Calculate ROUGE scores average for all dataset
    rouge1_f1 = 0
    rouge2_f1 = 0
    rougeL_f1 = 0
    rougeLsum_f1 = 0
    rouge1_precision = 0
    rouge2_precision = 0
    rougeL_precision = 0
    rougeLsum_precision = 0
    rouge1_recall = 0
    rouge2_recall = 0
    rougeL_recall = 0
    rougeLsum_recall = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    for i in range(len(df)):
        scores = scorer.score(df['summary'][i], df['abstractive_summary'][i])
        rouge1_f1 += scores['rouge1'].fmeasure
        rouge2_f1 += scores['rouge2'].fmeasure
        rougeL_f1 += scores['rougeL'].fmeasure
        rougeLsum_f1 += scores['rougeLsum'].fmeasure
        rouge1_precision += scores['rouge1'].precision
        rouge2_precision += scores['rouge2'].precision
        rougeL_precision += scores['rougeL'].precision
        rougeLsum_precision += scores['rougeLsum'].precision
        rouge1_recall += scores['rouge1'].recall
        rouge2_recall += scores['rouge2'].recall
        rougeL_recall += scores['rougeL'].recall
        rougeLsum_recall += scores['rougeLsum'].recall

    print("====== ROUGE-1 SCORES ======")
    print('ROUGE-1 F1:', rouge1_f1/len(df))
    print('ROUGE-1 Recall:', rouge1_recall/len(df))
    print('ROUGE-1 Precision:', rouge1_precision/len(df))
    print("====== ROUGE-2 SCORES ======")
    print('ROUGE-2 F1:', rouge2_f1/len(df))
    print('ROUGE-2 Recall:', rouge2_recall/len(df))
    print('ROUGE-2 Precision:', rouge2_precision/len(df))
    print("====== ROUGE-L SCORES ======")
    print('ROUGE-L F1:', rougeL_f1/len(df))
    print('ROUGE-L Precision:', rougeL_precision/len(df))
    print('ROUGE-L Recall:', rougeL_recall/len(df))
    print("====== ROUGE-Lsum SCORES ======")
    print('ROUGE-Lsum F1:', rougeLsum_f1/len(df))
    print('ROUGE-Lsum Precision:', rougeLsum_precision/len(df))
    print('ROUGE-Lsum Recall:', rougeLsum_recall/len(df))