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