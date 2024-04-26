import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import accelerate
import bitsandbytes
from peft import PeftModel
from getpass import getpass
import os 

os.environ['HF_HOME'] = '/content/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/content/transformers_cache'
os.environ['HF_DATASETS_CACHE'] = '/content/datasets_cache'
os.environ['HF_METRICS_CACHE'] = '/content/metrics_cache'
os.environ['HUGGINGFACE_TOKEN'] = getpass('Enter your Hugging Face API token: ')

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

ft_model = PeftModel.from_pretrained(base_model,"axel-datos/mistral_finetuned")


def mistral_process_nl_query(nl_query):
    model_input = eval_tokenizer(nl_query, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        query_after_mistral = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=200, repetition_penalty=1.15)[0], skip_special_tokens=True)
        print(query_after_mistral)

    return query_after_mistral    
