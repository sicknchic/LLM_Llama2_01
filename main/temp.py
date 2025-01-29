import time
import os
import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MarianMTModel,
    MarianTokenizer,
)

load_dotenv()
llama2_token = os.getenv("Llama2_token")
translation_token = os.getenv("Translation_token")

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

llama_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", token=llama2_token
)
llama_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    token=llama2_token,
)
llama_model.to(device)
llama_model.to(memory_format=torch.channels_last)

input_text = (
    "You are a professional who analyzes healthcare data to improve patient outcomes and optimize medical services. "
    "How can machine learning be used in hospitals to improve patient care and aid in disease prevention? Summarize it in a single paragraph.\n\n"
)
inputs = llama_tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

start = time.time()
outputs = llama_model.generate(**inputs, max_length=200)
# english_response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
english_response = llama_tokenizer.decode(
    outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
).strip()
print(f"First model output: {english_response}")

translation_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
translation_tokenizer = MarianTokenizer.from_pretrained(
    translation_model_name, token=translation_token, src_lang="en"
)
translation_model = MarianMTModel.from_pretrained(
    translation_model_name, token=translation_token
)
translation_model.to(device)
sentences = english_response.split(". ")
translated_sentences = []

for sentence in sentences:
    if not sentence.strip():
        continue

    translation_inputs = translation_tokenizer(
        sentence, return_tensors="pt", padding=True, src_lang="en"
    )
    translation_inputs = {
        key: value.to(device) for key, value in translation_inputs.items()
    }

    translated = translation_model.generate(**translation_inputs)
    korean_translation = translation_tokenizer.decode(
        translated[0], skip_special_tokens=True
    ).strip()
    translated_sentences.append(korean_translation)

final_translation = " ".join(translated_sentences)

print(f"\n Translated output:\n{final_translation}")

end = time.time()
print(f"\n Total time: {end - start:.5f} sec")
