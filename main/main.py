import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def save_talk(role, talk):
    with open("talk_log.txt", "a", encoding="utf-8") as file:
        file.write(f"{role}: {talk}\n\n")


start = time.time()
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16
)

model.to(device)

input_text = (
    "You are a professional who analyzes healthcare data to improve patient outcomes and optimize medical services.\n"
    "How can machine learning be used in hospitals to improve patient care and aid in disease prevention? Summarize it in a single paragraph."
)
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)
response = (
    tokenizer.decode(outputs[0], skip_special_tokens=True)
    .replace(input_text, "")
    .strip()
)

# 2차 번역
translator = pipeline("translation", model="facebook/nllb-200-distilled-1.3B")
translated_response = translator(response, src_lang="eng_Latn", tgt_lang="kor_Hang")[0][
    "translation_text"
]

with open("talk_log.txt", "a", encoding="utf-8") as file:
    file.write("-----" * 20 + "\nOne role in a concise sentence\n")

save_talk("user", input_text)
save_talk("Llama2 (English)", ".\n".join(response.split(". ")))
save_talk("Translated text (Korean)", ".\n".join(translated_response.split(". ")))
end = time.time()


print(f"\n User input:\n{input_text}")
print(f"\n Llama2 output (English):\n{response}")
print(f"\n Translated text (Korean):\n{translated_response}")
print(f"\n Time Taken >>>> {end - start:.5f} sec")
