from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_name = "Open-Orca/Mistral-7B-OpenOrca"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)
print(model_inputs)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
text = tokenizer.batch_decode(generated_ids)
print(text)
