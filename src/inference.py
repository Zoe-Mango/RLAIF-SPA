import torch
import librosa
from transformers import AutoModel, AutoTokenizer
import json
import os
import logging
import csv
from peft import PeftModel
# Setup logging
logging.basicConfig(filename='/RLAIF_SPA/baseline/test_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
 
# Set GPU to the 1st card
torch.cuda.set_device("cuda:7") 
  
# Load the model and tokenizer
model_path = "/RLAIF_SPA/MiniCPM-o"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
lora_path = "/RLAIF_SPA/yq/tiny-grpo-main/checkpoint/step_7979" 
model = PeftModel.from_pretrained(model, lora_path)

model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Initialize TTS
model.init_tts()
model.tts.float()

# Function to generate speech from the given instruction
def generate_speech(sentence_data):
    sentence = sentence_data['text']
    sentence_id = sentence_data['id']
    
    instruction = f"Please express the sentence '{sentence}' in the voice of a middle-aged woman.Make sure the expression is clear."
    
    logging.info(f"Processing sentence ID {sentence_id} with instruction: {instruction}")

    msgs = [{'role': 'user', 'content': [instruction]}]
    
    output_path = f"/RLAIF_SPA/baseline/{sentence_id}.wav"
    
    try:
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=128,
            use_tts_template=True,
            generate_audio=True,
            temperature=0.3,
            output_audio_path=output_path,
        )
        logging.info(f"Successfully saved audio for sentence ID {sentence_id} at {output_path}")
    except Exception as e:
        logging.error(f"Failed to process sentence ID {sentence_id}: {e}")

def process_batch(dataset):
    for sentence_data in dataset:
        generate_speech(sentence_data)

dataset_path = '/RLAIF_SPA/baseline/test.jsonl'
dataset = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue 

        try:
            obj = json.loads(line)
        except Exception as e:
            logging.warning(f"Skipping malformed line {line_no}: {e} | {line}")
            continue

        if 'id' not in obj or 'text' not in obj:
            logging.warning(f"Skipping line {line_no}, missing id/text: {obj}")
            continue

        sent_id = str(obj['id']).strip()
        text = str(obj['text']).strip()
        dataset.append({'id': sent_id, 'text': text})

logging.info(f"Loaded {len(dataset)} samples from {dataset_path}")

process_batch(dataset)
