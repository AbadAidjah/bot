import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

API_TOKEN = os.getenv("API_TOKEN", "CHANGE_ME")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Pre-trained model from HuggingFace

app = FastAPI()

# Request body model
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

# Lazy load model - downloads from HuggingFace on first request
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Downloading pre-trained model from HuggingFace...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        print("Model loaded!", flush=True)
    return tokenizer, model

@app.get("/health")
async def health():
    return {"status": "ready", "model": MODEL_NAME}

@app.post("/generate")
async def generate(request: GenerateRequest, token: str = Header(None)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tok, mod = load_model()
    
    # Format prompt for TinyLlama chat format
    chat_prompt = f"<|system|>\nYou are a helpful AI assistant knowledgeable about computer science and programming.</s>\n<|user|>\n{request.prompt}</s>\n<|assistant|>\n"
    
    inputs = tok(chat_prompt, return_tensors="pt")
    output = mod.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        temperature=request.temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tok.eos_token_id,
    )
    
    # Extract only the assistant's response
    full_response = tok.decode(output[0], skip_special_tokens=True)
    # Get text after the last "assistant" marker
    if "assistant" in full_response.lower():
        response = full_response.split("assistant")[-1].strip()
        # Clean up any remaining markers
        response = response.replace("</s>", "").replace("<|", "").strip()
    else:
        response = full_response
    
    return {"response": response}
