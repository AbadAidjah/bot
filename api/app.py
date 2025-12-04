import os
from fastapi import FastAPI, Header, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM

API_TOKEN = os.getenv("API_TOKEN", "CHANGE_ME")
MODEL_PATH = "./model"

app = FastAPI()

# Lazy load model - only load when first request comes in
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
            raise HTTPException(status_code=503, detail="Model not ready. Training may still be in progress.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

@app.get("/health")
async def health():
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        return {"status": "ready"}
    return {"status": "model not ready"}

@app.post("/generate")
async def generate(prompt: str, token: str = Header(None)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tok, mod = load_model()
    inputs = tok(prompt, return_tensors="pt")
    output = mod.generate(
        **inputs,
        max_new_tokens=200,
        repetition_penalty=1.2,      # Penalize repeated tokens
        no_repeat_ngram_size=3,      # Don't repeat 3-grams
        temperature=0.7,             # Add some randomness
        do_sample=True,              # Enable sampling
        top_p=0.9,                   # Nucleus sampling
    )
    
    return {"response": tok.decode(output[0], skip_special_tokens=True)}
