# summrizing_token 
# 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

from fastapi import FastAPI, HTTPException
import torch
import transformers
from huggingface_hub import login

# Autenticación directamente con el token
login("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")


app = FastAPI()

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids(""),
        ]

    def get_response(self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

# Initialize the model
model = Llama3("meta-llama/Meta-Llama-3.1-8B-Instruct")

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/chat/")
async def chat(query: str, message_history: list = []):
    try:
        response, updated_history = model.get_response(query, message_history)
        return {"response": response, "updated_history": updated_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
