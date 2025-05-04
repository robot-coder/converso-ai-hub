# README.md

# Web-Based Chat Assistant with FastAPI and LiteLLM

This project implements a web-based conversational AI assistant that allows users to select different large language models (LLMs), maintain conversation history, and upload files for context or comparison. The backend is built with Python's FastAPI, and it interacts with LiteLLM for language modeling. The frontend is a JavaScript UI (not included here) that communicates with the backend via REST API.

## Features

- Select and switch between multiple LLMs
- Maintain persistent conversation history
- Upload files to provide context or compare models
- Modular and error-resilient backend architecture

## Requirements

Python 3.8+  
[fastapi](https://fastapi.tiangolo.com/)  
[uvicorn](https://www.uvicorn.org/)  
[lite_llm](https://github.com/yourusername/lite_llm) (or your specific LiteLLM package)  
[pydantic](https://pydantic.dev/)  
[starlette](https://starlette.io/)  
[httpx](https://www.python-httpx.org/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/chat-assistant.git
cd chat-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `POST /chat/`  
  Send a message and receive a response. Payload includes conversation history, selected model, and optional files.

- `POST /upload/`  
  Upload files for context or comparison.

- `GET /models/`  
  List available LLMs.

## Usage

Integrate with your frontend JavaScript UI to send user messages, select models, upload files, and display conversation responses.

## License

MIT License

---

# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import lite_llm
import os

app = FastAPI()

# Initialize available models (assuming LiteLLM supports model listing)
AVAILABLE_MODELS = ["model_a", "model_b", "model_c"]
MODEL_INSTANCES: Dict[str, lite_llm.LiteLLM] = {}

# Load or initialize model instances
def get_model_instance(model_name: str) -> lite_llm.LiteLLM:
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Model not available")
    if model_name not in MODEL_INSTANCES:
        try:
            MODEL_INSTANCES[model_name] = lite_llm.LiteLLM(model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return MODEL_INSTANCES[model_name]

# Data models
class ChatRequest(BaseModel):
    messages: List[str]
    model_name: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    conversation_id: str

# In-memory storage for conversation history
conversation_histories: Dict[str, List[str]] = {}

@app.get("/models/")
async def list_models() -> List[str]:
    """
    Return the list of available LLM models.
    """
    return AVAILABLE_MODELS

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Handle chat messages, maintain conversation history, and generate responses.
    """
    model = get_model_instance(request.model_name)
    conv_id = request.conversation_id or "default"
    history = conversation_histories.get(conv_id, [])
    try:
        # Append user message
        history.extend(request.messages)
        # Generate response from model
        response_text = model.chat(history)
        # Append model response
        history.append(response_text)
        # Save updated history
        conversation_histories[conv_id] = history
        return ChatResponse(reply=response_text, conversation_id=conv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...), context: Optional[str] = Form(None)) -> Dict[str, str]:
    """
    Upload files to provide context or for comparison.
    """
    saved_files = {}
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files[file.filename] = file_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}: {str(e)}")
    return {"status": "success", "files": saved_files}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# requirements.txt

fastapi
uvicorn
lite_llm
pydantic
starlette
httpx