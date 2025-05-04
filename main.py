from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import lite_llm
import httpx

app = FastAPI(title="Chat Assistant API")

# Initialize LiteLLM model (assuming a default model or configuration)
# You may want to configure model selection dynamically
llm = lite_llm.LiteLLM()

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    model_name: Optional[str] = None  # Optional to specify LLM model

# In-memory store for conversations (for demo purposes)
conversations = {}

@app.post("/start_conversation")
async def start_conversation() -> dict:
    """
    Initialize a new conversation and return its ID.
    """
    import uuid
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = []
    return {"conversation_id": conv_id}

@app.post("/send_message")
async def send_message(
    conversation_id: str = Form(...),
    message: str = Form(...),
    model_name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> dict:
    """
    Send a message to the conversation, optionally with a file for context.
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Append user message to conversation history
    conversations[conversation_id].append({"role": "user", "content": message})
    
    # Prepare context
    context = ""
    for msg in conversations[conversation_id]:
        role = msg["role"]
        content = msg["content"]
        context += f"{role.capitalize()}: {content}\n"
    
    # If a file is uploaded, read its content and append
    if file:
        try:
            file_content = await file.read()
            context += f"File Content:\n{file_content.decode('utf-8')}\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")
    
    # Select model if specified
    selected_model = model_name if model_name else "default"
    
    # Generate response from LiteLLM
    try:
        response_text = llm.chat(prompt=context, model=selected_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation error: {str(e)}")
    
    # Append assistant response to conversation history
    conversations[conversation_id].append({"role": "assistant", "content": response_text})
    
    return {"response": response_text}

@app.get("/get_conversation/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    """
    Retrieve the full conversation history.
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conversations[conversation_id]}

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """
    Endpoint to upload a file for context or comparison.
    """
    try:
        content = await file.read()
        # For demo, just return size info
        return {"filename": file.filename, "size": len(content)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")