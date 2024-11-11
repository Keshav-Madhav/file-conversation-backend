from fastapi import APIRouter, HTTPException, status
from RAG_Model import Rag_Model
from utils.Json_Objects import ChatBotData

router = APIRouter()
model = Rag_Model()
@router.post("/")
async def chat(input:ChatBotData):
    try:
        model.load_embedings(input.username, input.file_hash)
        model.qas()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Embeddings not found")
    l = len(input.answer_hist)
    history = [(input.question_hist[i], input.answer_hist[i]) for i in range(l)]
    return model.output(input.input_query, history)