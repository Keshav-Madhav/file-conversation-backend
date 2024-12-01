from fastapi import APIRouter, HTTPException, status
from utils.Json_Objects import InputData
from RAG_Model import Rag_Model
router = APIRouter()
model = Rag_Model()
@router.post("/")
async def textembed(input:InputData):
    try:
        model.load_embedings(input.username, input.file_hash)
        return status.HTTP_200_OK
    except:
        a = model.file_loader(input.file_path, input.file_type)
        if a is False:
            raise HTTPException(status_code=400, detail="File loading failed. No embedding performed.")
        b = model.text_splitter(a)
        if len(b) > 0:
            model.embed_data(b, input.username, input.file_hash)
            return status.HTTP_200_OK
        else:
            raise HTTPException(status_code=400, detail="Scanned Document. No embedding performed.")