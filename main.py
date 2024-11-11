import uvicorn
from fastapi import FastAPI
from RAG.api.api import router as RAGAPI_router


app = FastAPI()

@app.get("/")
async def health():
    return {"Message": "API is healthy"}

app.include_router(RAGAPI_router,prefix="/api/v1")
