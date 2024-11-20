import uvicorn
from fastapi import FastAPI
from RAG.api.api import router as RAGAPI_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.get("/")
async def health():
    return {"Message": "API is healthy"}

app.include_router(RAGAPI_router,prefix="/api/v1")
