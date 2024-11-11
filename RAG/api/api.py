from fastapi import APIRouter
from RAG.api.v1 import textembeddings,chatbot

router = APIRouter()

router.include_router(chatbot.router, prefix="/chatbot", tags=["chatbot"])

router.include_router(textembeddings.router,prefix="/textembeddings", tags=["textembeddings"])