from pydantic import BaseModel

class InputData(BaseModel):
    file_type: str
    file_path: str
    file_hash: str
    username: str

class ChatBotData(BaseModel):
    file_hash: str
    username: str
    input_query: str
    question_hist: list[str]
    answer_hist: list[str]

