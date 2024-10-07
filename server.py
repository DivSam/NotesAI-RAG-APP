from pydantic import BaseModel
from fastapi import FastAPI


class Query(BaseModel):
    q: str
    user: str


app = FastAPI()


@app.post("/query/")
async def query(query: Query):
    return {"q": query.q, "user": query.user}
