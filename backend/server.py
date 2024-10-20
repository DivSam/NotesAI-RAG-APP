from pydantic import BaseModel
from fastapi import FastAPI
from query_database import query_vector_database, query_model


class Query(BaseModel):
    user: str
    q: str


app = FastAPI()


@app.post("/query/")
async def query(query: Query):
    context = query_vector_database(query.q)
    response = query_model(context)
    return {"user": query.user, "response": response}
