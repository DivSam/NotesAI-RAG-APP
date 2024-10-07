from pydantic import BaseModel
from fastapi import FastAPI
from query_database import query_vector_database


class Query(BaseModel):
    user: str
    q: str


app = FastAPI()


@app.post("/query/")
async def query(query: Query):
    results = query_vector_database(query.q)
    return {"user": query.user, "results": results}
