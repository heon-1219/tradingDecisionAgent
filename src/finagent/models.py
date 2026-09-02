from pydantic import BaseModel


class AnswerResponse(BaseModel):
    """Exact response contract mandated by the competition (all five fields required)."""

    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str
