"""Answers a question from the retrieved chunks, as a Pydantic object with citations."""
from pydantic import BaseModel, Field

from rag.chunking import Chunk

MODEL = "claude-haiku-5-5"

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks)


def ask(question: str, chunks: list[Chunk], client=None) -> tuple[Answer, object]:
    """Returns the parsed Answer and the token usage. Needs ANTHROPIC_API_KEY in the environment."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{build_context(chunks)}\n\nשאלה: {question}"}],
        output_format=Answer,
    )
    return response.parsed_output, response.usage


def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    """TODO (step 5): return the citations that are NOT backed by the chunks.

    A citation is bad if its chunk_id is not one of the chunks we sent,
    or if its quote does not appear in that chunk's text.
    An empty list means every citation checks out.
    """
    def normalize(s: str) -> str:
        # Spaces, tabs and new lines all become one space, so a quote that crosses a line still matches.
        return " ".join(s.split())

    texts = {c.id: normalize(c.text) for c in chunks}
    bad = []
    for cit in answer.citations:
        quote = normalize(cit.quote)
        if cit.chunk_id not in texts or not quote or quote not in texts[cit.chunk_id]:
            bad.append(cit)
    return bad
