from typing import Any, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field


class CorrectnessVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = completely wrong; 1.0 = fully correct match with the golden answer.",
    )
    rationale: str = Field(description="One-sentence justification for the score.")


_CORRECTNESS_SYSTEM_PROMPT = """You are a strict grader for a Kubernetes Q&A system.

Given a question, a reference (golden) answer, and a candidate answer, score how well \
the candidate addresses the question in terms of factual content. Ignore style and \
wording differences; focus on whether the same facts are conveyed.

Scoring rubric:
- 1.0  : candidate is fully correct and covers what the golden covers.
- 0.7-0.9 : mostly correct, missing only minor points.
- 0.4-0.6 : partial — some key facts present, others missing or wrong.
- 0.1-0.3 : contradictory or mostly wrong.
- 0.0  : completely wrong, unrelated, or empty.

Be strict but fair. Provide a one-sentence rationale."""


class AnswerCorrectness:
    def __init__(
        self, *, client: AsyncOpenAI, model: str = "gpt-4o-mini"
    ) -> None:
        self.client = client
        self.model = model

    async def score(
        self,
        question: str,
        golden_answer: str,
        predicted_answer: str,
    ) -> CorrectnessVerdict:
        messages = [
            {"role": "system", "content": _CORRECTNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Golden answer: {golden_answer}\n\n"
                    f"Candidate answer: {predicted_answer}"
                ),
            },
        ]
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, messages),
            response_format=CorrectnessVerdict,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return CorrectnessVerdict(
                score=0.0, rationale="model returned no parseable verdict"
            )
        return parsed
