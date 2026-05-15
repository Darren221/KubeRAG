from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: int = Field(
        ge=1, description="The [n] marker number from the answer text"
    )
    claim_span: str = Field(
        description="The sentence or phrase from the answer that this citation supports"
    )


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(description="The answer text with [n] citation markers inline")
    citations: list[Citation] = Field(
        description="One Citation per inline [n] marker in the text"
    )
    insufficient_context: bool = Field(
        description="True if the provided context did not contain enough information"
    )
