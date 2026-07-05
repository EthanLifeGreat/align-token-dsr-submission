from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _BaseTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(..., min_length=1, description="Speech token sequence.")

    @model_validator(mode="after")
    def validate_tokens(self) -> "_BaseTokenRequest":
        if len(self.tokens) == 0:
            raise ValueError("`tokens` must not be empty.")
        return self


class Token2WavRequest(_BaseTokenRequest):
    prompt_wav_base64: str | None = Field(default=None, min_length=1)
    prompt_wav_path: str | None = Field(default=None, min_length=1)
    speed: float = Field(default=1.0, gt=0.0, le=3.0)

    @model_validator(mode="after")
    def validate_prompt_mode(self) -> "Token2WavRequest":
        if self.prompt_wav_base64 and self.prompt_wav_path:
            raise ValueError("Only one of `prompt_wav_base64` or `prompt_wav_path` may be set.")
        return self


class Token2WerRequest(_BaseTokenRequest):
    ground_truth_text: str = Field(..., min_length=1)
    prompt_wav_base64: str | None = Field(default=None, min_length=1)
    prompt_wav_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_prompt_mode(self) -> "Token2WerRequest":
        if self.prompt_wav_base64 and self.prompt_wav_path:
            raise ValueError("Only one of `prompt_wav_base64` or `prompt_wav_path` may be set.")
        return self


class Token2RewardsRequest(_BaseTokenRequest):
    ground_truth_text: str = Field(..., min_length=1)
    spk_xvector_centroid: list[float] = Field(..., min_length=1)
    prompt_wav_base64: str = Field(..., min_length=1)


class HealthzResponse(BaseModel):
    status: str
    cosyvoice_model_dir: str
    asr_model_id: str
    device: str
    loaded_at: str
    token2rewards_is_mock: bool


class Token2WerResponse(BaseModel):
    wer: float
    asr_hyp_text: str
    ground_truth_text_norm: str
    asr_hyp_text_norm: str


class Token2RewardsResponse(BaseModel):
    r_asr: float
    r_spk_sim: float
    r_dns_mos: float
