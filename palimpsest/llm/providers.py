import os
import json
from typing import Optional
from palimpsest.llm.interface import BaseLLMClient
from palimpsest.llm.mock_llm import MockLLMClient

class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            self.client = None

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0
    ) -> str:
        if not self.client or not self.api_key:
            return MockLLMClient().generate(prompt, system, json_mode, temperature)
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLMClient:
    """
    Factory to return an LLM client.
    Checks environment for OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY.
    Defaults to MockLLMClient if keys are not present.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "auto")
    
    if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        return OpenAILLMClient(model=model or "gpt-4o-mini")
    
    return MockLLMClient(model_name=model or "mock-gpt-4o")
