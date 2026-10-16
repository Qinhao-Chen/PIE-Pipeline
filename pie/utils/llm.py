import os
import asyncio
import logging
import httpx
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Response:
    text: str

class OpenAIChatClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 4,
        max_concurrent: int = 16,
        timeout: int = 60
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.sema = asyncio.Semaphore(max_concurrent)

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Response:
        temp = kwargs.get("temperature", 0.2)
        # Match old script: use max_completion_tokens if possible, fallback to max_tokens logic implies standard behavior
        max_tok = kwargs.get("max_tokens", 256)

        # Construct URL
        if self.base_url:
            base = self.base_url.rstrip("/")
            # If user forgot /v1, we usually shouldn't guess, but we need to append the endpoint
            url = f"{base}/chat/completions" if not base.endswith("/chat/completions") else base
        else:
            url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Payload matching old script behavior
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_completion_tokens": max_tok, 
        }

        async with self.sema:
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        r = await client.post(url, headers=headers, json=payload)
                        
                        # Custom handling to debug non-JSON responses
                        if r.status_code != 200:
                            r.raise_for_status()

                        try:
                            data = r.json()
                        except json.JSONDecodeError:
                            # This is the fix for your specific error: log what we actually got
                            logging.error(f"[LLM] JSON Decode Error. URL: {url}, Status: {r.status_code}")
                            logging.error(f"[LLM] Raw Response: {r.text[:200]}...") # Print first 200 chars
                            return Response(text="")

                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            return Response(text=content or "")
                        else:
                            logging.warning(f"[LLM] Unexpected JSON format: {data}")
                            return Response(text="")

                except Exception as e:
                    err_msg = str(e)
                    if attempt == self.max_retries:
                        logging.error(f"[LLM] Failed after {self.max_retries} attempts: {err_msg}")
                        return Response(text="")
                    
                    wait_time = min(60, 2 ** attempt)
                    if "429" in err_msg:
                        wait_time = min(60, 5 * (2 ** attempt))
                    await asyncio.sleep(wait_time)
        return Response(text="")