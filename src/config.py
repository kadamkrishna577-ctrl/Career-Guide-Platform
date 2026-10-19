# src/config.py
import os
import time
import itertools
import threading
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv(override=True)

def _clean_key(raw: str) -> str:
    return raw.strip().strip('"').strip("'").rstrip(",").strip()

def _is_valid_looking_key(k: str) -> bool:
    if not k or len(k) < 30:
        return False
    lower_k = k.lower()
    placeholders = ["your", "key_here", "firstkey", "secondkey", "example", "..."]
    return not any(p in lower_k for p in placeholders)

def _load_all_gemini_keys() -> list[tuple[str, str]]:
    keys = []
    seen = set()

    for i in range(1, 21):
        var_name = f"GOOGLE_API_KEY_{i}"
        k = _clean_key(os.getenv(var_name, ""))
        if _is_valid_looking_key(k) and k not in seen:
            keys.append((var_name, k))
            seen.add(k)

    csv_keys = os.getenv("GOOGLE_API_KEYS", "")
    for idx, raw_k in enumerate(csv_keys.split(","), 1):
        k = _clean_key(raw_k)
        if _is_valid_looking_key(k) and k not in seen:
            keys.append((f"GOOGLE_API_KEYS[{idx}]", k))
            seen.add(k)

    single_key = _clean_key(os.getenv("GOOGLE_API_KEY", ""))
    if _is_valid_looking_key(single_key) and single_key not in seen:
        keys.append(("GOOGLE_API_KEY", single_key))
        seen.add(single_key)

    if not keys:
        raise ValueError("❌ No valid Google API Keys found in .env!")

    print(f"🔑 Loaded {len(keys)} Gemini API key(s) into the rotation pool.")
    return keys


API_KEYS = _load_all_gemini_keys()
_key_cycle = itertools.cycle(range(len(API_KEYS)))
_lock = threading.Lock()

def get_next_key_with_name() -> tuple[str, str]:
    with _lock:
        idx = next(_key_cycle)
        return API_KEYS[idx]


class FaultTolerantStructuredLLM:
    """
    Handles:
    - 429 RESOURCE_EXHAUSTED -> Immediately rotates to next API key
    - 400 API_KEY_INVALID    -> Skips broken key and alerts which key number is invalid
    - 503 UNAVAILABLE        -> Waits 2s and retries; falls back to backup model if primary is overloaded
    """
    def __init__(self, schema, model_name: str, temperature: float = 0.0):
        self.schema = schema
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, prompt: str):
        last_error = None
        # Try across all keys, and if 503 persists after 4 tries, fall back to gemini-3.5-flash
        models_to_try = [self.model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite"]

        for current_model in models_to_try:
            for attempt in range(len(API_KEYS)):
                key_name, api_key = get_next_key_with_name()
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=current_model,
                        google_api_key=api_key,
                        temperature=self.temperature,
                        max_retries=0,
                    )
                    structured_llm = llm.with_structured_output(self.schema)
                    return structured_llm.invoke(prompt)
                except Exception as e:
                    err_str = str(e)
                    last_error = e

                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        print(f"⚠️ {key_name} hit quota limit (429) on {current_model}. Rotating to next key...")
                        continue

                    if "API_KEY_INVALID" in err_str or "400" in err_str:
                        print(f"❌ {key_name} ({api_key[:6]}...{api_key[-4:]}) is INVALID! Skipping...")
                        continue

                    if "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                        print(f"⏳ Google Server 503 (High Demand) on {current_model} ({key_name}). Waiting 2s & retrying...")
                        time.sleep(2)
                        # If 503 happens 3 times in a row on the primary model, switch to backup model
                        if attempt >= 2 and current_model != models_to_try[-1]:
                            print(f"🔄 {current_model} servers are overloaded right now. Switching to backup model...")
                            break
                        continue

                    if "404" in err_str or "NOT_FOUND" in err_str:
                        print(f"⚠️ Model {current_model} not found (404). Switching to next model...")
                        break

                    raise e

        raise RuntimeError("❌ All API keys and fallback models failed!") from last_error


class RotatingGeminiWrapper:
    def __init__(self, temperature: float = 0.0):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.temperature = temperature

    def with_structured_output(self, schema):
        return FaultTolerantStructuredLLM(
            schema=schema,
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def invoke(self, prompt):
        last_error = None
        models_to_try = [self.model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite"]
        for current_model in models_to_try:
            for attempt in range(len(API_KEYS)):
                key_name, api_key = get_next_key_with_name()
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=current_model,
                        google_api_key=api_key,
                        temperature=self.temperature,
                        max_retries=0,
                    )
                    return llm.invoke(prompt)
                except Exception as e:
                    err_str = str(e)
                    last_error = e
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        continue
                    if "API_KEY_INVALID" in err_str or "400" in err_str:
                        print(f"❌ {key_name} is INVALID! Skipping...")
                        continue
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(2)
                        if attempt >= 2:
                            break
                        continue
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        break
                    raise e
        raise RuntimeError("❌ All API keys and fallback models failed!") from last_error


def get_llm(temperature: float = 0.0) -> RotatingGeminiWrapper:
    return RotatingGeminiWrapper(temperature=temperature)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    _, api_key = get_next_key_with_name()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )