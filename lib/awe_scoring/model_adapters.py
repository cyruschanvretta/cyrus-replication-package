from __future__ import annotations

import importlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


def render_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"<{message['role']}>\n{message['content']}" for message in messages)


def build_payload(messages: list[dict[str, str]], model: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "tgi":
        return {
            "inputs": render_prompt(messages),
            "parameters": {
                "temperature": model.get("temperature", 0),
                "max_new_tokens": model.get("max_tokens", 80),
                "return_full_text": False,
            },
        }
    return {
        "model": model.get("recommended_family", "gemma-3-12b-it"),
        "messages": messages,
        "temperature": model.get("temperature", 0),
        "max_tokens": model.get("max_tokens", 80),
    }


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return extract_text(value[0])
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported model response type: {type(value).__name__}")
    candidates = [
        value.get("generated_text"),
        value.get("text"),
        (value.get("message") or {}).get("content") if isinstance(value.get("message"), dict) else None,
        (value.get("generation") or {}).get("content") if isinstance(value.get("generation"), dict) else None,
    ]
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            candidates.extend([(choice.get("message") or {}).get("content"), choice.get("text")])
    outputs = value.get("outputs")
    if isinstance(outputs, list) and outputs:
        candidates.append(extract_text(outputs[0]))
    for candidate in candidates:
        if isinstance(candidate, str):
            return candidate
    raise ValueError(f"Could not locate generated text in keys {sorted(value)}")


class ModelAdapter(Protocol):
    def invoke(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class SageMakerAdapter:
    model: dict[str, Any]

    def __post_init__(self) -> None:
        import boto3

        settings = self.model["sagemaker"]
        endpoint_env = settings.get("endpoint_name_env", "AWE_SAGEMAKER_ENDPOINT")
        self.endpoint = os.environ.get(endpoint_env, "")
        if not self.endpoint:
            raise ValueError(f"Environment variable {endpoint_env} is required")
        region = os.environ.get(settings.get("region_env", "AWS_REGION"))
        self.client = boto3.client("sagemaker-runtime", region_name=region or None)

    def invoke(self, messages: list[dict[str, str]]) -> str:
        settings = self.model["sagemaker"]
        payload = build_payload(messages, self.model, settings.get("payload_mode", "messages"))
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint,
            ContentType=settings.get("content_type", "application/json"),
            Accept=settings.get("accept", "application/json"),
            Body=json.dumps(payload).encode("utf-8"),
        )
        return extract_text(json.loads(response["Body"].read().decode("utf-8")))


@dataclass
class HttpAdapter:
    model: dict[str, Any]

    def __post_init__(self) -> None:
        settings = self.model["http"]
        url_env = settings.get("url_env", "AWE_MODEL_URL")
        self.url = os.environ.get(url_env, "")
        if not self.url:
            raise ValueError(f"Environment variable {url_env} is required")
        self.token = os.environ.get(settings.get("token_env", "AWE_MODEL_TOKEN"), "")

    def invoke(self, messages: list[dict[str, str]]) -> str:
        settings = self.model["http"]
        body = json.dumps(build_payload(messages, self.model, settings.get("payload_mode", "messages"))).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.model.get("timeout_seconds", 120)) as response:
            return extract_text(json.loads(response.read().decode("utf-8")))


@dataclass
class OllamaAdapter:
    """Local Ollama chat transport; no external model service is required."""

    model: dict[str, Any]

    def __post_init__(self) -> None:
        settings = self.model.get("ollama", {})
        base_url = os.environ.get(settings.get("url_env", "AWE_OLLAMA_URL"), "")
        self.url = (base_url or settings.get("url", "http://127.0.0.1:11434")).rstrip("/") + "/api/chat"
        explicit = settings.get("_explicit_model_override")
        environment_model = os.environ.get(settings.get("model_env", "AWE_OLLAMA_MODEL"), "")
        self.model_name = (
            settings.get("model") if explicit else environment_model or settings.get("model")
        ) or self.model.get("recommended_family", "gemma3:12b")

    def invoke(self, messages: list[dict[str, str]]) -> str:
        settings = self.model.get("ollama", {})
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.model.get("temperature", 0),
                "num_predict": self.model.get("max_tokens", 80),
            },
        }
        if settings.get("think") is not None:
            body["think"] = bool(settings["think"])
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.model.get("timeout_seconds", 120)) as response:
            return extract_text(json.loads(response.read().decode("utf-8")))


@dataclass
class CallableAdapter:
    model: dict[str, Any]

    def __post_init__(self) -> None:
        target = self.model.get("callable", {}).get("target")
        if not target or ":" not in target:
            raise ValueError("model.callable.target must be module:function")
        module_name, function_name = target.split(":", 1)
        self.function = getattr(importlib.import_module(module_name), function_name)

    def invoke(self, messages: list[dict[str, str]]) -> str:
        return extract_text(self.function(messages=messages, model=self.model.get("recommended_family")))


class MockAdapter:
    def invoke(self, messages: list[dict[str, str]]) -> str:
        contract = messages[-1]["content"].rsplit("\n", 1)[-1]
        return contract.replace("#", "2")


def build_adapter(model: dict[str, Any], override: str | None = None) -> ModelAdapter:
    name = override or model.get("adapter", "sagemaker")
    if name == "sagemaker":
        return SageMakerAdapter(model)
    if name == "http":
        return HttpAdapter(model)
    if name == "ollama":
        return OllamaAdapter(model)
    if name == "callable":
        return CallableAdapter(model)
    if name == "mock":
        return MockAdapter()
    raise ValueError(f"Unknown model adapter: {name}")


def invoke_with_retries(adapter: ModelAdapter, messages: list[dict[str, str]], attempts: int) -> tuple[str, int]:
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return adapter.invoke(messages), attempt
        except Exception as caught:  # adapter errors are recorded by the caller
            error = caught
            if attempt < attempts:
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
    assert error is not None
    raise error
