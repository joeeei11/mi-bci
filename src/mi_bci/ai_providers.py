"""Local Ollama and OpenAI-compatible text generation with OS-held credentials."""

import json
import urllib.parse
import urllib.request


def _local(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


class ModelProvider:
    def __init__(self, kind: str, endpoint: str, model: str):
        if kind not in {"ollama", "openai"} or not model.strip():
            raise ValueError("invalid model provider")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("invalid model endpoint")
        if not _local(endpoint) and parsed.scheme != "https":
            raise ValueError("remote model endpoints require HTTPS")
        self.kind, self.endpoint, self.model = kind, endpoint.rstrip("/"), model

    def set_credential(self, credential: str) -> None:
        import keyring
        keyring.set_password("mi-bci-ai", self.endpoint, credential)

    def generate(self, prompt: str, *, cloud_confirmed: bool = False) -> str:
        if not _local(self.endpoint) and not cloud_confirmed:
            raise PermissionError("cloud model use requires explicit confirmation")
        url = self.endpoint + ("/api/generate" if self.kind == "ollama" else "/chat/completions")
        payload = ({"model": self.model, "prompt": prompt, "stream": False} if self.kind == "ollama"
                   else {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False})
        headers = {"Content-Type": "application/json"}
        if self.kind == "openai":
            import keyring
            credential = keyring.get_password("mi-bci-ai", self.endpoint)
            if credential:
                headers["Authorization"] = "Bearer " + credential
        request = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
        return body["response"] if self.kind == "ollama" else body["choices"][0]["message"]["content"]
