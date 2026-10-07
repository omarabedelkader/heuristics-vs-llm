"""Ensure the frozen benchmark source's models exist in the local Ollama server."""
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


def ensure_models(models, tags):
    if (not isinstance(models, list) or not models
            or any(not isinstance(model, str) or not model for model in models)):
        raise ValueError("Expected a nonempty list of Ollama model names")
    installed = {entry["name"] for entry in tags["models"]}
    for model in dict.fromkeys(models):
        if model in installed:
            print(f"Ollama model available: {model}", flush=True)
            continue
        print(f"Pulling Ollama model: {model}", flush=True)
        request = Request("http://127.0.0.1:11434/api/pull",
                          data=json.dumps({"model": model, "stream": True}).encode(),
                          headers={"Content-Type": "application/json"})
        success = False
        # Streaming keeps long downloads alive and surfaces API errors even when
        # Ollama returns HTTP 200. Timeouts apply to individual socket operations.
        with urlopen(request, timeout=300) as response:
            for line in response:
                if not line.strip():
                    continue
                progress = json.loads(line)
                if "error" in progress:
                    raise RuntimeError(f"Ollama pull failed for {model}: {progress['error']}")
                success = progress.get("status") == "success"
        if not success:
            raise RuntimeError(f"Ollama pull did not finish successfully for {model}")
        installed.add(model)


if __name__ == "__main__":
    ensure_models(json.loads(Path(sys.argv[1]).read_text()),
                  json.loads(Path(sys.argv[2]).read_text()))
