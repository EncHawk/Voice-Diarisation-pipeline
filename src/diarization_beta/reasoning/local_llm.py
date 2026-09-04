"""Local LLM via llama-cli subprocess (llama.cpp) — structured JSON only (spec §21)."""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import tempfile


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}, "required": ["name", "type"]},
        },
        "identity_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker_id": {"type": "string"},
                    "candidate_name": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["speaker_id", "candidate_name", "confidence"],
            },
        },
    },
    "required": ["entities", "identity_hypotheses"],
}

NAME_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["names"],
}


def _json_schema_for_names(allowed_names: list[str] | None) -> dict:
    """Build a closed-world schema so the model cannot emit a new name."""
    schema = json.loads(json.dumps(JSON_SCHEMA))
    if allowed_names is not None:
        names = list(dict.fromkeys(allowed_names))
        candidate_schema = {"enum": [None, *names]}
        entity_schema = {"type": "string", "enum": names} if names else {"type": "string", "enum": [""]}
        schema["properties"]["identity_hypotheses"]["items"]["properties"]["candidate_name"] = candidate_schema
        schema["properties"]["entities"]["items"]["properties"]["name"] = entity_schema
    return schema


def _extract_first_json(text: str) -> dict | None:
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first { ... } block
    # Greedy then back off
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        # Remove <think> blocks if present
        snippet = re.sub(r"<think>.*?</think>", "", snippet, flags=re.DOTALL)
        snippet = snippet.strip()
        try:
            return json.loads(snippet)
        except Exception:
            # try to fix trailing commas
            snippet = re.sub(r",\s*}", "}", snippet)
            snippet = re.sub(r",\s*]", "]", snippet)
            try:
                return json.loads(snippet)
            except Exception:
                pass
    # Try to find JSON line-by-line
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


class LocalLLM:
    def __init__(
        self,
        model_path: str | pathlib.Path | None = None,
        llama_cli: str = "llama-cli",
        ctx_size: int = 4096,
        temp: float = 0.2,
        n_predict: int = 512,
        threads: int = 4,
    ):
        self.model_path = pathlib.Path(model_path) if model_path else None
        self.llama_cli = llama_cli
        self.ctx_size = ctx_size
        self.temp = temp
        self.n_predict = n_predict
        self.threads = threads

    def _model_available(self) -> bool:
        import shutil

        if shutil.which(self.llama_cli) is None:
            return False
        if self.model_path and not self.model_path.exists():
            # try default discovery: models/*.gguf
            candidates = list(pathlib.Path("models").glob("*.gguf")) if pathlib.Path("models").exists() else []
            if not candidates:
                return False
            self.model_path = candidates[0]
        return True

    def generate(
        self,
        prompt: str,
        retries: int = 1,
        allowed_names: list[str] | None = None,
        mode: str = "identity",
    ) -> dict:
        """Generate structured JSON. Returns parsed dict; on failure returns empty hypotheses."""
        empty_output = {"names": []} if mode == "names" else {"entities": [], "identity_hypotheses": []}
        if not self._model_available():
            print(f"[llm] model not available at {self.model_path}, returning no hypotheses")
            return empty_output

        # Write prompt to temp file to avoid shell quoting issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
            pf.write(prompt)
            prompt_file = pf.name

        schema_file = None
        try:
            # Write JSON schema to temp file for --json-schema-file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as sf:
                schema = NAME_LIST_SCHEMA if mode == "names" else _json_schema_for_names(allowed_names)
                json.dump(schema, sf)
                schema_file = sf.name

            cmd = [
                self.llama_cli,
                "-m",
                str(self.model_path),
                "-f",
                prompt_file,
                "-c",
                str(self.ctx_size),
                "--temp",
                str(self.temp),
                "-n",
                str(self.n_predict),
                "-t",
                str(self.threads),
                "--no-display-prompt",
                "-jf",
                schema_file,
                "--simple-io",
                "--single-turn",
            ]
            # Some builds use --json-schema-file, some use -jf; --simple-io helps subprocess
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            # llama-cli prints to stdout; warnings to stderr
            # Extract generated text: after prompt, the JSON
            # With --no-display-prompt, stdout is just generation
            text = stdout.strip()
            if not text:
                # fallback: try without --no-display-prompt and strip prompt
                text = (stdout + stderr).strip()

            parsed = _extract_first_json(text)
            if parsed is not None:
                if mode == "names":
                    return {"names": self._sanitize_name_list(parsed)}
                # Validate minimally
                if "identity_hypotheses" in parsed and "entities" in parsed:
                    return self._sanitize_output(parsed, allowed_names)
                # wrap
                return self._sanitize_output(
                    {"entities": parsed.get("entities", []), "identity_hypotheses": parsed.get("identity_hypotheses", [])},
                    allowed_names,
                )

            if retries > 0:
                constrained = prompt + "\n\nIMPORTANT: Respond with valid JSON only. No explanation, no <think> tags. Example: {\"entities\":[],\"identity_hypotheses\":[]}"
                if mode == "names":
                    constrained = prompt + "\n\nIMPORTANT: Return JSON only in this exact form: {\"names\":[\"Name\"]}. Do not return an empty list when names are explicitly present."
                return self.generate(constrained, retries - 1, allowed_names=allowed_names, mode=mode)

            print(f"[llm] Failed to parse JSON. stdout[:500]={text[:500]} stderr[:500]={stderr[:500]}")
            return empty_output
        except subprocess.TimeoutExpired:
            print("[llm] timeout")
            return empty_output
        except Exception as e:
            print(f"[llm] error: {e}")
            return empty_output
        finally:
            try:
                pathlib.Path(prompt_file).unlink(missing_ok=True)
            except Exception:
                pass
            if schema_file:
                try:
                    pathlib.Path(schema_file).unlink(missing_ok=True)
                except Exception:
                    pass

    @staticmethod
    def _sanitize_name_list(parsed: dict) -> list[str]:
        """Normalize the dedicated name-list response without inventing values."""
        raw_names = parsed.get("names", [])
        if not isinstance(raw_names, list):
            return []
        names = []
        for value in raw_names:
            if isinstance(value, dict):
                value = value.get("name")
            if not isinstance(value, str):
                continue
            name = " ".join(value.split())
            if name and name not in names:
                names.append(name)
        return names

    @staticmethod
    def _sanitize_output(parsed: dict, allowed_names: list[str] | None) -> dict:
        """Keep only well-formed hypotheses and exact closed-world name selections."""
        allowed = set(allowed_names) if allowed_names is not None else None
        entities = []
        for entity in parsed.get("entities", []):
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            if not isinstance(name, str) or (allowed is not None and name not in allowed):
                continue
            entities.append({"name": name, "type": entity.get("type", "person")})

        hypotheses = []
        for hypothesis in parsed.get("identity_hypotheses", []):
            if not isinstance(hypothesis, dict):
                continue
            speaker_id = hypothesis.get("speaker_id")
            name = hypothesis.get("candidate_name")
            if not isinstance(speaker_id, str) or (name is not None and not isinstance(name, str)):
                continue
            if allowed is not None and name is not None and name not in allowed:
                name = None
            try:
                confidence = max(0.0, min(1.0, float(hypothesis.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            evidence = hypothesis.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            hypotheses.append(
                {
                    "speaker_id": speaker_id,
                    "candidate_name": name,
                    "confidence": confidence,
                    "evidence": [str(item) for item in evidence if isinstance(item, str)],
                }
            )
        return {"entities": entities, "identity_hypotheses": hypotheses}
