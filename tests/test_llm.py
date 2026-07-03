"""LLMClient mechanics against a local mock Anthropic endpoint: retry on
529 with an llm_retry event, token accounting, model name resolution.
No real credentials or network."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import activegraph_harness.llm as llm_mod


class MockAnthropic(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if type(self).calls == 1:
            payload = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "overloaded"},
                }
            ).encode()
            self.send_response(529)
        else:
            request = json.loads(body)
            payload = json.dumps(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": request["model"],
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 42, "output_tokens": 17},
                }
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_retry_on_overloaded_and_token_accounting(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), MockAnthropic)
    MockAnthropic.calls = 0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}"
        )
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setattr(llm_mod, "RETRY_BASE_DELAY_SEC", 0.01)

        retry_events = []
        client = llm_mod.LLMClient("anthropic/claude-test-model")
        result = asyncio.run(
            client.complete(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
                log_event=lambda t, p: retry_events.append((t, p)),
            )
        )
        assert MockAnthropic.calls == 2
        assert result.text == "ok"
        assert (result.input_tokens, result.output_tokens) == (42, 17)
        assert result.retries == 1
        assert retry_events[0][0] == "llm_retry"
        assert client.model == "claude-test-model"
    finally:
        server.shutdown()


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROP_API_KEY", raising=False)
    with pytest.raises(llm_mod.LLMError, match="ANTHROPIC_API_KEY"):
        llm_mod.LLMClient("claude-test-model")


def test_fallback_key_name_is_accepted(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROP_API_KEY", "test-key")
    client = llm_mod.LLMClient("claude-test-model")
    assert client.model == "claude-test-model"


def test_resolve_model_rejects_foreign_provider():
    with pytest.raises(llm_mod.LLMError, match="provider"):
        llm_mod.resolve_model("openai/gpt-4o")


def test_apply_cache_control_places_two_breakpoints():
    system = "you are a test"
    turns = [
        {"role": "user", "content": "instruction"},
        {"role": "assistant", "content": "reply"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "stable feedback"},
                {"type": "text", "text": "budget line", "volatile": True},
            ],
        },
    ]
    system_blocks, cached = llm_mod.apply_cache_control(system, turns)

    assert system_blocks == [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    # Earlier messages pass through untouched.
    assert cached[0] == turns[0]
    assert cached[1] == turns[1]
    final_blocks = cached[2]["content"]
    # The last stable block carries the moving breakpoint.
    assert final_blocks[0] == {
        "type": "text",
        "text": "stable feedback",
        "cache_control": {"type": "ephemeral"},
    }
    # The volatile block follows the breakpoint, flag stripped, uncached.
    assert final_blocks[1] == {"type": "text", "text": "budget line"}
    # The input was not mutated.
    assert "cache_control" not in str(turns[2]["content"][0])


def test_apply_cache_control_string_final_message():
    system_blocks, cached = llm_mod.apply_cache_control(
        "sys", [{"role": "user", "content": "only message"}]
    )
    assert cached[0]["content"] == [
        {
            "type": "text",
            "text": "only message",
            "cache_control": {"type": "ephemeral"},
        }
    ]
