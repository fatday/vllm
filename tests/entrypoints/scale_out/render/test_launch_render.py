# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for render endpoints via `vllm launch` (GPU-less serving)."""

import json
from typing import Any

import httpx
import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from tests.utils import RemoteLaunchRenderServer

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server(request, tmp_path_factory):
    args: list[str] = []
    if getattr(request, "param", None) == "tools":
        template = tmp_path_factory.mktemp("tools") / "chat_template.jinja"
        template.write_text("{{ tools | tojson }}")
        args = ["--chat-template", str(template)]
    with RemoteLaunchRenderServer(MODEL_NAME, args, max_wait_seconds=120) as srv:
        yield srv


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


# -- Chat Completion Render --


@pytest.mark.asyncio
async def test_chat_render_basic(client):
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Response should be a GenerateRequest dict
    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0
    assert all(isinstance(t, int) for t in data["token_ids"])


@pytest.mark.asyncio
async def test_chat_render_multi_turn(client):
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, dict)
    assert "token_ids" in data
    assert isinstance(data["token_ids"], list)
    assert len(data["token_ids"]) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("server", ["tools"], indirect=True)
@pytest.mark.parametrize("api", ["chat/completions", "responses"])
async def test_render_preserves_tool_extensions(client, api):
    """Tool extensions must survive HTTP validation into real template tokens."""
    function = {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {}},
        "response": {"type": "number", "x-unit": "celsius"},
        "return": {"type": "number"},
        "x-metadata": {"labels": ["weather"], "optional": None},
    }
    tool = {"type": "function", "function": function}
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "tool_choice": "none",
    }
    messages = [{"role": "user", "content": "Hello"}]
    if api == "responses":
        payload.update(input=messages, tools=[{"type": "function", **function}])
    else:
        tool["x-tool-metadata"] = {"readOnly": True}
        payload.update(messages=messages, tools=[tool])

    response = await client.post(f"/v1/{api}/render", json=payload)
    assert response.status_code == 200, response.text
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt = tokenizer.decode(response.json()["token_ids"], skip_special_tokens=True)
    assert json.loads(prompt) == [tool]


@pytest.mark.asyncio
async def test_chat_render_multiple_chat_templates():
    # Ships chat_template.jinja and additional_chat_templates/reranker.jinja
    model = "Qwen/Qwen3-VL-Reranker-2B"
    with RemoteLaunchRenderServer(model, [], max_wait_seconds=120) as srv:
        async with httpx.AsyncClient(
            base_url=srv.url_for(""), timeout=30.0
        ) as http_client:
            response = await http_client.post(
                "/v1/chat/completions/render",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    token_ids = response.json()["token_ids"]
    assert token_ids == [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]


# -- Completion Render --


@pytest.mark.asyncio
async def test_completion_render_basic(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Once upon a time",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0

    first_prompt = data[0]
    assert "token_ids" in first_prompt
    assert "sampling_params" in first_prompt
    assert "model" in first_prompt
    assert "request_id" in first_prompt
    assert isinstance(first_prompt["token_ids"], list)
    assert len(first_prompt["token_ids"]) > 0
    assert first_prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_completion_render_multiple_prompts(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": ["Hello world", "Goodbye world"],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    for prompt in data:
        assert "token_ids" in prompt
        assert "sampling_params" in prompt
        assert "model" in prompt
        assert "request_id" in prompt
        assert len(prompt["token_ids"]) > 0
        assert prompt["request_id"].startswith("cmpl-")


@pytest.mark.asyncio
async def test_completion_render_invalid_model(client):
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": "nonexistent-model",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.asyncio
async def test_render_is_fast(client):
    """Render should complete quickly since there is no inference."""
    import time

    start = time.perf_counter()
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Tell me a very long story about " * 10,
        },
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 2.0


# -- Health & Models --


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_endpoint(client):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]
    assert MODEL_NAME in model_ids
