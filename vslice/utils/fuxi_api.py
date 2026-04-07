"""
FuxiAPI - Async client for NetEase AIGC platform.
Ported from flowchart_help/utils/fuxi_api.py for standalone use in VideoRAG.

Supported models (set via model_name):
    deepseek-r1, gemini-2.5-flash-preview-04-17, qwen3-235b-a22b,
    claude-opus-4@20250514, claude-sonnet-4@20250514, gemini-2.5-pro-preview-05-06,
    gpt-4.1, gemini-2.5-pro
"""
import random
import string
import time
import hashlib
import traceback
import asyncio
from typing import List, Dict, Optional, Tuple

import aiohttp

# --- Default config (from flowchart_help/config/base.yaml) ---
DEFAULT_CONFIG = {
    "app_id": "fc35d142-5e91-4109-a94e-16a34460b6c6",
    "app_key": "petkqten1baaqrfre63i24msilll0l",
    "project_id": "big_world_agent_pid_for_flow_chart_tool",
    "bearer_app_key": "sk-g0v76xdnloEcXfp2jb9muWFuMphBt43T",
    "end_point": "https://aigc-api.fuxi.netease.com/v1/chat/completions",
    "model_name": "claude-opus-4-6",
    "max_answer_tokens": 32000,
}


class FuxiAPI:
    def __init__(self, app_id=None, app_key=None, project_id=None, 
                 model_name=None, end_point=None, max_answer_tokens=None):
        self.app_id = app_id or DEFAULT_CONFIG["app_id"]
        self.app_key = app_key or DEFAULT_CONFIG["app_key"]
        self.project_id = project_id or DEFAULT_CONFIG["project_id"]
        self.model_name = model_name or DEFAULT_CONFIG["model_name"]
        self.end_point = end_point or DEFAULT_CONFIG["end_point"]
        self.max_answer_tokens = max_answer_tokens or DEFAULT_CONFIG["max_answer_tokens"]
        self.session = None

    @property
    def headers(self):
        nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        timestamp = str(int(time.time()))
        str2sign = f"appId={self.app_id}&nonce={nonce}&timestamp={timestamp}&appkey={self.app_key}"
        sign = hashlib.md5(str2sign.encode()).hexdigest().upper()
        return {
            "appId": self.app_id,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": sign,
            "version": "v2",
            "Content-Type": "application/json",
            "projectId": self.project_id
        }

    async def ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def get_response(self, prompt: str, model_name: str = None) -> str:
        """Simple single-prompt call. Returns the model's text response."""
        request_body = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model_name or self.model_name,
            "max_tokens": self.max_answer_tokens,
        }

        try:
            session = await self.ensure_session()
            async with session.post(self.end_point, headers=self.headers, json=request_body) as response:
                if response.status == 200:
                    resp_data = await response.json()
                    status = resp_data.get("status", "")
                    
                    # Legacy Fuxi wrapper format
                    if status == "000000":
                        content = resp_data["detail"]["choices"][0]["message"]["content"]
                        usage = resp_data["detail"].get("usage", {})
                        print(f"✅ Tokens — prompt: {usage.get('promptTokens', '?')}, "
                              f"completion: {usage.get('completionTokens', '?')}, "
                              f"total: {usage.get('totalTokens', '?')}")
                        return content
                    # Standard OpenAI format (no wrapper)
                    elif "choices" in resp_data:
                        content = resp_data["choices"][0]["message"]["content"]
                        usage = resp_data.get("usage", {})
                        print(f"✅ Tokens — prompt: {usage.get('prompt_tokens', '?')}, "
                              f"completion: {usage.get('completion_tokens', '?')}, "
                              f"total: {usage.get('total_tokens', '?')}")
                        return content
                    else:
                        return f"❌ API error: {resp_data.get('desc', resp_data)}"
                else:
                    error_text = await response.text()
                    return f"❌ HTTP {response.status}: {error_text}"
        except Exception as e:
            return f"❌ Request failed: {e}\n{traceback.format_exc()}"

    async def chat(self, messages: List[Dict], model_name: str = None,
                   max_answer_tokens: int = None, **kwargs) -> Tuple[str, list]:
        """Full chat with message history. Returns (content, tool_calls)."""
        new_messages = [m for m in messages if m.get("content") not in ("", None)]
        request_body = {
            "messages": new_messages,
            "model": model_name or self.model_name,
            "max_tokens": max_answer_tokens or self.max_answer_tokens,
        }
        request_body.update(kwargs)

        try:
            session = await self.ensure_session()
            async with session.post(self.end_point, headers=self.headers, json=request_body) as response:
                if response.status == 200:
                    resp_data = await response.json()
                    status = resp_data.get("status", "")
                    
                    # Legacy Fuxi wrapper format
                    if status == "000000":
                        content = resp_data["detail"]["choices"][0]["message"]["content"]
                        tool_calls = resp_data["detail"]["choices"][0]["message"].get("tool_calls", [])
                        usage = resp_data["detail"].get("usage", {})
                        print(f"✅ Tokens — prompt: {usage.get('promptTokens', '?')}, "
                              f"completion: {usage.get('completionTokens', '?')}, "
                              f"total: {usage.get('totalTokens', '?')}")
                        return content, tool_calls
                    # Standard OpenAI format (no wrapper)
                    elif "choices" in resp_data:
                        content = resp_data["choices"][0]["message"]["content"]
                        tool_calls = resp_data["choices"][0]["message"].get("tool_calls", [])
                        usage = resp_data.get("usage", {})
                        print(f"✅ Tokens — prompt: {usage.get('prompt_tokens', '?')}, "
                              f"completion: {usage.get('completion_tokens', '?')}, "
                              f"total: {usage.get('total_tokens', '?')}")
                        return content, tool_calls
                    else:
                        return f"❌ API error: {resp_data.get('desc', resp_data)}", []
                else:
                    error_text = await response.text()
                    return f"❌ HTTP {response.status}: {error_text}", []
        except Exception as e:
            return f"❌ Request failed: {e}\n{traceback.format_exc()}", []
