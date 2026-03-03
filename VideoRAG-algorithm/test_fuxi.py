"""
Quick test script for FuxiAPI integration.
Usage: python test_fuxi.py
"""
import asyncio
from fuxi_api import FuxiAPI

async def main():
    api = FuxiAPI()  # Uses default config

    print(f"🔧 Model:    {api.model_name}")
    print(f"🔧 Endpoint: {api.end_point}")
    print("-" * 50)

    # --- Test 1: Simple prompt ---
    print("\n📝 Test 1: get_response (simple prompt)")
    response = await api.get_response("Who are you?")
    print(f"Response: {response}")

    # --- Test 2: Chat with history ---
    print("\n📝 Test 2: chat (multi-turn)")
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    content, tool_calls = await api.chat(messages)
    print(f"Response: {content}")
    print(f"Tool calls: {tool_calls}")

    # --- Test 3: Switch model ---
    print("\n📝 Test 3: get_response with different model")
    response = await api.get_response(
        "Say 'Hello from Qwen!' and nothing else.", 
        model_name="qwen3-235b-a22b"
    )
    print(f"Response: {response}")

    await api.close()
    print("\n✅ All tests complete!")

if __name__ == "__main__":
    asyncio.run(main())
