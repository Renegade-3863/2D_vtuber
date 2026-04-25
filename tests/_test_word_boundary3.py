"""Test: edge_tts WordBoundary events with boundary param"""
import asyncio
import edge_tts

async def test_zh():
    print("=== Chinese (WordBoundary) ===")
    comm = edge_tts.Communicate(
        "你好，我是一个AI助手。请问有什么可以帮你的？",
        "zh-CN-XiaoxiaoNeural",
        boundary="WordBoundary"
    )
    async for chunk in comm.stream():
        if chunk["type"] == "WordBoundary":
            offset_s = chunk["offset"] / 1e7
            dur_s = chunk["duration"] / 1e7
            print(f"  offset={offset_s:.3f}s  duration={dur_s:.3f}s  end={offset_s+dur_s:.3f}s  text=\"{chunk['text']}\"")

async def test_en():
    print("\n=== English (WordBoundary) ===")
    comm = edge_tts.Communicate(
        "Hello, I am an AI assistant. How can I help you today?",
        "en-US-JennyNeural",
        boundary="WordBoundary"
    )
    async for chunk in comm.stream():
        if chunk["type"] == "WordBoundary":
            offset_s = chunk["offset"] / 1e7
            dur_s = chunk["duration"] / 1e7
            print(f"  offset={offset_s:.3f}s  duration={dur_s:.3f}s  end={offset_s+dur_s:.3f}s  text=\"{chunk['text']}\"")

asyncio.run(test_zh())
asyncio.run(test_en())
