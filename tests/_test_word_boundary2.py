"""Quick test: edge_tts all chunk types"""
import asyncio
import edge_tts

async def test():
    print("=== All chunk types ===")
    comm = edge_tts.Communicate("Hello, how are you?", "en-US-JennyNeural")
    audio_size = 0
    async for chunk in comm.stream():
        ctype = chunk["type"]
        if ctype == "audio":
            audio_size += len(chunk.get("data", b""))
        else:
            print(f"  type={ctype}")
            for k, v in chunk.items():
                if k != "type":
                    print(f"    {k}={v}")
    print(f"\n  Total audio bytes: {audio_size}")

asyncio.run(test())
