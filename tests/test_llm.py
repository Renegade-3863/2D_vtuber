from llm_api_client import LLMApiClient

if __name__ == "__main__":
    client = LLMApiClient()
    reply = client.chat_once("你好，请用一句话介绍你自己。")
    print("AI:", reply)