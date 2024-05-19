from mlc_llm.json_ffi import JSONFFIEngine

class ChatState:
    """Helper class to manage chat state"""

    # we use JSON ffi engine to ensure broader coverage
    engine: JSONFFIEngine

    def __init__(self, engine):
        self.engine = engine
        self.history = []

    def process_system_prompts(self):
        """Process system prompts"""
        # TODO(mlc-team): possibly leverage debug option
        # pass a simple prompt to warm up
        self.engine.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )

    def generate(self, prompt: str):
        """Run one generatiohn with the prompt"""
        self.history.append({"role": "user", "content": prompt})
        output_text = ""
        for response in self.engine.chat_completion(messages=self.history):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                if isinstance(choice.delta.content, str):
                    output_text += choice.delta.content
                    print(choice.delta.content, end="", flush=True)
        # print additional \n when generation ends
        print()
        # record the history
        self.history.append({"role": "assistant", "content": output_text})

    def reset_chat(self):
        """Reset the chat history"""
        self.history = []

def test():
    model = "HF://mlc-ai/Mistral-7B-Instruct-v0.2-q3f16_1-MLC"
    engine = JSONFFIEngine(model, mode="interactive")
    chat = ChatState(engine)

    chat.generate("hello")
    chat.generate("tell me about cmu")

test()