from agent_builder.llm_providers.anaconda_llm import AnacondaLLM

if __name__ == "__main__":
    llm = AnacondaLLM()
    prompt = "Hello, how are you?"
    print("Prompt:", prompt)
    result = llm.generate(prompt)
    print("Result:", result) 