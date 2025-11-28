import os
from typing import List, Dict, Any, Optional

from kimina_client import KiminaClient, CheckResponse
from pydantic_ai import Agent, ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from dotenv import load_dotenv

from src.benchmark_providers import BaseLLMProvider

tool_prompt = """
You have access to a Lean4 REPL tool. To use it, your response should be in the form:
<lean_repl>
Your Lean4 imports here
Your Lean4 code here
</lean_repl>
"""


client = KiminaClient()
def lean_repl(code: str) -> CheckResponse:
    """Run Lean4 REPL with the provided code"""
    try:
        return client.check(code)
    except Exception as e:
        return str(e)



class DeepseekAgent:
    def __init__(self, max_steps: int = 10):
        load_dotenv()

        model = OpenAIChatModel(
            'deepseek/deepseek-prover-v2',
            provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
        )

        agent = Agent(model, system_prompt=tool_prompt)
        self.agent = agent
        self.history: list[ModelRequest | ModelResponse] = []
        self.max_steps = max_steps
        
    def run(self, system_prompt: str, user_prompt: str) -> str:
        """Run multi-step agent with tool parsing and history tracking"""
        step = 0
        self.history.append(ModelRequest.user_text_prompt(tool_prompt))
        self.history.append(ModelRequest.user_text_prompt(system_prompt))
        self.history.append(ModelRequest.user_text_prompt(user_prompt))

        while step < self.max_steps:
            result = self.agent.run_sync(None, message_history=self.history)
            response_text = result.response.text
            self.history.append(result.response)

            if '<lean_repl>' in response_text and '</lean_repl>' in response_text:
                start = response_text.find('<lean_repl>') + len('<lean_repl>')
                end = response_text.find('</lean_repl>')
                code = response_text[start:end].strip()
                
                # Execute tool
                print(code)
                tool_result = lean_repl(code)
                tool_message = f"Tool lean_repl returned:\n{tool_result}"
                print(tool_message)
                self.history.append(ModelRequest.user_text_prompt(tool_message))

                step += 1
            else:
                return response_text
        self.history.pop(0)
        result = self.agent.run_sync(None, message_history=self.history)
        return result.response.text

    def clear_history(self) -> None:
        self.history = []


class ProverProvider(BaseLLMProvider):
    def __init__(self, model,max_retries, retry_delay):
        super().__init__(model, max_retries, retry_delay)
        self.agent = DeepseekAgent()

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        self.agent.run(system_prompt, user_prompt)


if __name__ == "__main__":
    import asyncio

    agent = DeepseekAgent()
    result = agent.run(
        'Prove that for all natural numbers n, the sum of the first n natural numbers equals n*(n+1)/2', "")
    print(f"Result: {result}")