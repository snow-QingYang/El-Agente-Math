import os
from typing import List, Dict, Any, Optional

import logfire
from kimina_client import KiminaClient, CheckResponse
from pydantic_ai import Agent, ModelRequest, ModelResponse, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from dotenv import load_dotenv

from src.benchmark_providers import BaseLLMProvider

tool_prompt = """
You have access to a Lean4 REPL tool. To use it, your response should be in the form:
You must use this tool
<lean_repl>
Your Lean4 imports here
Your Lean4 code here
</lean_repl>
"""

client = KiminaClient()


def lean_repl(code: str) -> CheckResponse:
    """Run Lean4 REPL with the provided code"""
    try:
        return client.check(code).results[0].response['messages']
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
            print(response_text)
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
    def __init__(self, model, max_retries, retry_delay):
        super().__init__(model, max_retries, retry_delay)
        load_dotenv()
        from pydantic_ai import Agent
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        logfire.configure()
        logfire.instrument_pydantic_ai()

        provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
        model = GoogleModel('gemini-3-pro-preview', provider=provider)
        self.agent = Agent(model, output_type=str)

        @self.agent.system_prompt
        def system_prompt(ctx: RunContext) -> str:
            return ctx.deps["system_prompt"] + "\n You must call lean4_repl at least once."

        @self.agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
        def lean4_repl(code: str) -> CheckResponse:
            """Run Lean4 REPL with the provided code. You must supply syntatically correct LEAN4 with the necessary imports.
            Example 1: #check Nat
            result:
            [
              {
                "severity": "info",
                "pos": {
                  "line": 1,
                  "column": 0
                },
                "endPos": {
                  "line": 1,
                  "column": 6
                },
                "data": "Nat : Type"
              }
            ]
            Example 2:
            set_option linter.unusedVariables false
            set_option linter.unusedVariables false
            theorem t1 (p q : Prop) (hp : p) (hq : q) : p := hp
            variable (p q r s : Prop)
            #check t1 p q
            #check t1 r s
            #check t1 (r \u2192 s) (s \u2192 r)
            variable (h : r \u2192 s)
            #check t1 (r \u2192 s) (s \u2192 r) h
            result:
            [
            {'severity': 'info', 'pos': {'line': 5, 'column': 12}, 'endPos': {'line': 5, 'column': 18}, 'data': 't1 p q : p → q → p'},
            {'severity': 'info', 'pos': {'line': 6, 'column': 12}, 'endPos': {'line': 6, 'column': 18}, 'data': 't1 r s : r → s → r'},
            {'severity': 'info', 'pos': {'line': 7, 'column': 12}, 'endPos': {'line': 7, 'column': 18}, 'data': 't1 (r → s) (s → r) : (r → s) → (s → r) → r → s'},
            {'severity': 'info', 'pos': {'line': 9, 'column': 12}, 'endPos': {'line': 9, 'column': 18}, 'data': 't1 (r → s) (s → r) h : (s → r) → r → s'}
            ]


            Args:
                code: Code to be ran
            """
            return lean_repl(code)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        self.agent.system_prompt()
        return self.agent.run_sync(user_prompt=user_prompt, deps={"system_prompt": system_prompt}).output


if __name__ == "__main__":
    import asyncio

    print(lean_repl("""set_option linter.unusedVariables false
            set_option linter.unusedVariables false
            theorem t1 (p q : Prop) (hp : p) (hq : q) : p := hp
            variable (p q r s : Prop)
            #check t1 p q
            #check t1 r s
            #check t1 (r \u2192 s) (s \u2192 r)
            variable (h : r \u2192 s)
            #check t1 (r \u2192 s) (s \u2192 r) h"""))