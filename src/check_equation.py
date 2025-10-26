import os
import subprocess
import tempfile
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
from openai import OpenAI
from src.formula_extractor import MathFormula


@dataclass
class Definition:
    symbol: str
    expression: str


@dataclass
class EquationCheckResult:
    is_valid: bool
    method: Literal['sympy', 'openai']
    description: str
    confidence: Literal['high', 'medium', 'low', 'not_provided'] = 'not_provided'


def create_sympy_verification_script(equation: str, definitions: List[Definition]) -> str:
    """Create a Python script that uses SymPy to verify an equation, optionally with OpenAI assistance."""

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    definitions_text = "\n".join([f"{d.symbol} = {d.expression}" for d in definitions])

    prompt = f"""Generate ONLY Python code. No explanations, no markdown, no text before or after. Just pure Python code that starts with import statements.

Equation to verify: {equation}

Definitions:
{definitions_text if definitions_text else "No definitions provided"}

Requirements:
1. Import sympy as sp, json, and sys
2. Define all necessary symbols
3. Apply the given definitions
4. Parse and verify the equation
5. Handle both equality (=) and expression evaluation
6. For equalities, check if left side equals right side after simplification
7. Use advanced SymPy techniques like:
   - expand() for polynomial expansion
   - factor() for factorization
   - trigsimp() for trigonometric simplification
   - powsimp() for power simplification
   - cancel() for rational function simplification
8. The program should output a JSON result with:
   - 'is_valid': boolean
   - 'description': a short description of the verification code
   - 'left', 'right', 'simplified_difference' for equalities (MUST convert to strings using str())
9. IMPORTANT: All SymPy expressions MUST be converted to strings using str() before adding to JSON

Start your response with 'import sympy as sp' and nothing else before it."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a code generator. Output only valid Python code. No explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1500
    )

    generated_script = response.choices[0].message.content.strip()
    return generated_script


def run_sympy_verification(equation: str, definitions: List[Definition]) -> EquationCheckResult:
    """Execute SymPy verification script and return results."""
    script = create_sympy_verification_script(equation, definitions)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            output = json.loads(result.stdout.strip())
            return EquationCheckResult(
                is_valid=output.get('is_valid', False),
                method="sympy",
                description=output.get('description', 'No description provided'),
                confidence="high"
            )
        else:
            error_msg = f"Script execution failed: {result.stderr}"
            return EquationCheckResult(
                is_valid=False,
                method="sympy",
                description=error_msg,
                confidence="not_provided"
            )
    except Exception as e:
        return EquationCheckResult(
            is_valid=False,
            method="sympy",
            description=f"Unexpected error: {str(e)}",
            confidence="not_provided"
        )
    finally:
        os.unlink(temp_file)


def check_with_llm(equation: str, definitions: List[Definition]) -> EquationCheckResult:
    """Use LLM to infer whether an equation is true."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    if not client.api_key:
        return EquationCheckResult(
            is_valid=False,
            method="openai",
            description="OpenAI API key not found in environment variables",
        )
    
    # Prepare the prompt
    definitions_text = "\n".join([f"{d.symbol} = {d.expression}" for d in definitions])
    
    prompt = f"""You are a mathematical verification assistant. Given the following definitions and equation, 
determine if the equation is mathematically correct.

Definitions:
{definitions_text if definitions_text else "No definitions provided"}

Equation to verify:
{equation}

Please respond with a JSON object containing:
1. "is_valid": true/false - whether the equation is mathematically correct
2. "reasoning": A brief explanation of your verification
3. "confidence": high/medium/low - your confidence in the assessment

Only respond with the JSON object, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise mathematical verification assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        return EquationCheckResult(
            is_valid=result.get('is_valid', False),
            method="openai",
            description=result.get('reasoning', 'No reasoning provided'),
            confidence=result.get('confidence', 'not_provided')
        )
        
    except Exception as e:
        return EquationCheckResult(
            is_valid=False,
            method="openai",
            description=f"Error during OpenAI verification: {str(e)}",
            confidence="not_provided"
        )


def check_equation(equation: MathFormula, definitions: List[Definition]) -> EquationCheckResult:
    """Assume definitions is a list of definition for all symbols that appear in the equation and definition expressions"""
    sympy_result = run_sympy_verification(equation.formula, definitions)
    if sympy_result.is_valid:
        return sympy_result
    
    return check_with_llm(equation.formula, definitions)


if __name__ == "__main__":
    test_equation = MathFormula(
        formula="p**2 + 2*p*q + q**2 = (p + q)**2",
        formula_type="equation",
        context_before="Perfect square formula",
        context_after="should always be true",
        line_number=1,
        raw_latex="p^2 + 2pq + q^2 = (p + q)^2"
    )
    
    definitions = [
        Definition(symbol="p", expression="3*a - 2"),
        Definition(symbol="q", expression="b + 5"),
        Definition(symbol="a", expression="a"),
        Definition(symbol="b", expression="b"),
    ]

    result = check_equation(test_equation, definitions)
    print(result)
    
