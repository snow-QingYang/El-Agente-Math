"""
Use GPT-5 Nano to extract detailed formula issues from ICLR formula_issues_iclr2024_simple.json.

Input: openreview_crawler/output/formula_issues_iclr2024_simple.json
Output: JSON with simplified fields:
{
  "metadata": {...},
  "papers": [
    {
      "paper_index": int,
      "paper_id": str,
      "paper_title": str,
      "paper_pdf_link": str,
      "issues": [
        {
          "review_id": str,
          "formula_errors": [
            {
              "formula_location": str,
              "category": "Wrong equation / notational inconsistency / symbol misuse"
                         | "Unclear/Confusion notation"
                         | "Missing justification/proof",
              "evidence": str
            }
          ]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

# Load .env similar to other scripts
_DOTENV_PATH = find_dotenv(usecwd=True)
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH)

PROMPT_TEMPLATE = """You are a review-analysis assistant. For each review, extract where the reviewer says a formula/equation/notation is wrong OR unclear OR missing justification/derivation, and classify the issue.

Allowed categories (choose one per issue):
- Wrong equation / notational inconsistency / symbol misuse
- Unclear/Confusion notation
- Missing justification/proof

Input review text:
{review_text}

Return JSON only:
{{
  "formula_errors": [
    {{
      "formula_location": "e.g., Equation 3.2, line 142, notation for sigma",
      "category": "<one of the allowed categories>",
      "evidence": "short quote or paraphrase from the review"
    }}
  ]
}}

Rules:
- Only include an issue if the reviewer references a specific equation/notation (e.g., “Eq. 6-8”, “Equation 3.2”, “loss definition notation”). If you cannot identify a concrete location, return an empty list.
- If the reviewer asks for a derivation/proof or says a step is missing, classify as “Missing justification/proof”.
- If the review does not mention any formula/notation/derivation concern, return an empty list.
- Be concise in evidence; avoid long excerpts.
- If multiple issues exist, include multiple entries in the list.
"""


def normalize(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get("value", ""))
    return str(val or "")


class FormulaDetailsAnalyzer:
    def __init__(self, model: str = "gpt-5-nano") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", model)
        print("using model:", self.model)

    def analyze_review(self, review_text: str) -> Dict[str, Any]:
        prompt = PROMPT_TEMPLATE.format(review_text=review_text)
        use_temp = not (self.model.startswith("gpt-5") or self.model.startswith("chatgpt-5"))
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract formula issues. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                **({"temperature": 0.1} if use_temp else {}),
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            errors = data.get("formula_errors", []) or []
        except Exception as exc:
            raise RuntimeError(f"OpenAI call failed: {exc}") from exc
        return {"formula_errors": errors}

    def run(
        self,
        input_file: Path,
        output_file: Path,
        limit_papers: int,
        concurrency: int = 10,
    ) -> Dict[str, Any]:
        raw = json.load(input_file.open())
        papers = raw.get("issues", [])
        if limit_papers:
            papers = papers[:limit_papers]

        results: List[Dict[str, Any]] = []
        keywords = ("equation", "eq.", "eq ", "eq(", "formula", "notation", "proof", "derivation", "symbol")

        def process_review(review_text: str) -> Dict[str, Any]:
            lines = [ln for ln in review_text.splitlines() if any(k.lower() in ln.lower() for k in keywords)]
            focused_text = "\n".join(lines) if lines else review_text
            try:
                analysis = self.analyze_review(focused_text)
            except Exception:
                analysis = {"formula_errors": []}
            errors = analysis.get("formula_errors", []) or []
            filtered_errors = []
            for err in errors:
                loc = err.get("formula_location", "") or ""
                ev = err.get("evidence", "") or ""
                if not loc or not ev:
                    continue
                filtered_errors.append(
                    {
                        "formula_location": loc,
                        "category": err.get("category", ""),
                        "evidence": ev,
                    }
                )
            if filtered_errors:
                return {"formula_errors": filtered_errors}

            import re as _re
            eq_pattern = _re.compile(r"(eq\.?\s*[0-9]+(?:-[0-9]+)?|equation\s*[0-9]+(?:\.[0-9]+)?)", _re.IGNORECASE)
            fallback_errors = []
            for ln in lines:
                lower = ln.lower()
                m = eq_pattern.search(lower)
                if not m:
                    continue
                loc = m.group(1).strip()
                if "deriv" in lower or "proof" in lower:
                    cat = "Missing justification/proof"
                elif "notation" in lower or "symbol" in lower or "typo" in lower or "inconsisten" in lower:
                    cat = "Wrong equation / notational inconsistency / symbol misuse"
                else:
                    cat = "Unclear/Confusion notation"
                evidence = ln.strip()
                if len(evidence) > 300:
                    evidence = evidence[:300]
                fallback_errors.append({"formula_location": loc, "category": cat, "evidence": evidence})
            return {"formula_errors": fallback_errors}

        def process_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
            paper_result = {
                "paper_index": paper.get("paper_index"),
                "paper_id": paper.get("paper_id"),
                "paper_title": paper.get("paper_title"),
                "paper_pdf_link": paper.get("paper_pdf_link"),
                "issues": [],
            }
            for issue in paper.get("issues", []):
                fields = issue.get("review_fields") or {}
                text_blocks = [str(v) for v in fields.values() if v]
                text = "\n".join(text_blocks) or issue.get("review_text", "")
                if not text.strip():
                    continue
                try:
                    analysis = process_review(text)
                except Exception as exc:
                    analysis = {"formula_errors": [], "error": str(exc)}
                paper_result["issues"].append(
                    {
                        "review_id": issue.get("review_id"),
                        "formula_errors": analysis.get("formula_errors", []),
                    }
                )
            return paper_result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_paper = {executor.submit(process_paper, paper): paper for paper in papers}
            for future in tqdm(as_completed(future_to_paper), total=len(future_to_paper), desc="Papers"):
                try:
                    res = future.result()
                except Exception as exc:
                    res = {"issues": [], "error": str(exc)}
                results.append(res)

        output = {
            "metadata": {
                "total_papers_analyzed": len(results),
                "model_used": self.model,
            },
            "papers": results,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(output, output_file.open("w"), indent=2)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formula issues in ICLR reviews with GPT-5 Nano.")
    parser.add_argument(
        "--input",
        default="openreview_crawler/output/formula_issues_iclr2024_simple.json",
        help="Path to formula_issues_iclr2024_simple.json",
    )
    parser.add_argument(
        "--output",
        default="openreview_crawler/output/formula_issues_iclr2024_detailed.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of papers to process (default: 10). Set 0 to process all.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent GPT calls per paper.",
    )
    args = parser.parse_args()

    analyzer = FormulaDetailsAnalyzer()
    analyzer.run(
        input_file=Path(args.input),
        output_file=Path(args.output),
        limit_papers=args.limit,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
