"""
Detect formula-related issues in reviews using GPT.

Output keeps only papers/reviews where GPT flags a formula issue.
Structure:
{
  "metadata": {...},
  "issues": [
    {
      "paper_id": ...,
      "paper_number": ...,
      "paper_title": ...,
      "paper_pdf": ...,
      "issues": [
        {
          "review_id": ...,
          "reviewer": ...,
          "review_fields": {summary/strengths/weaknesses/questions/...},
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from tqdm import tqdm

_DOTENV_PATH = find_dotenv(usecwd=True)
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH)


def norm_val(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get("value", ""))
    return str(val or "")


class FormulaIssueDetector:
    def __init__(self, model: str = "gpt-5-nano") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", model)

    @staticmethod
    def _get_first(review_content: Dict[str, Any], keys: List[str], default: str = "N/A") -> str:
        for k in keys:
            if k in review_content and review_content[k]:
                val = review_content[k]
                if isinstance(val, dict):
                    val = val.get("value", "")
                return str(val)
        return default

    def _build_prompt(self, review_content: Dict[str, Any]) -> str:
        summary = self._get_first(review_content, ["summary"], "N/A")
        strengths = self._get_first(review_content, ["strengths", "claims_and_evidence"], "N/A")
        weaknesses = self._get_first(review_content, ["weaknesses", "other_strengths_and_weaknesses", "soundness"], "N/A")
        questions = self._get_first(review_content, ["questions", "questions_for_authors"], "N/A")
        comment = self._get_first(review_content, ["comment", "other_comments_or_suggestions"], "N/A")

        return f"""Decide if this review explicitly raises formula/equation/notation issues (wrong equations, missing terms, sign/typo, unclear notation, missing derivation/proof). Be strict: only mark true if formula/notation/math expressions are discussed.

Review content:
[Summary] {summary}
[Strengths] {strengths}
[Weaknesses] {weaknesses}
[Questions] {questions}
[Comment] {comment}

Return JSON only:
{{
  "has_formula_issue": true/false,
  "confidence": "high/medium/low",
  "relevant_quotes": ["quote1", "quote2"],
  "summary": "one-line summary if true"
}}"""

    def analyze_review(self, review_content: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(review_content)
        use_temp = not (self.model.startswith("gpt-5") or self.model.startswith("chatgpt-5"))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a review analysis expert. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            **({"temperature": 0.1} if use_temp else {}),
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        result = json.loads(content)
        return {
            "has_formula_issue": bool(result.get("has_formula_issue")),
            "confidence": result.get("confidence", "low"),
            "relevant_quotes": result.get("relevant_quotes", []),
            "summary": result.get("summary", ""),
        }

    def run(
        self,
        input_file: Path,
        output_file: Path,
        limit: int | None = None,
        concurrency: int = 10,
    ) -> Dict[str, Any]:
        data = json.load(input_file.open())
        papers_in = data["submissions"]

        tasks: List[Dict[str, Any]] = []
        for paper in papers_in:
            title = norm_val(paper["content"].get("title"))
            pdf = paper["content"].get("pdf", "")
            number = paper.get("number")
            pid = paper.get("id")
            for review in paper.get("reviews", []):
                # skip author comments if any
                if any("Authors" in sig for sig in review.get("signatures", [])):
                    continue
                tasks.append(
                    {
                        "paper_id": pid,
                        "paper_number": number,
                        "paper_title": title,
                        "paper_pdf": pdf,
                        "review": review,
                    }
                )

        if limit is not None:
            tasks = tasks[:limit]

        results: List[Dict[str, Any]] = []

        def _task(task: Dict[str, Any]) -> Dict[str, Any]:
            analysis = self.analyze_review(task["review"]["content"])
            return {**task, "analysis": analysis}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            fut_map = {executor.submit(_task, t): t for t in tasks}
            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Detect formula issues"):
                results.append(fut.result())

        # group by paper and keep only has_formula_issue
        grouped: Dict[str, Dict[str, Any]] = {}
        for item in results:
            if not item["analysis"].get("has_formula_issue"):
                continue
            pid = item["paper_id"] or f"unknown_{len(grouped)}"
            paper_entry = grouped.setdefault(
                pid,
                {
                    "paper_id": item["paper_id"],
                    "paper_number": item["paper_number"],
                    "paper_title": item["paper_title"],
                    "paper_pdf": item["paper_pdf"],
                    "issues": [],
                },
            )
            content = item["review"]["content"]
            review_fields = {
                k: v for k, v in content.items() if k in ["summary", "strengths", "weaknesses", "questions", "soundness", "presentation", "contribution", "comment"]
            }
            paper_entry["issues"].append(
                {
                    "review_id": item["review"].get("id"),
                    "reviewer": item["review"].get("signatures", ["Unknown"])[0],
                    "review_fields": review_fields,
                    "analysis": item["analysis"],
                }
            )

        issues = list(grouped.values())
        out = {
            "metadata": {
                "total_reviews_analyzed": len(results),
                "papers_with_formula_issues": len(issues),
                "model_used": self.model,
            },
            "issues": issues,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, output_file.open("w"), indent=2)
        return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect formula issues using GPT.")
    parser.add_argument("--input", required=True, help="Path to raw OpenReview JSON")
    parser.add_argument("--output", required=True, help="Path to write formula issues JSON")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"), help="OpenAI model name")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of reviews (0 for all)")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency for GPT calls")
    args = parser.parse_args()

    det = FormulaIssueDetector(model=args.model)
    det.run(
        input_file=Path(args.input),
        output_file=Path(args.output),
        limit=args.limit if args.limit > 0 else None,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
