"""
Analyze reviews with ChatGPT 5 Nano to flag those that mention formula/equation issues.

Dependencies:
  - openai (official SDK)
  - python-dotenv (optional; loads OPENAI_API_KEY from .env)

Defaults:
  input_file  = output/ICML_cc_2025_Conference_rejected.json
  output_file = output/formula_issues_chatgpt5_nano.json
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Prefer project root .env (e.g., /Users/.../El-Agente-Math/.env)
_DOTENV_PATH = find_dotenv(usecwd=True)
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH)


class ChatGPT5NanoFormulaFilter:
    """Use ChatGPT 5 Nano to detect formula-related issues in reviews."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "chatgpt-5-nano",
        sleep_seconds: float = 0.4,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY. Set it via env or .env.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL", model)
        self.sleep_seconds = sleep_seconds
        self.debug = os.getenv("FORMULA_DEBUG", "0") not in {"", "0", "false", "False"}

    def _debug_print(self, msg: str) -> None:
        if self.debug:
            print(f"[DEBUG] {msg}")

    @staticmethod
    def _build_prompt(review_content: Dict[str, Any]) -> str:
        summary = review_content.get("summary", "N/A")
        claims = review_content.get("claims_and_evidence", "N/A")
        theoretical = review_content.get("theoretical_claims", "N/A")
        weaknesses = review_content.get("other_strengths_and_weaknesses", "N/A")
        questions = review_content.get("questions_for_authors", "N/A")
        comments = review_content.get("other_comments_or_suggestions", "N/A")
        comment_text = review_content.get("comment", comments)

        return f"""You are a review-analysis assistant. Decide whether the review explicitly points out issues with formulas/equations/mathematical expressions (e.g., incorrect formulas, wrong derivations, unclear definitions, symbol misuse, proof errors, formatting issues, mismatch between formula and text).

Review content:
[Summary] {summary}
[Claims & evidence] {claims}
[Theoretical claims] {theoretical}
[Strengths/weaknesses] {weaknesses}
[Questions for authors] {questions}
[Other comments] {comment_text}

Return JSON in English only:
{{
  "has_formula_issue": true/false,
  "confidence": "high/medium/low",
  "issue_types": ["type1", "type2"],
  "relevant_quotes": ["quote1", "quote2"],
  "summary": "one-line English summary if any issue"
}}

Only set has_formula_issue=true when the review clearly calls out formula/equation/math expression problems; vague requests for more math analysis count as false."""

    def analyze_review(self, review_content: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(review_content)

        try:
            use_temp = not (self.model.startswith("gpt-5") or self.model.startswith("chatgpt-5"))
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a review analysis expert. Always respond with JSON in English.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **({"temperature": 0.1} if use_temp else {}),  # GPT-5 models disallow temperature
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
        except Exception as exc:  # fail fast for auth/model errors
            raise RuntimeError(
                f"OpenAI call failed: {exc}. Check model name ({self.model}) access and OPENAI_API_KEY."
            ) from exc

        return {
            "has_formula_issue": bool(result.get("has_formula_issue")),
            "confidence": result.get("confidence", "low"),
            "issue_types": result.get("issue_types", []),
            "relevant_quotes": result.get("relevant_quotes", []),
            "summary": result.get("summary", ""),
        }

    def analyze_author_ack(
        self, review_text: str, author_reply_text: str
    ) -> Dict[str, Any]:
        """Use GPT to decide if the author reply acknowledges the formula error raised in the review."""
        prompt = f"""You will check whether an author reply explicitly acknowledges that the review's formula/equation concern is valid (e.g., admits an error/typo, agrees with correction, promises a fix).

Review (what the reviewer raised):
{review_text}

Author reply:
{author_reply_text}

Return JSON only:
{{
  "acknowledges_formula_error": true/false,
  "confidence": "high/medium/low",
  "evidence": ["short quote or reason"]
}}"""

        try:
            use_temp = not (self.model.startswith("gpt-5") or self.model.startswith("chatgpt-5"))
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a review-reply analyst. Judge only acknowledgment of formula/equation mistakes. Respond with JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **({"temperature": 0.1} if use_temp else {}),
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI call failed on author ack: {exc}. Check model/API key."
            ) from exc

        return {
            "acknowledges_formula_error": bool(result.get("acknowledges_formula_error")),
            "confidence": result.get("confidence", "low"),
            "evidence": result.get("evidence", []),
        }

    def run(
        self,
        input_file: Path,
        output_file: Path,
        limit: int | None = None,
        concurrency: int = 20,
    ) -> Dict[str, Any]:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Map review_id -> list of author replies for later join.
        replies_by_review: Dict[str, List[Dict[str, Any]]] = {}
        for paper in data["submissions"]:
            for review in paper.get("reviews", []):
                if any("Authors" in sig for sig in review.get("signatures", [])):
                    reply_to = review.get("replyto")
                    if reply_to:
                        replies_by_review.setdefault(reply_to, []).append(review)

        reviews: List[Dict[str, Any]] = []
        for paper in data["submissions"]:
            title = paper["content"].get("title", {}).get("value", "Unknown")
            for review in paper.get("reviews", []):
                # Exclude author-written comments; keep all other reviewer/AC comments
                if any("Authors" in sig for sig in review.get("signatures", [])):
                    continue
                reviews.append(
                    {
                        "paper_title": title,
                        "paper_id": paper.get("id"),
                        "paper_number": paper.get("number"),
                        "review_id": review.get("id"),
                        "reviewer": review.get("signatures", ["Unknown"])[0],
                        "content": review.get("content", {}),
                    }
                )

        total_non_author = len(reviews)
        with_author_reply = sum(1 for r in reviews if r["review_id"] in replies_by_review)
        self._debug_print(f"Non-author comments collected: {total_non_author}")
        self._debug_print(f"Non-author with author replies: {with_author_reply}")

        if limit is not None:
            reviews = reviews[:limit]

        analyses: List[Dict[str, Any]] = []

        def _task(review: Dict[str, Any]) -> Dict[str, Any]:
            analysis = self.analyze_review(review["content"])
            return {
                "paper_title": review["paper_title"],
                "paper_id": review["paper_id"],
                "paper_number": review["paper_number"],
                "review_id": review["review_id"],
                "reviewer": review["reviewer"],
                "analysis": analysis,
                "review_content": review["content"],
            }

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_review = {executor.submit(_task, r): r for r in reviews}
            for future in tqdm(as_completed(future_to_review), total=len(reviews), desc="Analyze reviews (concurrent)"):
                try:
                    result = future.result()
                except Exception as exc:
                    review = future_to_review[future]
                    raise RuntimeError(
                        f"Review {review['review_id']} failed: {exc}"
                    ) from exc

                analyses.append(result)

        formula_issues = [
            item for item in analyses if item["analysis"].get("has_formula_issue")
        ]

        def normalize_comment(comment: Any) -> str:
            if isinstance(comment, dict):
                return str(comment.get("value", ""))
            return str(comment or "")

        def build_review_text(review_item: Dict[str, Any]) -> str:
            """Flatten review content with emphasis on comment text for GPT context."""
            content = review_item.get("review_content", {})
            parts: List[str] = []
            if "comment" in content:
                parts.append(f"comment: {normalize_comment(content.get('comment'))}")
            for key, val in content.items():
                if key == "comment":
                    continue
                if isinstance(val, str):
                    parts.append(f"{key}: {val}")
                elif isinstance(val, dict):
                    parts.append(f"{key}: {val.get('value','')}")
                else:
                    parts.append(f"{key}: {val}")
            return "\n".join(parts)

        ack_checked = 0
        ack_found = 0
        for item in formula_issues:
            review_id = item["review_id"]
            review_text = build_review_text(item)

            # Check each author reply with GPT for acknowledgment
            for reply in replies_by_review.get(review_id, []):
                reply_text = normalize_comment(reply.get("content", {}).get("comment"))
                if not reply_text.strip():
                    continue
                ack_checked += 1
                ack_analysis = self.analyze_author_ack(review_text, reply_text)
                if ack_analysis.get("acknowledges_formula_error"):
                    ack_found += 1
                    item["author_acknowledgment"] = {
                        "reply_id": reply.get("id"),
                        "author_signature": reply.get("signatures", []),
                        "reply_text": reply_text,
                        "ack_analysis": ack_analysis,
                    }
                    break

        self._debug_print(f"Author replies checked with GPT: {ack_checked}, acknowledgments found: {ack_found}")

        formula_issues_with_author_ack = [
            item for item in formula_issues if item.get("author_acknowledgment")
        ]

        output = {
            "metadata": {
                "total_reviews_analyzed": len(analyses),
                "reviews_with_formula_issues": len(formula_issues),
                "reviews_with_formula_issues_and_author_ack": len(formula_issues_with_author_ack),
                "percentage": f"{(len(formula_issues) / len(analyses) * 100):.2f}%"
                if analyses
                else "0%",
                "model_used": self.model,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "all_analyses": analyses,
            "formula_issues_only": formula_issues,
            "formula_issues_with_author_ack": formula_issues_with_author_ack,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output


def main() -> None:
    input_file = Path("output/ICLR_cc_2024_Conference_rejected.json")
    output_file = Path("output/formula_issues_ICLR2024.json")
    concurrency = int(os.getenv("OPENAI_CONCURRENCY", "20"))

    print("=" * 60)
    print("Formula-Issue Review Filter (ChatGPT 5 Nano)")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {os.getenv('OPENAI_MODEL', 'chatgpt-5-nano')}")
    print(f"Concurrency: {concurrency}\n")

    analyzer = ChatGPT5NanoFormulaFilter()
    results = analyzer.run(
        input_file=input_file,
        output_file=output_file,
        concurrency=concurrency,
    )

    print("\nDone")
    print(f"Total reviews: {results['metadata']['total_reviews_analyzed']}")
    print(
        f"With formula issues: {results['metadata']['reviews_with_formula_issues']} "
        f"({results['metadata']['percentage']})"
    )
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
