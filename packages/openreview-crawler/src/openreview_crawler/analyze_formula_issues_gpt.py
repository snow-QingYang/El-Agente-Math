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



class FormulaIssueDetectorICML:
    """
    ICML-specific detector that combines detection and classification in one step.
    Uses the detailed PROMPT_TEMPLATE from analyze_iclr_formula_details.py
    """

    # Detailed prompt template adapted from analyze_iclr_formula_details.py
    PROMPT_TEMPLATE = """You are a review-analysis assistant. For each review, extract where the reviewer says a formula/equation/theorem/definition has an issue, and classify both the error type and the reviewer's confidence level.

## CRITICAL: Specific Location Required

⚠️ **ONLY extract issues that reference a SPECIFIC LOCATION**. Do NOT extract:
- General notation complaints without specific location (e.g., "the notation is confusing")
- Vague mathematical comments (e.g., "the formulation could be better")
- Issues without identifiable location (e.g., "the math is unclear")

✓ **INCLUDE**: "Equation 3 has a typo" (specific equation)
✓ **INCLUDE**: "Theorem 4.1 is incorrect" (specific theorem)
✓ **INCLUDE**: "Error on page 5, line 12" (specific page and line)
✓ **INCLUDE**: "Definition 2 is unclear" (specific definition)
✓ **INCLUDE**: "Lemma 3 needs proof" (specific lemma)

✗ **EXCLUDE**: "The notation is confusing" (no specific location)
✗ **EXCLUDE**: "Symbol usage is unclear" (no specific location)
✗ **EXCLUDE**: "The mathematical framework needs work" (too vague)
✗ **EXCLUDE**: "The loss function is wrong" (no specific location)

**Remember**: Only extract if reviewer explicitly mentions a specific, identifiable location.

## Location Format Requirements

**ALL locations must be in standardized format. Use ONE of the following:**

1. **Equation**: `EQ (X)`
   - "Equation 3" → `EQ (3)`
   - "Eq. 5-7" → `EQ (5-7)`
   - "equation 2.1" → `EQ (2.1)`
   - "Eqs 4 and 5" → Create separate entries with `EQ (4)` and `EQ (5)`

2. **Theorem**: `THEOREM (X)`
   - "Theorem 1" → `THEOREM (1)`
   - "Thm. 4.2" → `THEOREM (4.2)`

3. **Lemma**: `LEMMA (X)`
   - "Lemma 3" → `LEMMA (3)`
   - "Lemma A.1" → `LEMMA (A.1)`

4. **Definition**: `DEF (X)`
   - "Definition 2" → `DEF (2)`
   - "Def. 3.1" → `DEF (3.1)`

5. **Page**: `PAGE (X)`
   - "page 5" → `PAGE (5)`
   - "on page 12" → `PAGE (12)`

6. **Line**: `LINE (X)`
   - "line 402" → `LINE (402)`
   - "below line 15" → `LINE (15)`

7. **Page with Line**: `PAGE (X), LINE (Y)`
   - "page 5, line 10" → `PAGE (5), LINE (10)`
   - "line 15 on page 3" → `PAGE (3), LINE (15)`

8. **Section**: `SECTION (X)`
   - "Section 4" → `SECTION (4)`
   - "Sec. 3.2" → `SECTION (3.2)`

9. **Algorithm**: `ALGORITHM (X)`
   - "Algorithm 1" → `ALGORITHM (1)`
   - "Alg. 2" → `ALGORITHM (2)`

10. **Appendix**: `APPENDIX (X)`
   - "Appendix A" → `APPENDIX (A)`
   - "Appendix B.2" → `APPENDIX (B.2)`

## Error Type Categories (choose ONE per issue):

1. **Typo / Symbol misuse** - Wrong symbols, typos in formulas, incorrect variable names
2. **Mathematically wrong** - The equation is mathematically incorrect or produces wrong results
3. **Notational inconsistency** - Inconsistent use of notation across the paper (e.g., same symbol for different meanings)
4. **Redundancy** - Unnecessary or duplicate equations/notation
5. **Unclear/Confusing notation** - Hard to understand, ambiguous, or poorly explained notation
6. **Missing justification/proof** - Lacks derivation, proof, or explanation of how equation was obtained

## Confidence Levels (based on reviewer's certainty):

- **Very certain** - Reviewer is definite (e.g., "This is definitely wrong", "This equation is incorrect")
- **Confident** - Reviewer states issue without hedging (e.g., "This notation is confusing", "Equation X is wrong")
- **Suggestion** - Reviewer suggests an issue (e.g., "This might be wrong", "I suspect there's an error")
- **Not sure** - Reviewer is uncertain (e.g., "I'm not sure, but...", "This part might have an error")

## Input review text:
{review_text}

## Output Format:

Return JSON only. **IMPORTANT**: Output "evidence" field FIRST, then "formula_location", then "category", then "confidence".

{{
  "has_formula_issue": true/false,
  "formula_errors": [
    {{
      "evidence": "Direct quote from the review showing the issue",
      "formula_location": "Must use one of the standardized formats above",
      "category": "One of the 6 categories above (EXACT match required)",
      "confidence": "One of the 4 confidence levels above (EXACT match required)"
    }}
  ]
}}

## Rules:

1. **has_formula_issue**: Set to true if there are any mathematical/formula errors, false otherwise
2. **Evidence first**: Always extract the evidence quote BEFORE determining location/category
3. **Specific location required**: Only include an issue if you can identify a SPECIFIC location (equation, theorem, page, etc.)
4. **Use standardized location format**: Convert reviewer's location reference to the appropriate standardized format
5. **Classify error type accurately** using the 6 categories
6. **Assess confidence from reviewer's language**
7. **Evidence must be a direct quote** (be as strict as possible, keep under 200 characters)
8. **Empty list cases**: If no specific location mentioned, return empty formula_errors array
9. **Multiple locations**: Create separate entries for each location problem

## Examples:

Example 1 - Equation with typo:
Review: "Equation 3 has a typo: it should be σ² instead of σ."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "Equation 3 has a typo: it should be σ² instead of σ", "formula_location": "EQ (3)", "category": "Typo / Symbol misuse", "confidence": "Very certain"}}]}}

Example 2 - Theorem issue:
Review: "Theorem 4.1 is incorrect because it doesn't account for the boundary case."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "Theorem 4.1 is incorrect because it doesn't account for the boundary case", "formula_location": "THEOREM (4.1)", "category": "Mathematically wrong", "confidence": "Confident"}}]}}

Example 3 - Page and line reference:
Review: "There is a typo on page 5, line 12 where it says 'x > y' but should be 'x < y'."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "typo on page 5, line 12 where it says 'x > y' but should be 'x < y'", "formula_location": "PAGE (5), LINE (12)", "category": "Typo / Symbol misuse", "confidence": "Very certain"}}]}}

Example 4 - Definition unclear:
Review: "Definition 2 is confusing and needs better explanation."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "Definition 2 is confusing and needs better explanation", "formula_location": "DEF (2)", "category": "Unclear/Confusing notation", "confidence": "Confident"}}]}}

Example 5 - No specific location:
Review: "The notation is confusing throughout the paper."
Output: {{"has_formula_issue": false, "formula_errors": []}}

Example 6 - Multiple locations:
Review: "Equations 4 and 5 have inconsistent notation for the same variable."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "Equations 4 and 5 have inconsistent notation", "formula_location": "EQ (4)", "category": "Notational inconsistency", "confidence": "Confident"}}, {{"evidence": "Equations 4 and 5 have inconsistent notation", "formula_location": "EQ (5)", "category": "Notational inconsistency", "confidence": "Confident"}}]}}

Example 7 - Lemma needs proof:
Review: "The authors should provide a proof for Lemma 3."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "The authors should provide a proof for Lemma 3", "formula_location": "LEMMA (3)", "category": "Missing justification/proof", "confidence": "Confident"}}]}}

Example 8 - Line reference only:
Review: "Typo below line 402 where it says x=y."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "Typo below line 402 where it says x=y", "formula_location": "LINE (402)", "category": "Typo / Symbol misuse", "confidence": "Confident"}}]}}

Example 9 - Page reference only:
Review: "The math on page 7 contains errors."
Output: {{"has_formula_issue": true, "formula_errors": [{{"evidence": "The math on page 7 contains errors", "formula_location": "PAGE (7)", "category": "Mathematically wrong", "confidence": "Confident"}}]}}
"""

    def __init__(self, model: str = "gpt-5-nano", debug_mode: bool = False) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model#os.getenv("OPENAI_MODEL", model)
        self.debug_mode = debug_mode

    @staticmethod
    def _format_review_content(review_content: Dict[str, Any]) -> str:
        """
        Format review content as 'key: value' pairs for the prompt.
        Extracts value from dict structure if needed.
        """
        formatted_parts = []

        for key, val in review_content.items():
            # Extract actual value from nested dict structure
            if isinstance(val, dict) and "value" in val:
                actual_value = val["value"]
            else:
                actual_value = val

            # Skip empty values and metadata fields
            if not actual_value or key in ["code_of_conduct", "overall_recommendation"]:
                continue

            # Format as "key: value"
            formatted_parts.append(f"{key}: {actual_value}")

        return "\n\n".join(formatted_parts)

    def _build_prompt(self, review_content: Dict[str, Any]) -> str:
        """Build prompt using the detailed template and formatted review content."""
        review_text = self._format_review_content(review_content)
        return self.PROMPT_TEMPLATE.format(review_text=review_text)

    def analyze_review(self, review_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single review and return both detection and classification."""
        prompt = self._build_prompt(review_content)

        if self.debug_mode:
            print("="*60)
            print("PROMPT SENT TO GPT:")
            print("="*60)
            print(prompt)
            print("="*60)

        use_temp = not (self.model.startswith("gpt-5") or self.model.startswith("chatgpt-5"))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You extract formula issues from reviews. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            **({"temperature": 0.1} if use_temp else {}),
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        result = json.loads(content)
        if self.debug_mode:
            print("=" * 60)
            print("GPT RESPONSE:")
            print("=" * 60)
            print(result)
            print("=" * 60)

        # Extract and structure the result
        has_issue = bool(result.get("has_formula_issue", False))
        formula_errors = result.get("formula_errors", [])

        return {
            "has_formula_issue": has_issue,
            "formula_errors": formula_errors,
            # Keep legacy fields for backward compatibility
            "confidence": "high" if formula_errors else "low",
            "summary": f"Found {len(formula_errors)} formula issue(s)" if formula_errors else "No formula issues"
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
        total_formula_errors = 0

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

            # Count formula errors
            formula_errors = item["analysis"].get("formula_errors", [])
            total_formula_errors += len(formula_errors)

            paper_entry["issues"].append(
                {
                    "review_id": item["review"].get("id"),
                    "reviewer": item["review"].get("signatures", ["Unknown"])[0],
                    "review_fields": review_fields,
                    "analysis": item["analysis"],
                    "formula_errors": formula_errors,  # Include detailed errors
                }
            )

        issues = list(grouped.values())
        out = {
            "metadata": {
                "total_reviews_analyzed": len(results),
                "papers_with_formula_issues": len(issues),
                "total_formula_errors": total_formula_errors,
                "model_used": self.model,
            },
            "issues": issues,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, output_file.open("w"), indent=2, ensure_ascii=False)
        return out


class FormulaIssueDetectorNIPS(FormulaIssueDetectorICML):
    """NeurIPS-specific detector that limits prompt to key review fields."""

    def _build_prompt(self, review_content: Dict[str, Any]) -> str:
        def _extract(key: str) -> str:
            val = review_content.get(key, "")
            if isinstance(val, dict) and "value" in val:
                return str(val.get("value", ""))
            return str(val or "")

        summary = _extract("summary") or "N/A"
        strengths = (
            _extract("strengths_and_weaknesses")
            or _extract("strengths_and_weeknesses")
            or _extract("strengths")
            or "N/A"
        )
        weaknesses = (
            _extract("weaknesses")
            or "N/A"
        )
        questions = _extract("questions") or "N/A"

        limitations = _extract("limitations") or "N/A"

        review_text = "\n\n".join(
            [
                f"summary: {summary}\n",
                f"strengths_and_weaknesses: "
                f"{"strengths:" + strengths}\n"
                f"{weaknesses if weaknesses != 'N/A' else ''}\n",
                f"questions: {questions}\n"
                f"{("limitations: " + limitations) if limitations != 'N/A' else ''}",

            ]
        )
        return self.PROMPT_TEMPLATE.format(review_text=review_text)


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
