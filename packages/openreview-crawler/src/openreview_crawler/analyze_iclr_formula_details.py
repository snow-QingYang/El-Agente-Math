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
              "evidence": str,
              "formula_location": str,
              "category": "Typo / Symbol misuse"
                         | "Mathematically wrong"
                         | "Notational inconsistency"
                         | "Redundancy"
                         | "Unclear/Confusing notation"
                         | "Missing justification/proof",
              "confidence": "Very certain" | "Confident" | "Suggestion" | "Not sure"
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

PROMPT_TEMPLATE = """You are a review-analysis assistant. For each review, extract where the reviewer says a formula/equation has an issue, and classify both the error type and the reviewer's confidence level.

## CRITICAL: Equation-Specific Issues Only

⚠️ **ONLY extract issues that reference a SPECIFIC equation NUMBER**. Do NOT extract:
- General notation complaints without equation numbers (e.g., "the notation is confusing")
- Symbol issues without equation reference (e.g., "symbol usage is unclear")
- Vague mathematical comments (e.g., "the formulation could be better")

✓ **INCLUDE**: "Equation 3 has a typo" (has equation number)
✓ **INCLUDE**: "The loss in Eq. 5 is wrong" (has equation number)
✓ **INCLUDE**: "Derivation of equation 2.1 is missing" (has equation number)

✗ **EXCLUDE**: "The notation is confusing" (no equation number)
✗ **EXCLUDE**: "Symbol usage is unclear" (no equation number)
✗ **EXCLUDE**: "The mathematical framework needs work" (no equation number)
✗ **EXCLUDE**: "Formula 3 has an error" (says "formula", not "equation")
✗ **EXCLUDE**: "Line 150 has a mistake" (line reference, not equation number)
✗ **EXCLUDE**: "Lemma 3.2 is incorrect" (lemma, not equation)
✗ **EXCLUDE**: "Theorem 3 needs proof" (theorem, not equation)
✗ **EXCLUDE**: "Definition 2 is unclear" (definition, not equation)
✗ **EXCLUDE**: "The loss function is wrong" (no specific equation number)

**Remember**: Only extract if reviewer explicitly mentions "Equation X" or "Eq. X" where X is a number. Do NOT extract for lemmas, theorems, definitions, formulas, or line numbers.

## Location Format Requirement

**ALL locations must be in standardized format: EQ (xxx)**

Conversion examples:
- "Equation 3" → "EQ (3)"
- "Eq. 5-7" → "EQ (5-7)"
- "equation 2.1" → "EQ (2.1)"
- "Equations 10, 11, 12" → Create separate entries: "EQ (10)", "EQ (11)", "EQ (12)"
- "Eq 4" → "EQ (4)"

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

Return JSON only. **IMPORTANT**: Output "evidence" field FIRST, then "formula_location", then "category", then "confidence". This order helps the model reference the evidence when determining location and category.

{{
  "formula_errors": [
    {{
      "evidence": "Direct quote from the review showing the issue",
      "formula_location": "Must be in format: EQ (xxx)",
      "category": "One of the 6 categories above (EXACT match required)",
      "confidence": "One of the 4 confidence levels above (EXACT match required)"
    }}
  ]
}}

## Rules:

1. **Evidence first**: Always extract the evidence quote BEFORE determining location/category. **Be as strict as possible** - quote exactly what the reviewer wrote, preserving their wording.

2. **Specific location required**: Only include an issue if you can identify a specific equation NUMBER:
   - Good: "Eq. 6-8", "Equation 3.2", "equation 4"
   - Bad: vague references like "the loss function", "objective notation" (no equation number)

3. **Classify error type accurately**:
   - Use "Typo / Symbol misuse" only for literal typos or wrong symbols
   - Use "Mathematically wrong" when reviewer says the math/logic is incorrect
   - Use "Notational inconsistency" when same symbol has different meanings or inconsistent use
   - Use "Redundancy" when reviewer says equation is unnecessary or repetitive
   - Use "Unclear/Confusing notation" when reviewer finds it hard to understand
   - Use "Missing justification/proof" when reviewer asks for derivation or says proof is missing

4. **Assess confidence from reviewer's language**:
   - Look for certainty markers: "definitely", "clearly", "obviously", "is wrong" → Very certain
   - Look for hedging: "might", "could be", "possibly", "I think", "not sure" → Suggestion or Not sure
   - Default to "Confident" if reviewer states issue without qualifiers

5. **Evidence guidelines**:
   - **MUST be a direct quote** from the review text (be as strict as possible)
   - Keep quotes concise (under 200 characters preferred)
   - Capture the key criticism about the formula/equation
   - Preserve reviewer's certainty language (e.g., "might be wrong" vs "is wrong")

6. **Empty list cases**:
   - No specific equation NUMBER mentioned
   - Only general mathematical comments without targeting specific numbered equations
   - No formula-related concerns in the review

7. **Multiple issues**: If reviewer mentions multiple equation problems, create separate entries for each.

## Examples:

Example 1 - Very certain about typo (with equation number):
Review: "Equation 3 has a typo: it should be σ² instead of σ."
Output: {{"evidence": "Equation 3 has a typo: it should be σ² instead of σ", "formula_location": "EQ (3)", "category": "Typo / Symbol misuse", "confidence": "Very certain"}}

Example 2 - Not sure about math error (with equation number):
Review: "I'm not sure, but Eq. 5 might produce incorrect results in the boundary case."
Output: {{"evidence": "I'm not sure, but Eq. 5 might produce incorrect results", "formula_location": "EQ (5)", "category": "Mathematically wrong", "confidence": "Not sure"}}

Example 3 - Confident about missing proof (with equation number):
Review: "The authors should provide a derivation for Equation 2.1."
Output: {{"evidence": "The authors should provide a derivation for Equation 2.1", "formula_location": "EQ (2.1)", "category": "Missing justification/proof", "confidence": "Confident"}}

Example 4 - General notation issue (NO equation number - EXCLUDE):
Review: "The notation is confusing throughout the paper."
Output: {{"formula_errors": []}}

Example 5 - Multiple equations:
Review: "Equations 4 and 5 have inconsistent notation for the same variable."
Output: {{"formula_errors": [{{"evidence": "Equations 4 and 5 have inconsistent notation for the same variable", "formula_location": "EQ (4)", "category": "Notational inconsistency", "confidence": "Confident"}}, {{"evidence": "Equations 4 and 5 have inconsistent notation for the same variable", "formula_location": "EQ (5)", "category": "Notational inconsistency", "confidence": "Confident"}}]}}
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

    def validate_formula_errors(
        self,
        errors: List[Dict[str, Any]],
        review_text: str
    ) -> tuple[bool, List[str]]:
        """
        Validate extracted formula errors.

        Returns:
            (is_valid, error_messages)
        """
        import re

        validation_errors = []

        # Pattern for EQ (xxx) format
        location_pattern = re.compile(r'^EQ \([^)]+\)$')

        # Allowed categories
        allowed_categories = {
            "Typo / Symbol misuse",
            "Mathematically wrong",
            "Notational inconsistency",
            "Redundancy",
            "Unclear/Confusing notation",
            "Missing justification/proof"
        }

        # Allowed confidence levels
        allowed_confidence = {
            "Very certain",
            "Confident",
            "Suggestion",
            "Not sure"
        }

        # Normalize review text for comparison (fuzzy matching)
        normalized_review = ' '.join(review_text.split()).lower()

        for idx, err in enumerate(errors):
            # Check 1: Location format
            location = err.get("formula_location", "")
            if not location_pattern.match(location):
                validation_errors.append(
                    f"Error {idx+1}: Location must be 'EQ (xxx)' format, got '{location}'"
                )

            # Check 2: Evidence is direct quote (fuzzy matching - 80% threshold)
            evidence = err.get("evidence", "")
            normalized_evidence = ' '.join(evidence.split()).lower()
            if normalized_evidence:
                # Check word-level matching
                evidence_words = set(normalized_evidence.split())
                matching_words = sum(1 for word in evidence_words if word in normalized_review)
                match_ratio = matching_words / len(evidence_words) if evidence_words else 0

                if match_ratio < 0.8:  # Allow some flexibility but not too much
                    validation_errors.append(
                        f"Error {idx+1}: Evidence appears to not be a direct quote (only {match_ratio:.0%} match)"
                    )

            # Check 3: Category exact match
            category = err.get("category", "")
            if category not in allowed_categories:
                validation_errors.append(
                    f"Error {idx+1}: Category '{category}' is not in allowed list. "
                    f"Must be exactly one of: {', '.join(allowed_categories)}"
                )

            # Check 4: Confidence exact match
            confidence = err.get("confidence", "")
            if confidence not in allowed_confidence:
                validation_errors.append(
                    f"Error {idx+1}: Confidence '{confidence}' is not in allowed list. "
                    f"Must be exactly one of: {', '.join(allowed_confidence)}"
                )

        return (len(validation_errors) == 0, validation_errors)

    def analyze_review(self, review_text: str, max_retries: int = 3) -> tuple[Dict[str, Any], int]:
        """
        Analyze review with validation and retry.

        Returns:
            (result_dict, retry_count) where retry_count is 0-indexed (0 = first attempt succeeded)
        """
        prompt = PROMPT_TEMPLATE.format(review_text=review_text)
        use_temp = not (self.model.startswith("gpt-5") or self.model.startswith("chatgpt-5"))

        for attempt in range(max_retries):
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

                # Validate output
                is_valid, validation_errors = self.validate_formula_errors(errors, review_text)

                if is_valid:
                    return ({"formula_errors": errors}, attempt)  # Return retry count (0-indexed)

                # Validation failed - prepare retry prompt
                if attempt < max_retries - 1:
                    error_msg = "\n".join(validation_errors)
                    retry_prompt = f"""Your previous response had validation errors:

{error_msg}

Please fix these issues and provide corrected output. Remember:
1. Location MUST be in format: EQ (xxx) - examples: EQ (3), EQ (5-7), EQ (2.1)
2. Evidence MUST be a direct quote from the review text (be as strict as possible)
3. Category MUST exactly match one of the 6 allowed categories
4. Confidence MUST exactly match one of the 4 allowed levels
5. ONLY extract issues about specific NUMBERED equations - skip general notation complaints

Original review text:
{review_text}
"""
                    prompt = retry_prompt
                    continue
                else:
                    # Max retries reached - return failure
                    print(f"⚠️  Validation failed after {max_retries} attempts: {validation_errors[:3]}")  # Show first 3 errors
                    return ({"formula_errors": [], "validation_failed": True}, max_retries - 1)

            except Exception as exc:
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {exc}") from exc

        return ({"formula_errors": []}, max_retries - 1)

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

        # Statistics tracking
        total_retry_counts = []
        total_validation_failures = 0
        total_reviews_processed = 0

        # Updated keywords - equation-focused only
        keywords = ("equation", "eq.", "eq ", "eq(")

        def process_review(review_text: str) -> tuple[Dict[str, Any], int, bool]:
            """
            Returns: (result, retry_count, validation_failed)
            """
            # Focus on lines with equation keywords
            lines = [ln for ln in review_text.splitlines()
                     if any(k.lower() in ln.lower() for k in keywords)]
            focused_text = "\n".join(lines) if lines else review_text

            try:
                analysis, retry_count = self.analyze_review(focused_text)
            except Exception:
                return ({"formula_errors": []}, 0, False)

            errors = analysis.get("formula_errors", []) or []
            validation_failed = analysis.get("validation_failed", False)

            # Filter and structure errors
            filtered_errors = []
            for err in errors:
                ev = err.get("evidence", "") or ""
                loc = err.get("formula_location", "") or ""
                cat = err.get("category", "") or ""
                conf = err.get("confidence", "") or ""

                # Skip if missing required fields
                if not ev or not loc or not cat:
                    continue

                # Default confidence if missing
                if not conf:
                    conf = "Confident"

                filtered_errors.append({
                    "evidence": ev,
                    "formula_location": loc,
                    "category": cat,
                    "confidence": conf,
                })

            return ({"formula_errors": filtered_errors}, retry_count, validation_failed)

        def process_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal total_reviews_processed, total_validation_failures, total_retry_counts

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

                total_reviews_processed += 1

                try:
                    analysis, retry_count, validation_failed = process_review(text)
                    total_retry_counts.append(retry_count)
                    if validation_failed:
                        total_validation_failures += 1
                except Exception:
                    analysis = {"formula_errors": []}
                    total_retry_counts.append(0)

                paper_result["issues"].append({
                    "review_id": issue.get("review_id"),
                    "formula_errors": analysis.get("formula_errors", []),
                })
            return paper_result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_paper = {executor.submit(process_paper, paper): paper for paper in papers}
            for future in tqdm(as_completed(future_to_paper), total=len(future_to_paper), desc="Papers"):
                try:
                    res = future.result()
                except Exception as exc:
                    res = {"issues": [], "error": str(exc)}
                results.append(res)

        # Calculate statistics
        avg_retry_count = sum(total_retry_counts) / len(total_retry_counts) if total_retry_counts else 0

        output = {
            "metadata": {
                "total_papers_analyzed": len(results),
                "total_reviews_processed": total_reviews_processed,
                "average_retry_count": round(avg_retry_count, 2),
                "validation_failures": total_validation_failures,
                "model_used": self.model,
            },
            "papers": results,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(output, output_file.open("w"), indent=2)

        # Print statistics summary
        print(f"\n📊 Statistics:")
        print(f"  Total reviews processed: {total_reviews_processed}")
        print(f"  Average retries per review: {avg_retry_count:.2f}")
        if total_reviews_processed > 0:
            print(f"  Validation failures: {total_validation_failures} ({total_validation_failures/total_reviews_processed*100:.1f}%)")
        else:
            print(f"  Validation failures: {total_validation_failures}")

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
