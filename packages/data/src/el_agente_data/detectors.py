"""Formula issue detection using GPT models.

Detectors analyze reviews to find formula-related issues:
- FormulaIssueDetector: Simple binary detection (has issue or not)
- FormulaIssueDetectorICML: Combined detection + classification with location types
- FormulaIssueDetectorNIPS: NeurIPS-specific variant with focused review fields
- FormulaDetailsAnalyzer: Equation-specific detailed analysis with validation
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .prompts import render

if TYPE_CHECKING:
    from pathlib import Path

_DOTENV_PATH = find_dotenv(usecwd=True)
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH)


def _norm_val(val: Any) -> str:
    """Extract string value from possibly dict-wrapped value."""
    if isinstance(val, dict):
        return str(val.get("value", ""))
    return str(val or "")


def _extract_numeric(val: str) -> str:
    """Extract leading integer from a string."""
    m = re.match(r"\s*([0-9]+)", val or "")
    return m.group(1) if m else ""


def _should_use_temperature(model: str) -> bool:
    """Determine if temperature param should be used (not for gpt-5 family)."""
    return not (model.startswith("gpt-5") or model.startswith("chatgpt-5"))


class FormulaIssueDetector:
    """Simple binary detection: does a review mention formula issues?"""

    def __init__(self, model: str = "gpt-5-nano") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", model)

    @staticmethod
    def _get_first(review_content: dict[str, Any], keys: list[str], default: str = "N/A") -> str:
        for k in keys:
            if review_content.get(k):
                val = review_content[k]
                if isinstance(val, dict):
                    val = val.get("value", "")
                return str(val)
        return default

    def _build_prompt(self, review_content: dict[str, Any]) -> str:
        summary = self._get_first(review_content, ["summary"], "N/A")
        strengths = self._get_first(review_content, ["strengths", "claims_and_evidence"], "N/A")
        weaknesses = self._get_first(
            review_content,
            ["weaknesses", "other_strengths_and_weaknesses", "soundness"],
            "N/A",
        )
        questions = self._get_first(review_content, ["questions", "questions_for_authors"], "N/A")
        comment = self._get_first(
            review_content, ["comment", "other_comments_or_suggestions"], "N/A"
        )
        return render(
            "detect_formula_issue.jinja2",
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            questions=questions,
            comment=comment,
        )

    def analyze_review(self, review_content: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(review_content)
        system_prompt = render("detect_formula_issue_system.jinja2")
        resp = self.client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            **({} if not _should_use_temperature(self.model) else {"temperature": 0.1}),
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        result = json.loads(content or "null")
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
    ) -> dict[str, Any]:
        data = json.load(input_file.open())
        papers_in = data["submissions"]

        tasks: list[dict[str, Any]] = []
        for paper in papers_in:
            title = _norm_val(paper["content"].get("title"))
            pdf = paper["content"].get("pdf", "")
            number = paper.get("number")
            pid = paper.get("id")
            for review in paper.get("reviews", []):
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

        results: list[dict[str, Any]] = []

        def _task(task: dict[str, Any]) -> dict[str, Any]:
            analysis = self.analyze_review(task["review"]["content"])
            return {**task, "analysis": analysis}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            fut_map = {executor.submit(_task, t): t for t in tasks}
            for fut in tqdm(
                as_completed(fut_map), total=len(fut_map), desc="Detect formula issues"
            ):
                results.append(fut.result())

        grouped: dict[str, dict[str, Any]] = {}
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
                k: v
                for k, v in content.items()
                if k
                in [
                    "summary",
                    "strengths",
                    "weaknesses",
                    "questions",
                    "soundness",
                    "presentation",
                    "contribution",
                    "comment",
                ]
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
        out: dict[str, Any] = {
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
    """Combined detection + classification with expanded location types."""

    def __init__(self, model: str = "gpt-5-nano", debug_mode: bool = False) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.debug_mode = debug_mode

    @staticmethod
    def _format_review_content(review_content: dict[str, Any]) -> str:
        formatted_parts: list[str] = []
        for key, val in review_content.items():
            actual_value = val["value"] if isinstance(val, dict) and "value" in val else val
            if not actual_value or key in ["code_of_conduct", "overall_recommendation"]:
                continue
            formatted_parts.append(f"{key}: {actual_value}")
        return "\n\n".join(formatted_parts)

    def _build_prompt(self, review_content: dict[str, Any]) -> str:
        review_text = self._format_review_content(review_content)
        return render("detect_and_classify.jinja2", review_text=review_text)

    def analyze_review(self, review_content: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(review_content)
        system_prompt = render("detect_and_classify_system.jinja2")

        if self.debug_mode:
            print("=" * 60, "\nPROMPT SENT TO GPT:\n", "=" * 60)
            print(prompt)

        resp = self.client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            **({} if not _should_use_temperature(self.model) else {"temperature": 0.1}),
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        result = json.loads(content or "null")

        if self.debug_mode:
            print("=" * 60, "\nGPT RESPONSE:\n", "=" * 60)
            print(result)

        has_issue = bool(result.get("has_formula_issue", False))
        formula_errors = result.get("formula_errors", [])
        return {
            "has_formula_issue": has_issue,
            "formula_errors": formula_errors,
            "confidence": "high" if formula_errors else "low",
            "summary": (
                f"Found {len(formula_errors)} formula issue(s)"
                if formula_errors
                else "No formula issues"
            ),
        }

    def run(
        self,
        input_file: Path,
        output_file: Path,
        limit: int | None = None,
        concurrency: int = 10,
    ) -> dict[str, Any]:
        data = json.load(input_file.open())
        papers_in = data["submissions"]

        tasks: list[dict[str, Any]] = []
        for paper in papers_in:
            title = _norm_val(paper["content"].get("title"))
            pdf = paper["content"].get("pdf", "")
            number = paper.get("number")
            pid = paper.get("id")
            for review in paper.get("reviews", []):
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

        results: list[dict[str, Any]] = []

        def _task(task: dict[str, Any]) -> dict[str, Any]:
            analysis = self.analyze_review(task["review"]["content"])
            return {**task, "analysis": analysis}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            fut_map = {executor.submit(_task, t): t for t in tasks}
            for fut in tqdm(
                as_completed(fut_map), total=len(fut_map), desc="Detect formula issues"
            ):
                results.append(fut.result())

        grouped: dict[str, dict[str, Any]] = {}
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
                k: v
                for k, v in content.items()
                if k
                in [
                    "summary",
                    "strengths",
                    "weaknesses",
                    "questions",
                    "soundness",
                    "presentation",
                    "contribution",
                    "comment",
                ]
            }
            formula_errors = item["analysis"].get("formula_errors", [])
            total_formula_errors += len(formula_errors)
            paper_entry["issues"].append(
                {
                    "review_id": item["review"].get("id"),
                    "reviewer": item["review"].get("signatures", ["Unknown"])[0],
                    "review_fields": review_fields,
                    "analysis": item["analysis"],
                    "formula_errors": formula_errors,
                }
            )

        issues = list(grouped.values())
        out: dict[str, Any] = {
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

    def _build_prompt(self, review_content: dict[str, Any]) -> str:
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
        weaknesses = _extract("weaknesses") or "N/A"
        questions = _extract("questions") or "N/A"
        limitations = _extract("limitations") or "N/A"

        review_text = "\n\n".join(
            [
                f"summary: {summary}\n",
                f"strengths_and_weaknesses: strengths:{strengths}\n"
                f"{weaknesses if weaknesses != 'N/A' else ''}\n",
                f"questions: {questions}\n"
                f"{('limitations: ' + limitations) if limitations != 'N/A' else ''}",
            ]
        )
        return render("detect_and_classify.jinja2", review_text=review_text)


class FormulaDetailsAnalyzer:
    """Equation-specific detailed analysis with validation and retry logic."""

    EQUATION_KEYWORDS = ("equation", "eq.", "eq ", "eq(")

    def __init__(self, model: str = "gpt-5-nano") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", model)
        print("using model:", self.model)

    def validate_formula_errors(
        self, errors: list[dict[str, Any]], review_text: str
    ) -> tuple[bool, list[str]]:
        """Validate extracted formula errors against strict rules."""
        location_pattern = re.compile(r"^EQ \([^)]+\)$")
        allowed_categories = {
            "Typo / Symbol misuse",
            "Mathematically wrong",
            "Notational inconsistency",
            "Redundancy",
            "Unclear/Confusing notation",
            "Missing justification/proof",
        }
        allowed_confidence = {"Very certain", "Confident", "Suggestion", "Not sure"}
        normalized_review = " ".join(review_text.split()).lower()

        validation_errors: list[str] = []
        for idx, err in enumerate(errors):
            location = err.get("formula_location", "")
            if not location_pattern.match(location):
                validation_errors.append(
                    f"Error {idx + 1}: Location must be 'EQ (xxx)' format, got '{location}'"
                )
            evidence = err.get("evidence", "")
            normalized_evidence = " ".join(evidence.split()).lower()
            if normalized_evidence:
                evidence_words = set(normalized_evidence.split())
                matching_words = sum(1 for word in evidence_words if word in normalized_review)
                match_ratio = matching_words / len(evidence_words) if evidence_words else 0
                if match_ratio < 0.8:
                    validation_errors.append(
                        f"Error {idx + 1}: Evidence not a direct quote ({match_ratio:.0%} match)"
                    )
            category = err.get("category", "")
            if category not in allowed_categories:
                validation_errors.append(f"Error {idx + 1}: Invalid category '{category}'")
            confidence = err.get("confidence", "")
            if confidence not in allowed_confidence:
                validation_errors.append(f"Error {idx + 1}: Invalid confidence '{confidence}'")

        return (len(validation_errors) == 0, validation_errors)

    def analyze_review(self, review_text: str, max_retries: int = 3) -> tuple[dict[str, Any], int]:
        """Analyze review with validation and retry. Returns (result, retry_count)."""
        prompt = render("analyze_formula_details.jinja2", review_text=review_text)
        system_prompt = render("analyze_formula_details_system.jinja2")

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(  # type: ignore[call-overload]
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **({} if not _should_use_temperature(self.model) else {"temperature": 0.1}),
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                data = json.loads(content or "null")
                errors = data.get("formula_errors", []) or []
                is_valid, validation_errors = self.validate_formula_errors(errors, review_text)

                if is_valid:
                    return ({"formula_errors": errors}, attempt)

                if attempt < max_retries - 1:
                    error_msg = "\n".join(validation_errors)
                    prompt = (
                        f"Your previous response had validation errors:\n\n{error_msg}\n\n"
                        f"Please fix these issues. Remember:\n"
                        f"1. Location MUST be 'EQ (xxx)' format\n"
                        f"2. Evidence MUST be a direct quote\n"
                        f"3. Category and confidence MUST exactly match allowed values\n\n"
                        f"Original review text:\n{review_text}"
                    )
                else:
                    print(
                        f"Validation failed after {max_retries} attempts: {validation_errors[:3]}"
                    )
                    return (
                        {"formula_errors": [], "validation_failed": True},
                        max_retries - 1,
                    )
            except Exception as exc:
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(
                    f"OpenAI call failed after {max_retries} attempts: {exc}"
                ) from exc

        return ({"formula_errors": []}, max_retries - 1)

    def run(
        self,
        input_file: Path,
        output_file: Path,
        limit_papers: int,
        concurrency: int = 10,
    ) -> dict[str, Any]:
        raw = json.load(input_file.open())
        papers = raw.get("issues", [])
        if limit_papers:
            papers = papers[:limit_papers]

        results: list[dict[str, Any]] = []
        total_retry_counts: list[int] = []
        total_validation_failures = 0
        total_reviews_processed = 0

        def process_review(
            review_text: str,
        ) -> tuple[dict[str, Any], int, bool]:
            lines = [
                ln
                for ln in review_text.splitlines()
                if any(k.lower() in ln.lower() for k in self.EQUATION_KEYWORDS)
            ]
            focused_text = "\n".join(lines) if lines else review_text
            try:
                analysis, retry_count = self.analyze_review(focused_text)
            except Exception:
                return ({"formula_errors": []}, 0, False)

            errors = analysis.get("formula_errors", []) or []
            validation_failed = analysis.get("validation_failed", False)
            filtered_errors: list[dict[str, Any]] = []
            for err in errors:
                ev = err.get("evidence", "") or ""
                loc = err.get("formula_location", "") or ""
                cat = err.get("category", "") or ""
                conf = err.get("confidence", "") or "Confident"
                if not ev or not loc or not cat:
                    continue
                filtered_errors.append(
                    {
                        "evidence": ev,
                        "formula_location": loc,
                        "category": cat,
                        "confidence": conf,
                    }
                )
            return ({"formula_errors": filtered_errors}, retry_count, validation_failed)

        def process_paper(paper: dict[str, Any]) -> dict[str, Any]:
            nonlocal total_reviews_processed, total_validation_failures
            paper_result: dict[str, Any] = {
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
                paper_result["issues"].append(
                    {
                        "review_id": issue.get("review_id"),
                        "formula_errors": analysis.get("formula_errors", []),
                    }
                )
            return paper_result

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_paper = {executor.submit(process_paper, paper): paper for paper in papers}
            for future in tqdm(
                as_completed(future_to_paper),
                total=len(future_to_paper),
                desc="Papers",
            ):
                try:
                    res = future.result()
                except Exception as exc:
                    res = {"issues": [], "error": str(exc)}
                results.append(res)

        avg_retry = sum(total_retry_counts) / len(total_retry_counts) if total_retry_counts else 0
        output: dict[str, Any] = {
            "metadata": {
                "total_papers_analyzed": len(results),
                "total_reviews_processed": total_reviews_processed,
                "average_retry_count": round(avg_retry, 2),
                "validation_failures": total_validation_failures,
                "model_used": self.model,
            },
            "papers": results,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(output, output_file.open("w"), indent=2)

        print("\nStatistics:")
        print(f"  Total reviews processed: {total_reviews_processed}")
        print(f"  Average retries per review: {avg_retry:.2f}")
        print(f"  Validation failures: {total_validation_failures}")
        return output
