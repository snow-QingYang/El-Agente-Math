"""Streamlit interface for reviewing formula issues."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


PAPERS_PER_FILE = 500


def get_output_filename(paper_index: int) -> str:
    """Get the output filename based on paper_index."""
    start_range = (paper_index // PAPERS_PER_FILE) * PAPERS_PER_FILE
    end_range = start_range + PAPERS_PER_FILE
    return f"human_review_{start_range}_{end_range}.json"


def load_data(input_file: str) -> dict:
    """Load the filtered issues data."""
    with open(input_file, encoding="utf-8") as f:
        return json.load(f)


def load_review_lookup(lookup_file: str) -> dict:
    """Load the review lookup data indexed by review_id."""
    try:
        with open(lookup_file, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Review lookup file not found: {lookup_file}")
        return {}


def load_reviews(output_dir: str) -> dict:
    """Load all existing reviews."""
    reviews: dict = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return reviews

    for filename in os.listdir(output_dir):
        if filename.startswith("human_review_") and filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, encoding="utf-8") as f:
                file_reviews = json.load(f)
                reviews.update(file_reviews)

    return reviews


def save_review(
    output_dir: str,
    paper_id: str,
    issue_idx: int,
    review_data: dict,
    paper_index: int,
) -> None:
    """Save a single review to the appropriate file."""
    filename = get_output_filename(paper_index)
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            file_reviews = json.load(f)
    else:
        file_reviews = {}

    review_key = f"{paper_id}_{issue_idx}"
    file_reviews[review_key] = review_data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(file_reviews, f, indent=2, ensure_ascii=False)


def main(
    input_file: str | None = None,
    review_lookup_file: str | None = None,
    output_dir: str | None = None,
) -> None:
    """Run the formula review Streamlit interface."""
    st.set_page_config(page_title="Formula Issue Reviewer", layout="wide")
    st.title("Formula Issue Review Interface")
    st.markdown("Review issues with category starting with 'Wrong equation'")

    # Resolve paths from env or defaults
    if input_file is None:
        input_file = os.environ.get(
            "FORMULA_REVIEW_INPUT",
            "packages/openreview-crawler/output/formula_issues_iclr2024_wrong_equations.json",
        )
    if review_lookup_file is None:
        review_lookup_file = os.environ.get(
            "FORMULA_REVIEW_LOOKUP",
            "packages/openreview-crawler/output/review_lookup_iclr2024.json",
        )
    if output_dir is None:
        output_dir = os.environ.get(
            "FORMULA_REVIEW_OUTPUT",
            "packages/openreview-crawler/output/human_reviews",
        )

    if "data" not in st.session_state:
        st.session_state.data = load_data(input_file)
        st.session_state.reviews = load_reviews(output_dir)
        st.session_state.review_lookup = load_review_lookup(review_lookup_file)
        st.session_state.current_paper_idx = 0
        st.session_state.current_issue_idx = 0

    data = st.session_state.data
    papers = data["papers"]
    reviews = st.session_state.reviews
    review_lookup = st.session_state.review_lookup

    total_papers = len(papers)
    total_issues = sum(len(p["issues"]) for p in papers)
    reviewed_count = len(reviews)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", total_papers)
    with col2:
        st.metric("Total Issues", total_issues)
    with col3:
        st.metric("Reviewed Issues", reviewed_count)

    st.progress(reviewed_count / total_issues if total_issues > 0 else 0)

    st.sidebar.header("Navigation")
    paper_idx = st.sidebar.number_input(
        "Paper Index",
        min_value=0,
        max_value=total_papers - 1,
        value=st.session_state.current_paper_idx,
        step=1,
    )

    if paper_idx != st.session_state.current_paper_idx:
        st.session_state.current_paper_idx = paper_idx
        st.session_state.current_issue_idx = 0

    paper = papers[paper_idx]
    paper_id = paper["paper_id"]
    paper_index = paper["paper_index"]

    st.header(f"Paper {paper_idx + 1}/{total_papers}")
    st.subheader(paper["paper_title"])
    st.markdown(f"**Paper ID:** {paper_id}")
    st.markdown(f"**Paper Index:** {paper_index}")
    st.markdown(f"**PDF Link:** [{paper['paper_pdf_link']}]({paper['paper_pdf_link']})")
    st.markdown(f"**Output File:** `{get_output_filename(paper_index)}`")

    st.divider()

    issues = paper["issues"]
    st.markdown(f"### Issues ({len(issues)} total)")

    for issue_idx, issue in enumerate(issues):
        review_key = f"{paper_id}_{issue_idx}"

        with st.container():
            st.markdown(f"#### Issue {issue_idx + 1}/{len(issues)}")
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"**Formula Location:** {', '.join(map(str, issue['formula_location']))}"
                )
                st.markdown(f"**Category:** {issue['category']}")
                st.markdown("**Evidence:**")
                st.info(issue["evidence"])

                review_id = issue.get("review_id")
                if review_id and review_id in review_lookup:
                    with st.expander("View Full Original Review"):
                        review_data = review_lookup[review_id]
                        for section in ("summary", "weaknesses", "questions"):
                            if review_data.get(section):
                                st.markdown(f"### {section.title()}")
                                st.write(review_data[section])
                                st.divider()
                        if review_data.get("info"):
                            st.markdown("### Review Scores")
                            info = review_data["info"]
                            score_cols = st.columns(4)
                            for i, key in enumerate(
                                ("soundness", "presentation", "contribution", "confidence")
                            ):
                                if key in info:
                                    score_cols[i].metric(key.title(), info[key])

            with col2:
                st.markdown("**Review Status:**")
                if review_key in reviews:
                    current_review = reviews[review_key]["status"]
                    if current_review == "good":
                        st.success("Good")
                    else:
                        st.error("Bad")
                else:
                    st.warning("Not reviewed")

                col_good, col_bad = st.columns(2)
                with col_good:
                    if st.button(
                        "Good",
                        key=f"good_{paper_id}_{issue_idx}",
                        use_container_width=True,
                    ):
                        rd = {
                            "paper_id": paper_id,
                            "paper_index": paper_index,
                            "paper_title": paper["paper_title"],
                            "issue_index": issue_idx,
                            "formula_location": issue["formula_location"],
                            "category": issue["category"],
                            "evidence": issue["evidence"],
                            "status": "good",
                        }
                        save_review(output_dir, paper_id, issue_idx, rd, paper_index)
                        st.session_state.reviews[review_key] = rd
                        st.rerun()

                with col_bad:
                    if st.button(
                        "Bad",
                        key=f"bad_{paper_id}_{issue_idx}",
                        use_container_width=True,
                    ):
                        rd = {
                            "paper_id": paper_id,
                            "paper_index": paper_index,
                            "paper_title": paper["paper_title"],
                            "issue_index": issue_idx,
                            "formula_location": issue["formula_location"],
                            "category": issue["category"],
                            "evidence": issue["evidence"],
                            "status": "bad",
                        }
                        save_review(output_dir, paper_id, issue_idx, rd, paper_index)
                        st.session_state.reviews[review_key] = rd
                        st.rerun()

            st.divider()

    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous Paper", disabled=(paper_idx == 0), use_container_width=True):
            st.session_state.current_paper_idx = paper_idx - 1
            st.session_state.current_issue_idx = 0
            st.rerun()
    with col3:
        if st.button(
            "Next Paper",
            disabled=(paper_idx == total_papers - 1),
            use_container_width=True,
        ):
            st.session_state.current_paper_idx = paper_idx + 1
            st.session_state.current_issue_idx = 0
            st.rerun()


if __name__ == "__main__":
    main()
