"""Streamlit interface for reviewing check-bench results.

Shows evidence vs agent analysis side-by-side.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import streamlit as st

DEFAULT_BENCH_DIR = os.path.join("output", "bench", "neurips2025-gpt-5.2")


def _get_dir_from_args() -> str | None:
    if "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def _prompt_dir() -> str:
    prompt = f"Benchmark directory (default {DEFAULT_BENCH_DIR}): "
    try:
        value = input(prompt).strip()
    except EOFError:
        value = ""
    return value or DEFAULT_BENCH_DIR


def resolve_dir() -> str:
    bench_dir = os.getenv("BENCH_DIR")
    if not bench_dir:
        bench_dir = _get_dir_from_args() or _prompt_dir()
        os.environ["BENCH_DIR"] = bench_dir
    return bench_dir


def load_report(report_dir: str) -> dict:
    report_path = os.path.join(report_dir, "consistency_report.json")
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """Run the bench review Streamlit interface."""
    st.set_page_config(page_title="Bench Review Interface", layout="wide")

    if "report_dir" not in st.session_state:
        st.session_state.report_dir = resolve_dir()

    st.title("Bench Review Interface")
    st.caption("Compare reviewer evidence with agent analysis")

    st.sidebar.header("Settings")
    report_dir = st.sidebar.text_input(
        "Benchmark directory",
        value=st.session_state.report_dir,
        help="Directory containing consistency_report.json",
    )

    if report_dir != st.session_state.report_dir:
        st.session_state.report_dir = report_dir
        st.session_state.data = None
        st.session_state.current_idx = 0

    report_path = os.path.join(st.session_state.report_dir, "consistency_report.json")
    if not os.path.exists(report_path):
        st.error(f"Report not found: {report_path}")
        return

    if "data" not in st.session_state or st.session_state.data is None:
        st.session_state.data = load_report(st.session_state.report_dir)
        st.session_state.current_idx = 0

    data = st.session_state.data
    details = data.get("details", [])
    metadata = data.get("metadata", {})

    st.sidebar.divider()
    st.sidebar.header("Summary")
    for key in (
        "total_reports",
        "valid_comparisons",
        "matches",
        "mismatches",
        "errors",
        "skipped",
    ):
        st.sidebar.markdown(f"**{key}:** {metadata.get(key, 'N/A')}")
    st.sidebar.markdown(f"**Match Rate:** {metadata.get('match_rate', 'N/A')}%")

    st.sidebar.divider()
    st.sidebar.header("Filter")
    filter_option = st.sidebar.radio(
        "Show", ["All", "Matched", "Mismatched", "Errors", "Skipped"], index=0
    )

    def filter_entries(entries: list[dict]) -> list[dict]:
        if filter_option == "Matched":
            return [e for e in entries if e.get("status") == "success" and e.get("matched") is True]
        if filter_option == "Mismatched":
            return [
                e for e in entries if e.get("status") == "success" and e.get("matched") is False
            ]
        if filter_option == "Errors":
            return [e for e in entries if e.get("status") == "error"]
        if filter_option == "Skipped":
            return [e for e in entries if e.get("status") == "skipped"]
        return entries

    filtered = filter_entries(details)
    if not filtered:
        st.warning(f"No entries for filter: {filter_option}")
        return

    total_entries = len(filtered)
    current_idx = (
        st.sidebar.number_input(
            f"Entry (1-{total_entries})",
            min_value=1,
            max_value=total_entries,
            value=min(st.session_state.current_idx + 1, total_entries),
            step=1,
        )
        - 1
    )
    st.session_state.current_idx = current_idx

    entry = filtered[current_idx]
    status = entry.get("status", "unknown")

    st.subheader(f"Entry {current_idx + 1} / {total_entries}")
    if status == "success":
        matched = entry.get("matched")
        if matched is True:
            st.success("Matched")
        elif matched is False:
            st.error("Mismatched")
        else:
            st.info("Unknown match status")
    elif status == "error":
        st.error("Error")
    elif status == "skipped":
        st.warning("Skipped")
    else:
        st.info(f"Status: {status}")

    paper_id = entry.get("paper_id")
    issue_index = entry.get("issue_index")
    if paper_id is not None:
        st.markdown(f"**Paper ID:** `{paper_id}`")
    if issue_index is not None:
        st.markdown(f"**Issue Index:** {issue_index}")

    if entry.get("reason"):
        st.markdown("**Consistency Check Reason:**")
        st.info(entry.get("reason"))

    if status == "success":
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Reviewer Evidence**")
            st.info(entry.get("evidence", ""))
        with col_right:
            st.markdown("**Agent Analysis**")
            st.markdown(entry.get("analysis_snippet", ""))
    else:
        st.markdown("**Details**")
        st.code(json.dumps(entry, indent=2, ensure_ascii=False))

    st.divider()
    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.current_idx = max(0, current_idx - 1)
            st.rerun()
    with col3:
        if st.button("Next", disabled=(current_idx == total_entries - 1), use_container_width=True):
            st.session_state.current_idx = min(total_entries - 1, current_idx + 1)
            st.rerun()

    st.caption(f"Loaded from: {st.session_state.report_dir}")

    st.sidebar.divider()
    st.sidebar.header("Export")
    if st.sidebar.button("Download Current Entry", use_container_width=True):
        entry_json = json.dumps(entry, indent=2, ensure_ascii=False)
        st.sidebar.download_button(
            label="Click to Download",
            data=entry_json,
            file_name=f"bench_entry_{paper_id}_{issue_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
