"""
Search for arXiv papers by exact title.

This module provides functionality to:
- Search for papers by exact title match
- Return author list and first version upload date
"""

import arxiv
from typing import Optional, Dict, List
from datetime import datetime


def search_by_exact_title(title: str) -> Optional[Dict[str, any]]:
    """
    Search for an arXiv paper with an exact title match.

    Args:
        title: Exact title of the paper to search for

    Returns:
        Dictionary containing paper information if found:
        {
            'arxiv_id': str,
            'title': str,
            'authors': List[str],  # List of author names
            'first_version_date': datetime,  # Upload date of first version
            'url': str  # Link to the paper
        }
        Returns None if no exact match is found.

    Examples:
        >>> result = search_by_exact_title("Attention Is All You Need")
        >>> if result:
        ...     print(f"Authors: {', '.join(result['authors'])}")
        ...     print(f"First uploaded: {result['first_version_date']}")
    """
    # Search arXiv with the title query
    # Using title search with quotes for more precise matching
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=10,  # Get a few results to check for exact match
        sort_by=arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client()

    try:
        for paper in client.results(search):
            # Check for exact title match (case-insensitive)
            if paper.title.strip().lower() == title.strip().lower():
                # Extract author names
                authors = [author.name for author in paper.authors]

                # Get first version date from published date
                # The 'published' field represents the first version upload date
                first_version_date = paper.published

                return {
                    'arxiv_id': paper.entry_id.split('/abs/')[-1],
                    'title': paper.title,
                    'authors': authors,
                    'first_version_date': first_version_date,
                    'url': paper.entry_id
                }

        # No exact match found
        return None

    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return None


def search_papers_by_titles(titles: List[str]) -> Dict[str, Optional[Dict[str, any]]]:
    """
    Search for multiple papers by their exact titles.

    Args:
        titles: List of paper titles to search for

    Returns:
        Dictionary mapping each title to its paper information (or None if not found)

    Examples:
        >>> titles = ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers"]
        >>> results = search_papers_by_titles(titles)
        >>> for title, info in results.items():
        ...     if info:
        ...         print(f"{title}: {len(info['authors'])} authors")
    """
    results = {}

    for title in titles:
        print(f"Searching for: {title}")
        result = search_by_exact_title(title)
        results[title] = result

        if result:
            print(f"  Found: {result['arxiv_id']}")
        else:
            print(f"  Not found")

    return results
