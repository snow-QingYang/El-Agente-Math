"""
Search for arXiv papers by exact title.

This module provides functionality to:
- Search for papers by exact title match
- Return author list and first version upload date
- Get comprehensive metadata including all version dates
"""

import arxiv
from typing import Optional, Dict, List
from datetime import datetime
import re


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


def get_all_version_dates(arxiv_id: str, max_versions: int = 20) -> List[Dict[str, any]]:
    """
    Get upload dates for all versions of an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345" or "2301.12345v1")
        max_versions: Maximum number of versions to check (default: 20)

    Returns:
        List of dictionaries containing version information:
        [
            {'version': 1, 'date': datetime, 'arxiv_id': '2301.12345v1'},
            {'version': 2, 'date': datetime, 'arxiv_id': '2301.12345v2'},
            ...
        ]
        Returns empty list if paper not found.

    Examples:
        >>> versions = get_all_version_dates("2301.12345")
        >>> for v in versions:
        ...     print(f"v{v['version']}: {v['date']}")
    """
    # Remove version suffix if present (e.g., "2301.12345v1" -> "2301.12345")
    base_id = re.sub(r'v\d+$', '', arxiv_id)

    versions = []
    client = arxiv.Client()

    # Try to fetch each version sequentially until we hit a non-existent version
    for version_num in range(1, max_versions + 1):
        version_id = f"{base_id}v{version_num}"

        try:
            search = arxiv.Search(id_list=[version_id])
            paper = next(client.results(search))

            # The 'updated' field contains the date for this specific version
            versions.append({
                'version': version_num,
                'date': paper.updated,
                'arxiv_id': version_id
            })

        except StopIteration:
            # This version doesn't exist, we've reached the end
            break
        except Exception as e:
            # Some other error occurred, log it but continue
            print(f"Warning: Error fetching version {version_num}: {e}")
            break

    return versions


def get_paper_metainfo(title: str) -> Optional[Dict[str, any]]:
    """
    Get comprehensive metadata for a paper by its title, including all version dates.

    Args:
        title: Exact title of the paper to search for

    Returns:
        Dictionary containing comprehensive paper information if found:
        {
            'arxiv_id': str,  # Base arXiv ID (e.g., "2301.12345")
            'title': str,
            'authors': List[str],
            'first_version_date': datetime,  # Date of v1
            'latest_version_date': datetime,  # Date of latest version
            'url': str,
            'versions': List[Dict]  # List of all versions with dates
                # Each version dict: {'version': int, 'date': datetime, 'arxiv_id': str}
        }
        Returns None if no exact match is found.

    Examples:
        >>> info = get_paper_metainfo("Attention Is All You Need")
        >>> if info:
        ...     print(f"arXiv ID: {info['arxiv_id']}")
        ...     print(f"Number of versions: {len(info['versions'])}")
        ...     for v in info['versions']:
        ...         print(f"  v{v['version']}: {v['date']}")
    """
    # First, search for the paper by title
    basic_info = search_by_exact_title(title)

    if not basic_info:
        return None

    arxiv_id = basic_info['arxiv_id']

    # Get all version dates
    versions = get_all_version_dates(arxiv_id)

    # Build comprehensive metadata
    metainfo = {
        'arxiv_id': arxiv_id,
        'title': basic_info['title'],
        'authors': basic_info['authors'],
        'first_version_date': basic_info['first_version_date'],
        'latest_version_date': versions[-1]['date'] if versions else basic_info['first_version_date'],
        'url': basic_info['url'],
        'versions': versions
    }

    return metainfo
