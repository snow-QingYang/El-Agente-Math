"""
Consolidate multi-file LaTeX projects into a single TeX file.

This module handles:
- Extracting ZIP archives containing LaTeX source files
- Identifying the main .tex file
- Resolving \\input{} and \\include{} commands
- Merging all files into a single consolidated .tex file
- Preserving context around mathematical formulas
"""

from pathlib import Path
from typing import Optional, Set
import zipfile
import tarfile
import shutil
import re
from pylatexenc.latexwalker import LatexWalker, LatexWalkerError
from pylatexenc.latexnodes.nodes import LatexMacroNode


def extract_tex_source(zip_path: Path, output_dir: Path) -> Path:
    """
    Extract arXiv TeX source archive (supports .zip, .tar.gz, .tar).

    Args:
        zip_path: Path to the archive file containing LaTeX source
        output_dir: Directory to extract files into

    Returns:
        Path to the extraction directory

    Raises:
        FileNotFoundError: If archive file doesn't exist
        zipfile.BadZipFile: If file is not a valid archive

    Examples:
        >>> extract_dir = extract_tex_source(Path("paper.tar.gz"), Path("./extracted"))
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    # Validate file exists
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive file not found: {zip_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine archive type and extract
    if zip_path.suffix == '.gz' or zip_path.name.endswith('.tar.gz'):
        # Handle tar.gz files (most common for arXiv)
        print(f"Extracting tar.gz archive: {zip_path.name}")
        with tarfile.open(zip_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    elif zip_path.suffix == '.tar':
        # Handle tar files
        print(f"Extracting tar archive: {zip_path.name}")
        with tarfile.open(zip_path, 'r') as tar:
            tar.extractall(path=output_dir)
    elif zip_path.suffix == '.zip':
        # Handle zip files
        print(f"Extracting zip archive: {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(path=output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {zip_path.suffix}")

    print(f"Extracted to: {output_dir}")
    return output_dir


def find_main_tex_file(tex_dir: Path) -> Path:
    """
    Identify the main .tex file in a LaTeX project.

    Args:
        tex_dir: Directory containing extracted LaTeX files

    Returns:
        Path to the main .tex file

    Raises:
        FileNotFoundError: If no .tex files found
        ValueError: If cannot determine main file

    Notes:
        Heuristics for identifying main file:
        - File named main.tex, paper.tex, or article.tex
        - File containing \\documentclass
        - Largest .tex file
        - File not referenced by \\input or \\include

    Examples:
        >>> main_file = find_main_tex_file(Path("./extracted"))
    """
    tex_dir = Path(tex_dir)

    # Find all .tex files
    tex_files = list(tex_dir.rglob("*.tex"))

    if not tex_files:
        raise FileNotFoundError(f"No .tex files found in {tex_dir}")

    print(f"Found {len(tex_files)} .tex files")

    # Heuristic 1: Check for common main file names
    common_names = ['main.tex', 'paper.tex', 'article.tex', 'ms.tex']
    for name in common_names:
        for tex_file in tex_files:
            if tex_file.name == name:
                print(f"Found main file by name: {tex_file.name}")
                return tex_file

    # Heuristic 2: Find files with \documentclass
    files_with_documentclass = []
    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            if '\\documentclass' in content:
                files_with_documentclass.append(tex_file)
        except Exception as e:
            print(f"Warning: Could not read {tex_file}: {e}")

    if len(files_with_documentclass) == 1:
        print(f"Found main file by \\documentclass: {files_with_documentclass[0].name}")
        return files_with_documentclass[0]
    elif len(files_with_documentclass) > 1:
        # If multiple files have \documentclass, pick the largest one
        main_file = max(files_with_documentclass, key=lambda f: f.stat().st_size)
        print(f"Multiple files with \\documentclass, choosing largest: {main_file.name}")
        return main_file

    # Heuristic 3: Fall back to largest .tex file
    main_file = max(tex_files, key=lambda f: f.stat().st_size)
    print(f"Using largest .tex file as main: {main_file.name}")
    return main_file


def _process_tex_file(
    tex_file: Path,
    base_dir: Path,
    visited_files: Set[Path],
    depth: int = 0
) -> str:
    """
    Recursively process a LaTeX file, substituting \\input and \\include commands.

    Args:
        tex_file: Path to the .tex file to process
        base_dir: Base directory for resolving relative paths
        visited_files: Set of already visited files (for circular dependency detection)
        depth: Current recursion depth (for debugging)

    Returns:
        Consolidated LaTeX content as string

    Raises:
        RecursionError: If circular dependency detected
        FileNotFoundError: If included file not found
    """
    # Circular dependency check
    tex_file = tex_file.resolve()
    if tex_file in visited_files:
        raise RecursionError(f"Circular dependency detected: {tex_file}")

    visited_files.add(tex_file)
    indent = "  " * depth
    print(f"{indent}Processing: {tex_file.name}")

    # Read file content
    try:
        content = tex_file.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"{indent}Warning: Could not read {tex_file}: {e}")
        return f"% Error reading file: {tex_file}\n"

    # Parse LaTeX with pylatexenc
    try:
        walker = LatexWalker(content)
        nodelist, _, _ = walker.get_latex_nodes(pos=0)
    except (LatexWalkerError, Exception) as e:
        # If parsing fails, fall back to regex-based approach
        print(f"{indent}Warning: pylatexenc parsing failed, using regex fallback")
        return _process_tex_file_regex(content, tex_file.parent, visited_files, depth)

    # Find all \input and \include commands
    replacements = []  # List of (start_pos, end_pos, replacement_content)

    def find_input_commands(nodes, parent_pos=0):
        """Recursively find input/include commands in node tree"""
        if not nodes:
            return

        for node in nodes:
            if isinstance(node, LatexMacroNode):
                if node.macroname in ['input', 'include']:
                    # Extract filename from arguments
                    filename = _extract_filename_from_node(node)
                    if filename:
                        # Resolve file path
                        file_path = _resolve_tex_path(filename, tex_file.parent, base_dir)

                        if file_path and file_path.exists():
                            # Recursively process the included file
                            included_content = _process_tex_file(
                                file_path, base_dir, visited_files, depth + 1
                            )

                            # Add replacement
                            replacements.append((
                                node.pos,
                                node.pos + node.len,
                                f"\n% BEGIN INCLUDE: {filename}\n{included_content}\n% END INCLUDE: {filename}\n"
                            ))
                        else:
                            print(f"{indent}  Warning: Could not find file: {filename}")

            # Recursively check child nodes
            if hasattr(node, 'nodelist') and node.nodelist:
                find_input_commands(node.nodelist, node.pos)

    find_input_commands(nodelist)

    # Apply replacements (from end to beginning to maintain positions)
    result = content
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    return result


def _extract_filename_from_node(node: LatexMacroNode) -> Optional[str]:
    """Extract filename from \input or \include macro node"""
    try:
        if node.nodeargd and node.nodeargd.argnlist:
            for arg in node.nodeargd.argnlist:
                if arg and hasattr(arg, 'nodelist'):
                    # Extract text from argument nodes
                    filename = ''.join([
                        n.chars if hasattr(n, 'chars') else ''
                        for n in arg.nodelist if hasattr(n, 'chars')
                    ])
                    if filename:
                        return filename.strip()
    except Exception:
        pass
    return None


def _resolve_tex_path(filename: str, current_dir: Path, base_dir: Path) -> Optional[Path]:
    """Resolve a .tex file path, handling various cases"""
    # Try as-is
    candidates = [
        current_dir / filename,
        base_dir / filename,
    ]

    # Try with .tex extension
    if not filename.endswith('.tex'):
        candidates.extend([
            current_dir / f"{filename}.tex",
            base_dir / f"{filename}.tex",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _process_tex_file_regex(
    content: str,
    current_dir: Path,
    visited_files: Set[Path],
    depth: int
) -> str:
    """Fallback regex-based processor for when pylatexenc fails"""
    # Pattern for \input{file} and \include{file}
    pattern = r'\\(?:input|include)\s*\{([^}]+)\}'

    def replace_match(match):
        filename = match.group(1).strip()
        file_path = _resolve_tex_path(filename, current_dir, current_dir)

        if file_path and file_path.exists():
            included_content = _process_tex_file(file_path, current_dir, visited_files, depth + 1)
            return f"\n% BEGIN INCLUDE: {filename}\n{included_content}\n% END INCLUDE: {filename}\n"
        else:
            return match.group(0)  # Keep original if file not found

    return re.sub(pattern, replace_match, content)


def consolidate_tex_project(tex_dir: Path, output_file: Path) -> Path:
    """
    Merge multi-file LaTeX project into a single consolidated .tex file.

    This function recursively resolves \\input{} and \\include{} commands,
    replacing them with the actual file contents. This preserves the context
    around mathematical formulas, making it easier to extract formulas with
    their surrounding text.

    Args:
        tex_dir: Directory containing LaTeX project files
        output_file: Path for the consolidated output .tex file

    Returns:
        Path to the consolidated .tex file

    Raises:
        FileNotFoundError: If main file or included files not found
        RecursionError: If circular includes detected

    Notes:
        - Preserves line breaks and formatting
        - Handles both \\input{file} and \\include{file} commands
        - Resolves relative paths correctly
        - Keeps comments for context
        - Handles .tex extension variations (with/without)

    Examples:
        >>> consolidated = consolidate_tex_project(
        ...     Path("./extracted"),
        ...     Path("./consolidated.tex")
        ... )
    """
    tex_dir = Path(tex_dir)
    output_file = Path(output_file)

    print("\n" + "="*70)
    print("Consolidating LaTeX Project")
    print("="*70)

    # Find the main .tex file
    main_file = find_main_tex_file(tex_dir)
    print(f"\nMain file: {main_file}")

    # Process the main file recursively
    visited_files: Set[Path] = set()
    print(f"\nProcessing files:")
    consolidated_content = _process_tex_file(
        main_file,
        tex_dir,
        visited_files,
        depth=0
    )

    # Write consolidated content
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(consolidated_content, encoding='utf-8')

    print(f"\n" + "="*70)
    print(f"Consolidation complete!")
    print(f"Processed {len(visited_files)} files")
    print(f"Output: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
    print("="*70)

    return output_file


def process_arxiv_source(zip_path: Path, output_file: Path) -> Path:
    """
    Complete pipeline: extract archive and consolidate into single .tex file.

    Args:
        zip_path: Path to arXiv source archive (.tar.gz, .zip, etc.)
        output_file: Path for consolidated output .tex file

    Returns:
        Path to the consolidated .tex file

    Examples:
        >>> consolidated = process_arxiv_source(
        ...     Path("./downloads/paper_source.tar.gz"),
        ...     Path("./output/paper_consolidated.tex")
        ... )
    """
    import tempfile

    zip_path = Path(zip_path)
    output_file = Path(output_file)

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract archive
        print(f"\n{'='*70}")
        print(f"Processing arXiv Source: {zip_path.name}")
        print(f"{'='*70}\n")

        extract_dir = extract_tex_source(zip_path, temp_path / "extracted")

        # Consolidate the project
        consolidated = consolidate_tex_project(extract_dir, output_file)

    return consolidated
