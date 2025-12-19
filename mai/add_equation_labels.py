import re


def add_equation_labels(tex_string):
    """
    Add labels to equations in a LaTeX string.

    This function automatically adds \label{eq1}, \label{eq2}, etc. to equations
    that don't already have labels. It handles various equation environments including:
    - equation
    - align, align*
    - gather, gather*
    - multline, multline*

    Args:
        tex_string (str): The input LaTeX string

    Returns:
        str: The LaTeX string with labels added to equations
    """
    label_counter = 1
    result = tex_string

    # Pattern to match equation environments
    # Matches: \begin{equation}...\end{equation}, \begin{align}...\end{align}, etc.
    equation_envs = ['equation', 'align', 'gather', 'multline', 'eqnarray']

    for env in equation_envs:
        # Pattern to find environment blocks
        # We need to handle both numbered and starred versions
        for starred in ['', r'\*']:
            env_name = env + starred.replace('\\', '')
            pattern = rf'(\\begin\{{{env}{starred}\}})(.*?)(\\end\{{{env}{starred}\}})'

            def replace_equation(match):
                nonlocal label_counter
                begin_tag = match.group(1)
                content = match.group(2)
                end_tag = match.group(3)

                # Skip starred environments (unnumbered)
                if '*' in env_name:
                    return match.group(0)

                # For align, gather, etc., we need to handle multiple equations (split by \\)
                if env in ['align', 'gather', 'eqnarray']:
                    lines = content.split(r'\\')
                    new_lines = []

                    for line in lines:
                        # Check if this line already has a label
                        if r'\label{' not in line and line.strip() and not r'\nonumber' in line:
                            # Add label before the line break
                            line = line.rstrip() + f' \\label{{eq{label_counter}}}'
                            label_counter += 1
                        new_lines.append(line)

                    content = r'\\'.join(new_lines)
                else:
                    # For single equation environments
                    if r'\label{' not in content:
                        # Add label at the end of the equation content
                        content = content.rstrip() + f' \\label{{eq{label_counter}}}'
                        label_counter += 1

                return begin_tag + content + end_tag

            result = re.sub(pattern, replace_equation, result, flags=re.DOTALL)

    return result


def add_equation_labels_preserve_existing(tex_string):
    """
    Add labels to equations in a LaTeX string, preserving existing labels.

    This function adds sequential labels only to equations that don't already have them.
    Existing labels are kept unchanged.

    Args:
        tex_string (str): The input LaTeX string

    Returns:
        str: The LaTeX string with labels added to unlabeled equations
    """
    # First, find all existing label numbers
    existing_labels = re.findall(r'\\label\{eq(\d+)\}', tex_string)
    if existing_labels:
        label_counter = max(int(num) for num in existing_labels) + 1
    else:
        label_counter = 1

    result = tex_string

    equation_envs = ['equation', 'align', 'gather', 'multline', 'eqnarray']

    for env in equation_envs:
        for starred in ['', r'\*']:
            env_name = env + starred.replace('\\', '')
            pattern = rf'(\\begin\{{{env}{starred}\}})(.*?)(\\end\{{{env}{starred}\}})'

            def replace_equation(match):
                nonlocal label_counter
                begin_tag = match.group(1)
                content = match.group(2)
                end_tag = match.group(3)

                # Skip starred environments (unnumbered)
                if '*' in env_name:
                    return match.group(0)

                if env in ['align', 'gather', 'eqnarray']:
                    lines = content.split(r'\\')
                    new_lines = []

                    for line in lines:
                        if r'\label{' not in line and line.strip() and not r'\nonumber' in line:
                            line = line.rstrip() + f' \\label{{eq{label_counter}}}'
                            label_counter += 1
                        new_lines.append(line)

                    content = r'\\'.join(new_lines)
                else:
                    if r'\label{' not in content:
                        content = content.rstrip() + f' \\label{{eq{label_counter}}}'
                        label_counter += 1

                return begin_tag + content + end_tag

            result = re.sub(pattern, replace_equation, result, flags=re.DOTALL)

    return result


if __name__ == "__main__":
    # Example usage
    sample_tex = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}

This is a simple equation:
\begin{equation}
    E = mc^2
\end{equation}

Here are multiple aligned equations:
\begin{align}
    a + b &= c \\
    d + e &= f \\
    g + h &= i
\end{align}

An unnumbered equation:
\begin{equation*}
    x = y + z
\end{equation*}

Another single equation:
\begin{equation}
    F = ma
\end{equation}

\end{document}
"""

    print("Original LaTeX:")
    print(sample_tex)
    print("\n" + "="*80 + "\n")

    result = add_equation_labels(sample_tex)
    print("LaTeX with labels added:")
    print(result)
