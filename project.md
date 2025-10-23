# 🧮 El Agente Math

**El Agente Math** is a command-line interface (CLI) tool designed to **extract, organize, and verify mathematical content** from scientific documents such as LaTeX or Markdown papers.

---

## 🎯 Objective

To build an intelligent CLI agent capable of:
- Extracting **mathematical symbols and formulas** from source files.
- Linking **symbol definitions** with their usages.
- Verifying **mathematical consistency** across equations and sections.

---

## ⚙️ Core Features (Planned)

| Stage | Feature | Description |
|:------|:---------|:-------------|
| **Milestone 1** | **Symbol Indexing** | Parse all math environments and create a symbol index including file paths and positions. |
| **Milestone 2** | **Definition Extraction** | Detect and extract symbol definitions, noting dependencies (e.g., `E = mc^2` defines `E`). |
| **Milestone 3** | **Equation Linking** | Build a dependency graph connecting formulas and symbol definitions. |
| **Milestone 4** | **Equation Checking** | Verify correctness or logical consistency of equations using the defined symbols. |
| **Milestone 5** | **Error Reporting** | Detect inconsistencies, undefined symbols, or circular dependencies. |

---

## 🧩 CLI Usage

### Basic Commands
| Command | Description |
|:--------|:-------------|
| `mai --index <path>` | Index all math symbols and formulas in the specified directory. |
| `mai --defs` | Generate definitions of all detected symbols. |
| `mai --check` | Check all equations for logical consistency using known definitions. |
| `mai --report` | Generate a markdown or JSON report of inconsistencies. |

## 🧩 CLI Usage

### Basic Commands
| Command | Description |
|:--------|:-------------|
| `mai --index <path>` | Index all math symbols and formulas in the specified directory. |
| `mai --defs` | Generate definitions of all detected symbols. |
| `mai --check` | Check all equations for logical consistency using known definitions. |
| `mai --report` | Generate a markdown or JSON report of inconsistencies. |

---

## 🧭 Example Workflow

Assume you have a LaTeX project organized as follows:

my_paper/
│
├── main.tex
├── intro.tex
├── method.tex
├── appendix.tex
├── refs.bib
└── figures/

# Step 1: Index one LaTeX project
mai --index ./

# This scans all .tex files under the current directory
# and creates ./reports/math_index.md listing all symbols and equations.

# Step 2: Extract definitions
mai --defs

# This identifies where symbols (e.g., E, m, c) are defined
# and stores the results in ./reports/symbol_definitions.md

# Step 3: Check equations
mai --check

# The tool compares each equation with known definitions,
# detecting undefined or inconsistent expressions.
# Results are saved in ./reports/math_check_report.md