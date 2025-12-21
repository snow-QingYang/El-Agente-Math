import json
from pathlib import Path


def test_replace():
    with open("../dataset/bao/2410.10253/2410.10253_consolidated_formulas.json", 'r', encoding='utf-8') as f:
        all_formulas_dict = json.load(f)
        def replace_labels_in_context(text: str) -> str:
            """Replace all formula labels in text with their original formulas."""
            import re
            label_pattern = r'<<FORMULA_\d+>>'
            labels_in_text = re.findall(label_pattern, text)

            result = text
            for label in labels_in_text:
                if label in all_formulas_dict:
                    original_formula = all_formulas_dict[label].get('raw_latex', label)
                    result = result.replace(label, original_formula)

            return result

        labeled_tex_path = Path(f"../dataset/bao/2410.10253/2410.10253_consolidated_labeled.tex")
        labeled_content = labeled_tex_path.read_text(encoding='utf-8', errors='ignore')
        print(all_formulas_dict)
        print(replace_labels_in_context(labeled_content))


if __name__ == "__main__":
    test_replace()