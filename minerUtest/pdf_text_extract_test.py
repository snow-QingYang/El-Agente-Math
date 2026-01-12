import pdfplumber
from collections import defaultdict

LINE_TOL = 3  # y 方向容差，值越小越“严格”

with pdfplumber.open("13536_Understanding_Task_Vecto.pdf") as pdf, \
     open("temp.txt", "w", encoding="utf-8") as f:

    for page_id, page in enumerate(pdf.pages):
        f.write(f"\n\n===== Page {page_id} =====\n\n")

        lines = defaultdict(list)

        for c in page.chars:
            # 用 top 近似作为行坐标
            y = round(c["top"] / LINE_TOL) * LINE_TOL
            lines[y].append(c)

        # 按 y 从上到下排序
        for y in sorted(lines.keys()):
            line_chars = sorted(lines[y], key=lambda c: c["x0"])

            prev_x1 = None
            for c in line_chars:
                if prev_x1 is not None:
                    gap = c["x0"] - prev_x1
                    if gap > 2:   # 控制“是否插入空格”
                        f.write(" ")
                f.write(c["text"])
                prev_x1 = c["x1"]

            f.write("\n")
