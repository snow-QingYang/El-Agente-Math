"""
快速测试脚本 - 分析前 5 条评审

这个脚本用于快速测试公式问题分析功能，
只分析前 5 条评审，帮助你快速验证 API 配置和分析效果。
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def quick_test():
    """快速测试分析功能"""

    print("=" * 60)
    print("快速测试 - 公式问题分析")
    print("=" * 60)

    # 检查 API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ 错误: 未找到 OPENAI_API_KEY")
        print("\n请在 .env 文件中设置:")
        print("  OPENAI_API_KEY=your_api_key_here")
        return

    print(f"✓ API Key 已设置: {api_key[:20]}...")

    # 检查数据文件
    input_file = Path("output/ICML_cc_2025_Conference_rejected.json")
    if not input_file.exists():
        print(f"\n❌ 错误: 找不到数据文件 {input_file}")
        print("请先运行爬虫获取数据")
        return

    print(f"✓ 数据文件存在: {input_file}")

    # 读取数据
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 提取前 5 个正式评审
    reviews = []
    for paper in data['submissions']:
        if len(reviews) >= 5:
            break

        for review in paper['reviews']:
            if len(reviews) >= 5:
                break

            if review['invitation'].endswith('Official_Review') or \
               ('/Official_Review' in review['invitation'] and 'Rebuttal' not in review['invitation']):
                reviews.append({
                    'paper_title': paper['content'].get('title', {}).get('value', 'Unknown'),
                    'review_id': review['id'],
                    'content': review['content']
                })

    print(f"✓ 提取了 {len(reviews)} 条评审用于测试\n")

    # 创建 OpenAI 客户端
    client = OpenAI(api_key=api_key)

    # 分析每条评审
    results = []
    for i, review_info in enumerate(reviews, 1):
        print(f"[{i}/5] 分析评审: {review_info['paper_title'][:50]}...")

        # 构建提示词
        summary = review_info['content'].get('summary', 'N/A')
        weaknesses = review_info['content'].get('other_strengths_and_weaknesses', 'N/A')

        prompt = f"""分析以下评审，判断是否指出了论文中的数学公式问题。

评审内容：
{summary}

{weaknesses}

请以 JSON 格式回复，包含：
{{
    "has_formula_issue": true/false,
    "confidence": "high/medium/low",
    "summary": "简短说明"
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 使用便宜的模型测试
                messages=[
                    {"role": "system", "content": "你是评审分析专家，请以 JSON 格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            analysis = json.loads(result_text)

            results.append({
                'paper_title': review_info['paper_title'],
                'has_formula_issue': analysis.get('has_formula_issue', False),
                'confidence': analysis.get('confidence', 'low'),
                'summary': analysis.get('summary', '')
            })

            status = "✓ 有公式问题" if analysis.get('has_formula_issue') else "○ 无公式问题"
            print(f"  {status} (置信度: {analysis.get('confidence', 'low')})")

        except Exception as e:
            print(f"  ❌ 分析失败: {e}")
            results.append({
                'paper_title': review_info['paper_title'],
                'has_formula_issue': False,
                'confidence': 'low',
                'summary': f"错误: {str(e)}"
            })

    # 显示结果
    print(f"\n{'=' * 60}")
    print("测试结果")
    print(f"{'=' * 60}")

    formula_count = sum(1 for r in results if r['has_formula_issue'])
    print(f"总评审数: {len(results)}")
    print(f"有公式问题: {formula_count} ({formula_count/len(results)*100:.0f}%)")

    print("\n详细结果:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['paper_title'][:60]}")
        print(f"   有公式问题: {'是' if result['has_formula_issue'] else '否'}")
        print(f"   置信度: {result['confidence']}")
        if result['summary']:
            print(f"   说明: {result['summary'][:100]}")

    print(f"\n{'=' * 60}")
    print("✓ 测试完成！")
    print(f"{'=' * 60}")
    print("\n如果测试成功，你可以运行完整分析:")
    print("  uv run python analyze_formula_issues.py")


if __name__ == "__main__":
    quick_test()
