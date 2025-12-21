#!/usr/bin/env python3
"""
快速启动脚本 - OpenReview Crawler

这是一个交互式脚本，帮助你快速开始使用 OpenReview Crawler。
"""

from openreview_crawler import OpenReviewCrawler
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*60)
    print("OpenReview Crawler - 快速启动")
    print("="*60)
    print("\n这个脚本将帮助你快速开始爬取 OpenReview 数据。")
    print("\n注意: 并非所有会议都公开被拒绝的论文，")
    print("某些会议可能需要认证或完全不提供拒稿数据。\n")


def get_common_venues():
    """返回常见会议列表"""
    return [
        ("ICML.cc/2025/Conference", "ICML 2025 (国际机器学习大会) - 有公开拒稿数据 ⭐"),
        ("ICLR.cc/2024/Conference", "ICLR 2024 (国际学习表征会议)"),
        ("ICLR.cc/2023/Conference", "ICLR 2023"),
        ("NeurIPS.cc/2023/Conference", "NeurIPS 2023 (神经信息处理系统)"),
        ("ICML.cc/2023/Conference", "ICML 2023"),
        ("custom", "输入自定义会议 ID"),
    ]


def select_venue():
    """让用户选择会议"""
    venues = get_common_venues()

    print("请选择要爬取的会议:\n")

    for i, (venue_id, name) in enumerate(venues, 1):
        print(f"  {i}. {name}")

    print("\n提示: ICML 2025 有公开的拒稿数据，推荐用于测试")

    while True:
        try:
            choice = input("\n请输入选项 (1-6) [默认: 1]: ").strip()

            if not choice:
                choice = "1"

            choice_num = int(choice)

            if 1 <= choice_num <= len(venues):
                venue_id, name = venues[choice_num - 1]

                if venue_id == "custom":
                    custom_id = input("\n请输入会议 ID (例如: ICLR.cc/2024/Conference): ").strip()
                    if custom_id:
                        return custom_id
                    else:
                        print("无效的会议 ID，请重试。")
                        continue
                else:
                    return venue_id
            else:
                print("无效的选项，请重试。")
        except ValueError:
            print("请输入有效的数字。")
        except KeyboardInterrupt:
            print("\n\n操作已取消。")
            exit(0)


def select_options():
    """选择爬取选项"""
    print("\n" + "-"*60)
    print("爬取选项")
    print("-"*60)

    # 是否获取评审
    get_reviews = input("\n是否获取评审内容? (y/n) [默认: y]: ").strip().lower()
    include_reviews = get_reviews != 'n'

    # 输出格式
    output_format = input("\n选择输出格式 (json/csv/both) [默认: json]: ").strip().lower()
    if output_format not in ['json', 'csv', 'both']:
        output_format = 'json'

    # 输出目录
    output_dir = input("\n输出目录 [默认: output]: ").strip()
    if not output_dir:
        output_dir = "output"

    return {
        'include_reviews': include_reviews,
        'output_format': output_format,
        'output_dir': output_dir
    }


def crawl_data(venue_id, options):
    """执行爬取"""
    print("\n" + "="*60)
    print("开始爬取数据...")
    print("="*60 + "\n")

    try:
        # 初始化爬虫
        print("正在初始化 OpenReview 客户端...")
        crawler = OpenReviewCrawler()

        # 获取会议信息
        print(f"\n正在连接到会议: {venue_id}")
        venue_info = crawler.get_venue_info(venue_id)

        if venue_info:
            print(f"✓ 成功连接到会议")

            # 显示会议信息
            if 'content' in venue_info and 'title' in venue_info['content']:
                title = venue_info['content']['title']
                if isinstance(title, dict):
                    title = title.get('value', '')
                print(f"  会议名称: {title}")
        else:
            print("⚠ 警告: 无法获取会议信息，将继续尝试爬取...")

        # 获取被拒绝的论文
        print(f"\n正在获取被拒绝的论文...")
        print(f"(包含评审: {'是' if options['include_reviews'] else '否'})")

        submissions = crawler.get_rejected_papers_with_reviews(
            venue_id=venue_id,
            include_reviews=options['include_reviews']
        )

        if not submissions:
            print("\n" + "="*60)
            print("未找到可访问的被拒绝论文")
            print("="*60)
            print("\n可能的原因:")
            print("  1. 该会议不公开被拒绝的论文")
            print("  2. 需要 OpenReview 账号认证")
            print("  3. 会议 ID 不正确")
            print("\n建议:")
            print("  - 尝试其他会议")
            print("  - 查看会议的数据公开政策")
            print("  - 如有权限，使用认证登录")
            return False

        # 显示统计信息
        print("\n" + "="*60)
        print("爬取完成!")
        print("="*60)
        print(f"\n找到 {len(submissions)} 篇被拒绝的论文")

        if options['include_reviews']:
            total_reviews = sum(len(s.reviews) for s in submissions)
            papers_with_reviews = sum(1 for s in submissions if s.reviews)
            print(f"获得 {total_reviews} 条评审")
            print(f"有评审的论文: {papers_with_reviews}/{len(submissions)}")

        # 保存数据
        output_dir = Path(options['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_venue_name = venue_id.replace('/', '_').replace('.', '_')

        print(f"\n保存数据到: {output_dir}/")

        if options['output_format'] in ['json', 'both']:
            json_path = output_dir / f"{safe_venue_name}_rejected.json"
            crawler.save_to_json(submissions, json_path)
            print(f"  ✓ JSON: {json_path}")

        if options['output_format'] in ['csv', 'both']:
            csv_path = output_dir / f"{safe_venue_name}_rejected.csv"
            crawler.export_to_csv(submissions, csv_path)
            print(f"  ✓ CSV: {csv_path}")

        # 显示示例数据
        if submissions:
            print("\n" + "-"*60)
            print("示例论文:")
            print("-"*60)

            first_paper = submissions[0]

            title = first_paper.content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', 'N/A')

            print(f"\n标题: {title}")
            print(f"作者: {', '.join(first_paper.authors)}")
            print(f"状态: {first_paper.status}")

            if first_paper.reviews:
                print(f"评审数量: {len(first_paper.reviews)}")

                first_review = first_paper.reviews[0]
                rating = first_review.content.get('rating', {})
                if isinstance(rating, dict):
                    rating = rating.get('value', 'N/A')
                print(f"第一条评审评分: {rating}")

        print("\n" + "="*60)
        print("爬取成功完成!")
        print("="*60)

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("错误")
        print("="*60)
        print(f"\n发生错误: {str(e)}")
        print("\n请检查:")
        print("  - 网络连接是否正常")
        print("  - 会议 ID 是否正确")
        print("  - 是否需要认证")

        return False


def main():
    """主函数"""
    print_banner()

    # 选择会议
    venue_id = select_venue()

    # 选择选项
    options = select_options()

    # 确认
    print("\n" + "="*60)
    print("确认爬取设置")
    print("="*60)
    print(f"\n会议 ID: {venue_id}")
    print(f"获取评审: {'是' if options['include_reviews'] else '否'}")
    print(f"输出格式: {options['output_format']}")
    print(f"输出目录: {options['output_dir']}")

    confirm = input("\n确认开始爬取? (y/n) [默认: y]: ").strip().lower()

    if confirm == 'n':
        print("\n操作已取消。")
        return

    # 执行爬取
    success = crawl_data(venue_id, options)

    if success:
        print("\n下一步:")
        print("  1. 查看输出文件")
        print("  2. 运行 examples/advanced_usage.py 进行数据分析")
        print("  3. 阅读 USAGE_GUIDE.md 了解更多用法")
    else:
        print("\n建议:")
        print("  1. 运行 examples/test_connection.py 测试连接")
        print("  2. 尝试其他会议 ID")
        print("  3. 查看 README.md 了解更多信息")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消。再见!")
    except Exception as e:
        print(f"\n\n发生未预期的错误: {str(e)}")
        print("请查看文档或提交 Issue。")
