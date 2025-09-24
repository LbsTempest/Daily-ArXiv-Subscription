import logging
import time
from datetime import datetime
from typing import List, Dict

import pytz
import yaml

from .paper import Paper
from .arxiv_client import ArXivClient
from .markdown_generator import MarkdownGenerator

# 设置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkflowRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.arxiv_client = ArXivClient(
            retries=self.config['arxiv']['retries'],
            wait_time_minutes=self.config['arxiv']['wait_time_minutes']
        )
        self.md_generator = MarkdownGenerator()

    def _process_entries(self, entries: List[Dict], target_fields: List[str]) -> List[Paper]:
        papers = []
        for entry in entries:
            tags = [tag['term'] for tag in entry.get('tags', [])]
            
            # 过滤领域
            if target_fields:
                if not any(tag.split('.')[0] in target_fields for tag in tags):
                    continue

            paper = Paper(
                title=" ".join(entry.title.replace("\n", " ").split()),
                link=entry.link,
                abstract=" ".join(entry.summary.replace("\n", " ").split()),
                authors=[author['name'] for author in entry.authors],
                tags=tags,
                comment=entry.get("arxiv_comment", ""),
                date=entry.updated
            )
            papers.append(paper)
        return papers

    def run(self):
        logging.info("开始执行每日论文更新工作流...")
        
        beijing_timezone = pytz.timezone('Asia/Shanghai')
        current_date = datetime.now(beijing_timezone).strftime("%Y-%m-%d")
        
        readme_content = [
            "# Daily Papers",
            "The project automatically fetches the latest papers from arXiv based on keywords.",
            "The subheadings in the README file represent the search keywords.",
            f"Last update: {current_date}\n"
        ]
        
        issue_title = f"Latest Papers - {datetime.now(beijing_timezone).strftime('%B %d, %Y')}"
        issue_content = [
            "---",
            f"title: {issue_title}",
            "labels: documentation",
            "---",
            "**Please check the [Github page](https://github.com/zezhishao/MTS_Daily_ArXiv) for a better reading experience and more papers.**\n"
        ]
        
        # 为邮件创建内容（不包含YAML front matter）
        mail_content = [
            "# 📧 每日ArXiv论文更新\n",
            "**请查看 [Github页面](https://github.com/zezhishao/MTS_Daily_ArXiv) 获得更好的阅读体验和更多论文。**\n"
        ]

        try:
            for keyword in self.config['keywords']:
                readme_content.append(f"## {keyword}")
                issue_content.append(f"## {keyword}")
                mail_content.append(f"## {keyword}")
                
                raw_entries = self.arxiv_client.fetch_papers(
                    keyword,
                    self.config['arxiv']['max_results_per_keyword']
                )

                if not raw_entries:
                    readme_content.append("Failed to fetch papers for this keyword.")
                    issue_content.append("Failed to fetch papers for this keyword.")
                    mail_content.append("Failed to fetch papers for this keyword.")
                    continue
                
                papers = self._process_entries(raw_entries, self.config['arxiv']['target_fields'])
                
                # 为 README 生成表格
                readme_table = self.md_generator.generate_table(papers, self.config['output']['columns'])
                readme_content.append(readme_table + "\n")
                
                # 为 Issue 生成表格
                issue_papers = papers[:self.config['output']['issue_max_papers']]
                issue_table = self.md_generator.generate_table(
                    issue_papers, 
                    self.config['output']['columns'],
                    self.config['output']['issue_ignore_columns']
                )
                issue_content.append(issue_table + "\n")
                
                # 为邮件生成表格（限制论文数量）
                mail_papers = papers[:self.config['output']['issue_max_papers']]
                mail_table = self.md_generator.generate_table(
                    mail_papers,
                    self.config['output']['columns'],
                    self.config['output']['issue_ignore_columns']
                )
                mail_content.append(mail_table + "\n")
                
                time.sleep(self.config['arxiv']['request_delay_seconds'])

            # 如果所有步骤都成功，才写入文件 (原子操作)
            with open(self.config['output']['readme_path'], 'w', encoding='utf-8') as f:
                f.write("\n".join(readme_content))
            logging.info(f"成功更新 {self.config['output']['readme_path']}")

            with open(self.config['output']['issue_template_path'], 'w', encoding='utf-8') as f:
                f.write("\n".join(issue_content))
            logging.info(f"成功更新 {self.config['output']['issue_template_path']}")
            
            # 添加邮件模板底部信息并写入文件
            mail_content.append("\n---\n*本邮件由 GitHub Actions 自动生成*")
            with open(self.config['output']['mail_template_path'], 'w', encoding='utf-8') as f:
                f.write("\n".join(mail_content))
            logging.info(f"成功更新 {self.config['output']['mail_template_path']}")

        except Exception as e:
            logging.critical(f"工作流执行过程中发生严重错误: {e}", exc_info=True)
            # 异常发生时，不会写入任何文件，从而保持了原有文件的完整性，无需备份恢复
            
        logging.info("工作流执行完毕。")