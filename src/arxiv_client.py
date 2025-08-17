import time
import logging
import urllib.parse
from typing import List, Optional

import requests
import feedparser
import requests
class ArXivClient:
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, retries: int = 5, wait_time_minutes: int = 30):
        self.retries = retries
        self.wait_time = wait_time_minutes * 60

    def fetch_papers(self, keyword: str, max_results: int) -> Optional[List[dict]]:
        logging.info(f"开始为关键词 '{keyword}' 获取论文...")
        
        # 关键词只有一个词时使用 AND，否则使用 OR
        link_operator = "AND" if len(keyword.split()) == 1 else "OR"
        
        # 构造查询URL
        query_keyword = f'"{keyword}"'
        url = (
            f"{self.BASE_URL}?search_query=ti:{query_keyword}+{link_operator}+abs:{query_keyword}"
            f"&max_results={max_results}&sortBy=lastUpdatedDate"
        )
        
        for attempt in range(self.retries):
            try:
                response = requests.get(url)
                response.raise_for_status()  # 如果请求失败 (如 404, 500), 会抛出异常
                
                feed = feedparser.parse(response.content)
                if feed.entries:
                    logging.info(f"成功获取到 {len(feed.entries)} 篇关于 '{keyword}' 的论文。")
                    return feed.entries
                
                logging.warning(f"API 为 '{keyword}' 返回了空列表。尝试次数 {attempt + 1}/{self.retries}...")
                if attempt < self.retries - 1:
                    time.sleep(self.wait_time)

            except requests.exceptions.RequestException as e:
                logging.error(f"请求 arXiv API 失败: {e}。尝试次数 {attempt + 1}/{self.retries}...")
                if attempt < self.retries - 1:
                    time.sleep(self.wait_time)
        
        logging.error(f"重试 {self.retries} 次后，仍未能获取到关于 '{keyword}' 的论文。")
        return None