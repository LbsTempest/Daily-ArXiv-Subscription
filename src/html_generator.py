from typing import List
from .paper import Paper

def _remove_duplicated_spaces(text: str) -> str:
    return " ".join(text.split())

class HtmlGenerator:
    @staticmethod
    def _format_paper(paper: Paper, keys_to_include: List[str], ignore_keys: List[str]) -> dict:
        formatted = {}
        # Title and Link are fundamental
        formatted["Title"] = f'<a href="{paper.link}" style="color: #0066cc; text-decoration: none;"><strong>{_remove_duplicated_spaces(paper.title)}</strong></a>'
        
        for key in keys_to_include:
            if key in ["Title", "Link"] or key in ignore_keys:
                continue
            
            value = getattr(paper, key.lower(), "")

            if key == "Date":
                formatted[key] = paper.date.split("T")[0]
            elif key == "Abstract" and value:
                formatted[key] = f'<span style="color: #666; font-size: 0.9em;">{value[:100]}...</span>'
            elif key == "Authors" and value:
                formatted[key] = f'<em style="color: #555;">{value[0]} et al.</em>'
            elif key == "Tags" and value:
                tags_str = ", ".join(value[:3])  # 只显示前3个标签
                formatted[key] = f'<span style="font-size: 0.8em; color: #888;">{tags_str}</span>'
            elif key == "Comment" and value:
                if len(value) > 50:
                    formatted[key] = f'<span style="color: #666; font-size: 0.9em;">{value[:50]}...</span>'
                else:
                    formatted[key] = f'<span style="color: #666; font-size: 0.9em;">{value}</span>'
            elif key not in formatted:
                formatted[key] = str(value)

        return formatted

    @staticmethod
    def generate_html_table(papers: List[Paper], columns: List[str], ignore_keys: List[str] = []) -> str:
        if not papers:
            return '<p style="color: #666; font-style: italic;">暂无相关论文。</p>'

        formatted_papers = [HtmlGenerator._format_paper(p, columns, ignore_keys) for p in papers]

        # Generate table
        table_style = '''
        <table style="
            width: 100%; 
            border-collapse: collapse; 
            margin: 10px 0;
            font-family: Arial, sans-serif;
        ">
        '''
        
        # Generate header
        header_cols = [col for col in columns if col not in ignore_keys]
        header = '<tr style="background-color: #f5f5f5;">'
        for col in header_cols:
            header += f'<th style="padding: 12px 8px; text-align: left; border: 1px solid #ddd; font-weight: bold; color: #333;">{col}</th>'
        header += '</tr>'

        # Generate body
        body_rows = []
        for i, fp in enumerate(formatted_papers):
            bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
            row = f'<tr style="background-color: {bg_color};">'
            for col in header_cols:
                cell_value = fp.get(col, "")
                row += f'<td style="padding: 10px 8px; border: 1px solid #ddd; vertical-align: top;">{cell_value}</td>'
            row += '</tr>'
            body_rows.append(row)

        return table_style + header + ''.join(body_rows) + '</table>'

    @staticmethod
    def generate_email_content(content_sections: List[dict]) -> str:
        """
        生成完整的HTML邮件内容
        content_sections: [{"title": "Section Title", "content": "HTML content"}, ...]
        """
        html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>每日ArXiv论文更新</title>
</head>
<body style="
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #ffffff;
">
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    ">
        <h1 style="margin: 0; font-size: 28px;">📧 每日ArXiv论文更新</h1>
        <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;">
            最新的学术论文，直达您的邮箱
        </p>
    </div>
    
    <div style="
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        border-left: 4px solid #007bff;
    ">
        <p style="margin: 0; color: #495057;">
            <strong>💡 提示：</strong>
            请查看 <a href="https://github.com/LbsTempest/Daily-ArXiv-Subscription" style="color: #007bff; text-decoration: none;">Github页面</a> 
            获得更好的阅读体验和更多论文。
        </p>
    </div>
'''
        
        for section in content_sections:
            html += f'''
    <div style="margin-bottom: 35px;">
        <h2 style="
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 22px;
        ">{section["title"]}</h2>
        {section["content"]}
    </div>
'''
        
        html += '''
    <div style="
        margin-top: 40px;
        padding: 20px;
        background-color: #e9ecef;
        border-radius: 8px;
        text-align: center;
        color: #6c757d;
        font-size: 14px;
    ">
        <p style="margin: 0;">
            本邮件由 <strong>GitHub Actions</strong> 自动生成<br>
            如有问题，请访问项目仓库或联系管理员
        </p>
    </div>
</body>
</html>
        '''
        
        return html