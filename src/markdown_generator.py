from typing import List
from .paper import Paper

def _remove_duplicated_spaces(text: str) -> str:
    return " ".join(text.split())

class MarkdownGenerator:
    @staticmethod
    def _format_paper(paper: Paper, keys_to_include: List[str], ignore_keys: List[str]) -> dict:
        formatted = {}
        # Title and Link are fundamental
        formatted["Title"] = f"**[{_remove_duplicated_spaces(paper.title)}]({paper.link})**"
        
        for key in keys_to_include:
            if key in ["Title", "Link"] or key in ignore_keys:
                continue
            
            value = getattr(paper, key.lower(), "")

            if key == "Date":
                formatted[key] = paper.date.split("T")[0]
            elif key == "Abstract" and value:
                formatted[key] = f"<details><summary>Show</summary><p>{value}</p></details>"
            elif key == "Authors" and value:
                formatted[key] = f"{value[0]} et al."
            elif key == "Tags" and value:
                tags_str = ", ".join(value)
                if len(tags_str) > 10:
                    formatted[key] = f"<details><summary>{tags_str[:5]}...</summary><p>{tags_str}</p></details>"
                else:
                    formatted[key] = tags_str
            elif key == "Comment" and value:
                 if len(value) > 20:
                    formatted[key] = f"<details><summary>{value[:5]}...</summary><p>{value}</p></details>"
                 else:
                    formatted[key] = value
            elif key not in formatted:
                formatted[key] = str(value)

        return formatted

    @staticmethod
    def generate_table(papers: List[Paper], columns: List[str], ignore_keys: List[str] = []) -> str:
        if not papers:
            return "暂无相关论文。"

        formatted_papers = [MarkdownGenerator._format_paper(p, columns, ignore_keys) for p in papers]

        # Generate header
        header_cols = [f"**{col}**" for col in columns if col not in ignore_keys]
        header = f"| {' | '.join(header_cols)} |"
        separator = f"| {' | '.join(['---'] * len(header_cols))} |"

        # Generate body
        body_rows = []
        for fp in formatted_papers:
            row_values = [fp.get(col, "") for col in columns if col not in ignore_keys]
            body_rows.append(f"| {' | '.join(row_values)} |")

        return "\n".join([header, separator] + body_rows)