"""Tiện ích nhỏ: ghi/thay một mục (section) trong model/results.md."""

import config


def update_section(title, body):
    """Thay nội dung mục '## <title>' trong results.md (thêm mới nếu chưa có)."""
    f = config.RESULTS_FILE
    text = f.read_text() if f.exists() else "# Kết quả thí nghiệm\n"
    head = f"## {title}\n"
    if head in text:
        start = text.index(head)
        nxt = text.find("\n## ", start + len(head))
        end = len(text) if nxt == -1 else nxt + 1
        text = text[:start] + head + "\n" + body.strip() + "\n\n" + text[end:]
    else:
        text = text.rstrip() + "\n\n" + head + "\n" + body.strip() + "\n"
    f.write_text(text.rstrip() + "\n")
