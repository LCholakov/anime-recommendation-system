import pandas as pd
from typing import Callable


def evaluate(
    recommend_fn: Callable,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 10
) -> dict:
    precisions, recalls, hits = [], [], []

    for user_id, test_group in test_df.groupby("user_id"):
        recs    = recommend_fn(user_id, n=n)
        rec_ids = set(recs["anime_id"])
        test_ids = set(test_group["anime_id"])

        hit       = len(rec_ids & test_ids) > 0
        precision = len(rec_ids & test_ids) / n
        recall    = len(rec_ids & test_ids) / len(test_ids) if test_ids else 0.0

        hits.append(hit)
        precisions.append(precision)
        recalls.append(recall)

    return {
        "hit_rate":  round(sum(hits) / len(hits), 4),
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall":    round(sum(recalls) / len(recalls), 4),
    }


def append_to_tracker(xlsx_path: str, row_data: list, headers: list) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = load_workbook(xlsx_path)
    ws = wb.active

    header_font   = Font(bold=True, color="FFFFFF")
    header_fill   = PatternFill("solid", fgColor="2F5496")
    row_fill      = PatternFill("solid", fgColor="E2EFDA")
    border_side   = Side(style="thin", color="CCCCCC")
    cell_border   = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)
    center        = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # write headers if sheet is empty
    if ws.max_row == 1 and ws.max_column == 1 and ws.cell(1, 1).value is None:
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = cell_border
        ws.row_dimensions[1].height = 30

    next_row = ws.max_row + 1 if ws.cell(1, 1).value is not None else 2

    for col, val in enumerate(row_data, 1):
        cell = ws.cell(row=next_row, column=col, value=val)
        cell.fill = row_fill
        cell.alignment = center
        cell.border = cell_border

    for col in range(1, len(headers) + 1):
        max_len = max(
            len(str(ws.cell(row=r, column=col).value or ""))
            for r in range(1, next_row + 1)
        )
        ws.column_dimensions[get_column_letter(col)].width = min(max_len + 4, 40)

    wb.save(xlsx_path)
