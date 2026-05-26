"""PDF report generation for an analysis result."""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_DISEASE_COLORS = {
    "Cancer": colors.HexColor("#ef4444"),
    "Diabetes": colors.HexColor("#f59e0b"),
    "Alzheimers": colors.HexColor("#a855f7"),
    "Normal": colors.HexColor("#10b981"),
}


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            textColor=colors.HexColor("#1e3a8a"),
            fontSize=20,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            textColor=colors.HexColor("#1e293b"),
            fontSize=13,
            spaceBefore=12,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#64748b"),
        ),
        "mono": ParagraphStyle(
            "mono",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#0f172a"),
        ),
        "verdict": ParagraphStyle(
            "verdict",
            parent=base["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
        ),
    }


def _verdict_box(payload: dict, st: dict) -> Table:
    pred = payload["prediction"]
    cls = pred["predicted_class"]
    color = _DISEASE_COLORS.get(cls, colors.HexColor("#3b82f6"))

    body = [
        [
            Paragraph(
                f'<font color="{color.hexval()}">●</font> Predicted Class', st["body"]
            ),
            Paragraph(
                f'<font name="Helvetica-Bold" size="22" color="{color.hexval()}">{cls}</font>',
                st["body"],
            ),
        ],
        [
            Paragraph("Confidence", st["body"]),
            Paragraph(
                f"<b>{pred['confidence'] * 100:.1f}%</b> · entropy {pred['entropy']:.3f}",
                st["body"],
            ),
        ],
        [
            Paragraph("Sequence Length", st["body"]),
            Paragraph(f"{payload['sequence_length']:,} bases", st["body"]),
        ],
    ]
    table = Table(body, colWidths=[5 * cm, 11 * cm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, color),
            ]
        )
    )
    return table


def _probability_table(payload: dict, st: dict) -> Table:
    probs = payload["prediction"]["probabilities"]
    rows = [["Class", "Probability", "Bar"]]
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for cls, p in sorted_items:
        bar = "█" * int(round(p * 30))
        rows.append([cls, f"{p * 100:.2f}%", bar])
    table = Table(rows, colWidths=[4 * cm, 3 * cm, 9 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (2, 1), (2, -1), "Courier"),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    return table


def _stats_table(payload: dict, st: dict) -> Table:
    s = payload["stats"]
    rows = [
        ["GC content", f"{s['gc_content'] * 100:.2f}%"],
        ["AT content", f"{s['at_content'] * 100:.2f}%"],
        ["Shannon entropy", f"{s['shannon_entropy']:.3f}"],
        ["Complexity score", f"{s['complexity_score']:.3f}"],
        ["Low complexity", "Yes" if s["is_low_complexity"] else "No"],
        [
            "Base counts",
            ", ".join(f"{k}={v}" for k, v in s["base_counts"].items()),
        ],
    ]
    table = Table(rows, colWidths=[5 * cm, 11 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _ood_box(payload: dict, st: dict):
    ood = payload.get("ood")
    if not ood or ood.get("error"):
        return None
    risk = ood["risk"]
    color_map = {
        "low": colors.HexColor("#10b981"),
        "medium": colors.HexColor("#f59e0b"),
        "high": colors.HexColor("#ef4444"),
    }
    color = color_map.get(risk, colors.HexColor("#3b82f6"))
    body = (
        f"<b>Risk:</b> {risk.upper()} &nbsp;&nbsp; "
        f"<b>Novelty:</b> {ood['novelty_score'] * 100:.1f}% &nbsp;&nbsp; "
        f"<b>Nearest similarity:</b> {ood['nearest_similarity']:.3f}<br/>"
        f"{ood['message']}"
    )
    para = Paragraph(body, st["body"])
    table = Table([[para]], colWidths=[16 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.7, color),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _similar_table(payload: dict, st: dict):
    similar = payload.get("similar") or []
    similar = [s for s in similar if not s.get("error")]
    if not similar:
        return None
    rows = [["Rank", "Disease", "Similarity", "Sequence ID", "Preview"]]
    for s in similar[:5]:
        rows.append(
            [
                f"#{s['rank']}",
                s["disease"],
                f"{s['similarity'] * 100:.1f}%",
                s["sequence_id"],
                s["preview"][:40] + "...",
            ]
        )
    table = Table(rows, colWidths=[1.5 * cm, 3 * cm, 2.5 * cm, 3.5 * cm, 5.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("FONTNAME", (4, 1), (4, -1), "Courier"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    return table


def _orf_table(payload: dict, st: dict):
    orfs = (payload.get("orfs") or [])[:8]
    if not orfs:
        return None
    rows = [["Strand", "Frame", "Start", "End", "Length (bp)", "Protein (aa)"]]
    for orf in orfs:
        rows.append(
            [
                orf["strand"],
                str(orf["frame"]),
                str(orf["start"]),
                str(orf["end"]),
                str(orf["length"]),
                str(orf["protein_length"]),
            ]
        )
    table = Table(rows, colWidths=[2 * cm] * 6)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    return table


def _sequence_block(sequence: str, st: dict, *, max_chars: int = 600):
    seq = sequence.upper()
    truncated = len(seq) > max_chars
    shown = seq[:max_chars]
    chunks = [shown[i : i + 60] for i in range(0, len(shown), 60)]
    text = "<br/>".join(chunks)
    if truncated:
        text += f"<br/><i>… truncated, full length {len(seq):,} bases</i>"
    return Paragraph(text, st["mono"])


def build_report(payload: dict, sequence: str) -> bytes:
    """Return PDF bytes for the given /predict payload."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        title="GenomeIQ Analysis Report",
        author="MD Naimur Rashid",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    flow = []
    flow.append(Paragraph("GenomeIQ Analysis Report", st["title"]))
    flow.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            f"Created by MD Naimur Rashid",
            st["small"],
        )
    )
    flow.append(Spacer(1, 8))

    flow.append(_verdict_box(payload, st))

    ood = _ood_box(payload, st)
    if ood is not None:
        flow.append(Spacer(1, 8))
        flow.append(ood)

    flow.append(Paragraph("Class Probabilities", st["h2"]))
    flow.append(_probability_table(payload, st))

    flow.append(Paragraph("Sequence Statistics", st["h2"]))
    flow.append(_stats_table(payload, st))

    similar_tbl = _similar_table(payload, st)
    if similar_tbl is not None:
        flow.append(Paragraph("Closest Known Sequences", st["h2"]))
        flow.append(similar_tbl)

    orf_tbl = _orf_table(payload, st)
    if orf_tbl is not None:
        flow.append(Paragraph("Open Reading Frames", st["h2"]))
        flow.append(orf_tbl)

    explanation = payload.get("explanation")
    if explanation and explanation.get("top_regions"):
        flow.append(Paragraph("Top Saliency Regions", st["h2"]))
        regions = explanation["top_regions"][:6]
        rows = [["Start", "End", "Mean score"]]
        for r in regions:
            rows.append([str(r["start"]), str(r["end"]), f"{r['mean_score']:.3f}"])
        tbl = Table(rows, colWidths=[3 * cm, 3 * cm, 3 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ]
            )
        )
        flow.append(tbl)

    flow.append(Paragraph("Sequence", st["h2"]))
    flow.append(_sequence_block(sequence, st))

    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(
            "Disclaimer: This report is generated by GenomeIQ, a research preview "
            "tool. Predictions and similarity scores are derived from a finite "
            "training corpus and are not intended for clinical decision-making.",
            st["small"],
        )
    )

    doc.build(flow)
    return buf.getvalue()
