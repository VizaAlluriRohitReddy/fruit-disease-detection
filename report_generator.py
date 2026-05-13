from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
import os
import datetime


def generate_pdf_report(
    predicted_class,
    confidence,
    disease_info,
    predictions,
    save_path="results/disease_report.pdf"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 90

    # Color palette
    PRIMARY = colors.HexColor('#00d4aa')
    SECONDARY = colors.HexColor('#4facfe')
    ACCENT = colors.HexColor('#a78bfa')
    DARK = colors.HexColor('#0d1117')
    CARD = colors.HexColor('#0f172a')
    TEXT = colors.HexColor('#e2e8f0')
    MUTED = colors.HexColor('#64748b')
    SUCCESS = colors.HexColor('#22c55e')
    DANGER = colors.HexColor('#ef4444')
    WARNING = colors.HexColor('#f59e0b')

    # Styles
    title_style = ParagraphStyle(
        'Title',
        fontSize=22,
        textColor=PRIMARY,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=28
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        fontSize=10,
        textColor=MUTED,
        spaceAfter=3,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    section_style = ParagraphStyle(
        'Section',
        fontSize=11,
        textColor=SECONDARY,
        spaceBefore=14,
        spaceAfter=8,
        fontName='Helvetica-Bold',
        letterSpacing=1
    )
    body_style = ParagraphStyle(
        'Body',
        fontSize=9.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=5,
        fontName='Helvetica',
        leading=15
    )
    label_style = ParagraphStyle(
        'Label',
        fontSize=9,
        textColor=MUTED,
        fontName='Helvetica-Bold'
    )

    elements = []

    # ── HEADER ──────────────────────────────────────────
    # College logo if exists
    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'veltech_logo.png'
    )

    if os.path.exists(logo_path):
        logo = RLImage(logo_path, width=1.2*inch,
                       height=0.8*inch)
        header_data = [[logo,
                        Paragraph(
                            "Fruit Disease Detection System",
                            title_style
                        ),
                        '']]
        header_table = Table(
            header_data,
            colWidths=[1.4*inch, W-2.8*inch, 1.4*inch]
        )
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ]))
        elements.append(header_table)
    else:
        elements.append(Paragraph(
            "Fruit Disease Detection System",
            title_style
        ))

    elements.append(Paragraph(
        "Naan-Nee Architecture — Hybrid ResNet50 + Vision Transformer — AI-Powered Agricultural Analysis",
        subtitle_style
    ))
    elements.append(Spacer(1, 4))

    # Developer names
    dev_style = ParagraphStyle(
        'Dev',
        fontSize=9,
        textColor=ACCENT,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceAfter=2
    )
    elements.append(Paragraph(
        "P. Kavya Sri  &  V. Alluri Rohit Reddy",
        dev_style
    ))
    elements.append(Paragraph(
        "Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology",
        subtitle_style
    ))
    elements.append(Paragraph(
        "B.Tech Computer Science and Engineering — Batch 2022–2026 — Final Year Major Project",
        subtitle_style
    ))

    elements.append(Spacer(1, 8))

    # Gradient line
    elements.append(HRFlowable(
        width="100%", thickness=2,
        color=PRIMARY, spaceAfter=4
    ))

    # Date
    now = datetime.datetime.now()
    date_str = now.strftime("%d %B %Y")
    time_str = now.strftime("%I:%M %p")
    elements.append(Paragraph(
        f"<font color='#64748b' size='8'>"
        f"Report Generated: {date_str} at {time_str}"
        f"</font>",
        ParagraphStyle('date', alignment=TA_RIGHT,
                       fontSize=8, textColor=MUTED)
    ))
    elements.append(Spacer(1, 10))

    # ── DETECTION RESULT CARD ───────────────────────────
    status = disease_info.get('status', 'Unknown')
    severity_score = disease_info.get('severity_score', 0)

    if status == 'Healthy':
        status_color = SUCCESS
        status_bg = colors.HexColor('#f0fdf4')
        status_text = 'HEALTHY'
    elif status == 'Diseased':
        status_color = DANGER
        status_bg = colors.HexColor('#fef2f2')
        status_text = 'DISEASED'
    else:
        status_color = WARNING
        status_bg = colors.HexColor('#fffbeb')
        status_text = 'UNKNOWN'

    display_name = predicted_class.replace(
        '__', ' '
    ).replace('_', ' ')

    elements.append(Paragraph(
        "DETECTION RESULT",
        section_style
    ))

    det_data = [
        [
            Paragraph('<font name="Helvetica-Bold" size="10"'
                      f' color="#0f172a">Detected:</font>',
                      body_style),
            Paragraph(f'<font name="Helvetica-Bold" size="12"'
                      f' color="#0d1117">{display_name}</font>',
                      body_style),
            Paragraph('<font name="Helvetica-Bold" size="10"'
                      f' color="#0f172a">Status:</font>',
                      body_style),
            Paragraph(f'<font name="Helvetica-Bold" size="11">'
                      f'{status_text}</font>',
                      ParagraphStyle('s', fontSize=11,
                                     textColor=status_color,
                                     fontName='Helvetica-Bold'))
        ],
        [
            Paragraph('<font name="Helvetica-Bold" size="10"'
                      f' color="#0f172a">Disease:</font>',
                      body_style),
            Paragraph(disease_info.get('disease_name',
                                       'Unknown'),
                      body_style),
            Paragraph('<font name="Helvetica-Bold" size="10"'
                      f' color="#0f172a">Confidence:</font>',
                      body_style),
            Paragraph(f'<font name="Helvetica-Bold" size="11"'
                      f' color="#00d4aa">{confidence:.2f}%</font>',
                      body_style)
        ],
        [
            Paragraph('<font name="Helvetica-Bold" size="10"'
                      f' color="#0f172a">Scientific:</font>',
                      body_style),
            Paragraph(f'<i>{disease_info.get("scientific_name", "Unknown")}</i>',
                      body_style),
            Paragraph('<font name="Helvetica-Bold" size="10"'
                      f' color="#0f172a">Severity:</font>',
                      body_style),
            Paragraph(
                f'{disease_info.get("severity", "Unknown")} '
                f'({severity_score}/10)',
                body_style
            )
        ]
    ]

    det_table = Table(
        det_data,
        colWidths=[1.1*inch, W/2-1.1*inch,
                   1.1*inch, W/2-1.1*inch]
    )
    det_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1),
         colors.HexColor('#f8fafc')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#f8fafc'),
          colors.HexColor('#ffffff'),
          colors.HexColor('#f8fafc')]),
        ('GRID', (0, 0), (-1, -1), 0.3,
         colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [8, 8, 8, 8]),
    ]))
    elements.append(det_table)
    elements.append(Spacer(1, 10))

    # ── MODEL METRICS ────────────────────────────────────
    elements.append(Paragraph(
        "MODEL PERFORMANCE",
        section_style
    ))

    metrics_data = [
        ['Metric', 'Score', 'Metric', 'Score'],
        ['Accuracy', '99.74%',
         'Training Images', '1,39,980'],
        ['Precision', '100.00%',
         'Disease Classes', '66 Types'],
        ['Recall', '100.00%',
         'Fruit Classes', '28 Types'],
        ['F1-Score', '100.00%',
         'Leaf Classes', '38 Types'],
    ]

    met_table = Table(
        metrics_data,
        colWidths=[1.5*inch, 1.5*inch,
                   1.8*inch, 1.7*inch]
    )
    met_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0),
         colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0),
         colors.white),
        ('FONTNAME', (0, 0), (-1, 0),
         'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9.5),
        ('FONTNAME', (1, 1), (1, -1),
         'Helvetica-Bold'),
        ('TEXTCOLOR', (1, 1), (1, -1), PRIMARY),
        ('FONTNAME', (3, 1), (3, -1),
         'Helvetica-Bold'),
        ('TEXTCOLOR', (3, 1), (3, -1), SECONDARY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f0fffe'),
          colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.3,
         colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 9),
    ]))
    elements.append(met_table)
    elements.append(Spacer(1, 10))

    # ── DISEASE ANALYSIS ─────────────────────────────────
    elements.append(Paragraph(
        "DISEASE ANALYSIS",
        section_style
    ))

    analysis_data = [
        ['Affected Parts',
         disease_info.get('affected_parts', 'Unknown')],
        ['Spread Risk',
         disease_info.get('spread_risk', 'Unknown')],
        ['Economic Impact',
         disease_info.get('economic_impact', 'Unknown')],
    ]

    analysis_table = Table(
        analysis_data,
        colWidths=[1.8*inch, W-1.8*inch]
    )
    analysis_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1),
         'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('TEXTCOLOR', (0, 0), (0, -1),
         colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (1, 0), (1, -1),
         colors.HexColor('#334155')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#f8fafc'),
          colors.white,
          colors.HexColor('#f8fafc')]),
        ('GRID', (0, 0), (-1, -1), 0.3,
         colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 9),
    ]))
    elements.append(analysis_table)
    elements.append(Spacer(1, 8))

    # Description
    elements.append(Paragraph(
        "<b>Description</b>",
        ParagraphStyle('bold', fontSize=10,
                       textColor=colors.HexColor('#0f172a'),
                       fontName='Helvetica-Bold',
                       spaceAfter=3)
    ))
    elements.append(Paragraph(
        disease_info.get('description',
                         'No description available.'),
        body_style
    ))

    elements.append(Paragraph(
        "<b>Recommended Treatment</b>",
        ParagraphStyle('bold', fontSize=10,
                       textColor=colors.HexColor('#0f172a'),
                       fontName='Helvetica-Bold',
                       spaceBefore=8, spaceAfter=3)
    ))
    elements.append(Paragraph(
        disease_info.get('treatment',
                         'No treatment info.'),
        body_style
    ))

    elements.append(Paragraph(
        "<b>Prevention Tips</b>",
        ParagraphStyle('bold', fontSize=10,
                       textColor=colors.HexColor('#0f172a'),
                       fontName='Helvetica-Bold',
                       spaceBefore=8, spaceAfter=3)
    ))
    elements.append(Paragraph(
        disease_info.get('prevention',
                         'No prevention info.'),
        body_style
    ))

    elements.append(Paragraph(
        "<b>Farmer Advisory</b>",
        ParagraphStyle('bold', fontSize=10,
                       textColor=colors.HexColor('#0f172a'),
                       fontName='Helvetica-Bold',
                       spaceBefore=8, spaceAfter=3)
    ))
    elements.append(Paragraph(
        disease_info.get('farmer_tips',
                         'No tips available.'),
        body_style
    ))

    elements.append(Spacer(1, 10))

    # ── TOP 5 PREDICTIONS ────────────────────────────────
    elements.append(Paragraph(
        "TOP 5 PREDICTIONS",
        section_style
    ))

    pred_data = [
        ['Rank', 'Condition', 'Confidence', 'Assessment']
    ]
    assessments = [
        'Primary Diagnosis',
        'Alternative',
        'Possible',
        'Low Probability',
        'Unlikely'
    ]
    for i, (cls, prob) in enumerate(predictions[:5]):
        pred_data.append([
            f"#{i+1}",
            cls,
            f"{prob*100:.2f}%",
            assessments[i]
        ])

    pred_table = Table(
        pred_data,
        colWidths=[0.6*inch, 3.0*inch,
                   1.2*inch, 1.7*inch]
    )
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0),
         colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTSIZE', (0, 1), (-1, -1), 9.5),
        ('BACKGROUND', (0, 1), (-1, 1),
         colors.HexColor('#f0fffe')),
        ('FONTNAME', (2, 1), (2, 1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (2, 1), (2, 1), PRIMARY),
        ('ROWBACKGROUNDS', (0, 2), (-1, -1),
         [colors.white,
          colors.HexColor('#f8fafc')]),
        ('GRID', (0, 0), (-1, -1), 0.3,
         colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 9),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 12))

    # ── FOOTER ───────────────────────────────────────────
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#e2e8f0')
    ))
    elements.append(Spacer(1, 6))

    footer_style = ParagraphStyle(
        'Footer',
        fontSize=8,
        textColor=MUTED,
        alignment=TA_CENTER,
        leading=13
    )
    elements.append(Paragraph(
        "This report was automatically generated by the "
        "Fruit Disease Detection System using AI/Deep Learning.",
        footer_style
    ))
    elements.append(Paragraph(
        "For accurate agricultural diagnosis, "
        "please consult a certified agricultural expert.",
        footer_style
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Developed by P. Kavya Sri & V. Alluri Rohit Reddy  |  "
        "Vel Tech Rangarajan Dr. Sagunthala R&D Institute  |  "
        "Batch 2022–2026",
        ParagraphStyle('footer2', fontSize=8,
                       textColor=ACCENT,
                       alignment=TA_CENTER,
                       fontName='Helvetica-Bold')
    ))

    doc.build(elements)
    return save_path
