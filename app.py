import gradio as gr
import torch
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import config
from model import get_model
from validator import validate_image
from disease_info import DISEASE_INFO
from report_generator import generate_pdf_report


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")

fruit_model = None
fruit_class_names = []
leaf_model = None
leaf_class_names = []


def load_fruit_model():
    global fruit_model, fruit_class_names
    if os.path.exists(config.FRUIT_MODEL_PATH):
        checkpoint = torch.load(
            config.FRUIT_MODEL_PATH, map_location=device
        )
        num_classes = checkpoint.get('num_classes', 28)
        fruit_class_names = checkpoint.get('class_names', [])
        fruit_model = get_model(
            num_classes=num_classes
        ).to(device)
        fruit_model.load_state_dict(
            checkpoint['model_state_dict']
        )
        fruit_model.eval()
        print(f"Fruit model loaded! Classes: {num_classes}")


def load_leaf_model():
    global leaf_model, leaf_class_names
    if os.path.exists(config.LEAF_MODEL_PATH):
        leaf_dataset = ImageFolder(root=config.LEAF_DATA_DIR)
        leaf_class_names = leaf_dataset.classes
        import torch.nn as nn
        import torchvision.models as models
        import timm

        class OldHybridModel(nn.Module):
            def __init__(self, num_classes):
                super(OldHybridModel, self).__init__()
                resnet = models.resnet50(weights=None)
                self.resnet_features = nn.Sequential(
                    *list(resnet.children())[:-1]
                )
                self.vit = timm.create_model(
                    "vit_small_patch16_224",
                    pretrained=False, num_classes=0
                )
                combined_size = 2048 + self.vit.num_features
                self.classifier = nn.Sequential(
                    nn.Linear(combined_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, num_classes)
                )

            def forward(self, x):
                resnet_out = self.resnet_features(x)
                resnet_out = resnet_out.view(
                    resnet_out.size(0), -1
                )
                vit_out = self.vit(x)
                combined = torch.cat(
                    (resnet_out, vit_out), dim=1
                )
                return self.classifier(combined)

        leaf_model = OldHybridModel(
            num_classes=len(leaf_class_names)
        ).to(device)
        leaf_model.load_state_dict(
            torch.load(
                config.LEAF_MODEL_PATH,
                map_location=device
            )
        )
        leaf_model.eval()
        print("Leaf model loaded!")


load_fruit_model()
load_leaf_model()

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

last_prediction = {}


def get_logo_base64():
    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'veltech_logo.png'
    )
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def create_confidence_chart(predictions):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    names = [p[0][:28] for p in predictions]
    values = [p[1] * 100 for p in predictions]
    colors = ['#6366f1','#8b5cf6','#06b6d4',
               '#10b981','#f59e0b']
    alphas = [1.0, 0.8, 0.6, 0.4, 0.25]
    for i,(n,v,c,a) in enumerate(
        zip(names,values,colors,alphas)
    ):
        ax.barh(i, v, color=c, alpha=a,
                height=0.5, edgecolor='none')
        ax.text(v+1, i, f'{v:.1f}%',
                va='center', color=c,
                fontsize=9.5, fontweight='700')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color='#9ca3af',
                       fontsize=8.5)
    ax.set_xlim(0, 118)
    ax.tick_params(axis='x', colors='#374151',
                   labelsize=7.5, labelcolor='#4b5563')
    ax.tick_params(axis='y', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.grid(True, color='#161b22', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title('Confidence Scores', color='#e5e7eb',
                 fontsize=10.5, fontweight='600',
                 pad=10, loc='left')
    plt.tight_layout(pad=1.0)
    return fig_to_base64(fig)


def create_metrics_chart():
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.8))
    fig.patch.set_facecolor('#0d1117')
    metrics = [
        ('Accuracy', 99.74, '#6366f1'),
        ('Precision', 100.0, '#8b5cf6'),
        ('Recall', 100.0, '#06b6d4'),
        ('F1-Score', 100.0, '#10b981'),
    ]
    theta = np.linspace(0, 2*np.pi, 300)
    for ax, (name, val, color) in zip(axes, metrics):
        ax.set_facecolor('#0d1117')
        ax.plot(np.cos(theta), np.sin(theta),
                color='#1f2937', linewidth=9,
                solid_capstyle='round', zorder=1)
        arc = np.linspace(
            np.pi/2, np.pi/2 - 2*np.pi*(val/100), 300
        )
        ax.plot(np.cos(arc), np.sin(arc),
                color=color, linewidth=9,
                solid_capstyle='round', zorder=2)
        ax.text(0, 0.1, f'{val:.1f}%',
                ha='center', va='center',
                fontsize=10.5, fontweight='700',
                color='#f9fafb')
        ax.text(0, -0.3, name,
                ha='center', va='center',
                fontsize=7.5, color='#6b7280',
                fontweight='500')
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.suptitle('Model Performance', color='#e5e7eb',
                 fontsize=10.5, fontweight='600',
                 y=1.0, x=0.05, ha='left')
    plt.tight_layout(pad=0.6)
    return fig_to_base64(fig)


def predict_fruit(image):
    global last_prediction

    if image is None:
        return ("", get_empty_result_html(), "")

    is_fruit, detected_class = validate_image(image)

    if not is_fruit:
        return (
            "Invalid Image",
            get_error_html(
                f"Not a fruit or plant image.\n"
                f"Detected: {detected_class}\n\n"
                f"Please upload a fruit or leaf image."
            ),
            ""
        )

    tensor = transform(image).unsqueeze(0).to(device)
    predictions = []
    top_class = ""
    top_prob = 0.0

    if fruit_model is not None and fruit_class_names:
        with torch.no_grad():
            outputs = fruit_model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top5_probs, top5_indices = probs.topk(
                min(5, len(fruit_class_names))
            )
        for i in range(len(top5_probs[0])):
            cls = fruit_class_names[
                top5_indices[0][i].item()
            ]
            cls_d = cls.replace("__"," ").replace("_"," ")
            prob = top5_probs[0][i].item()
            predictions.append((cls_d, prob))
        top_class = fruit_class_names[
            top5_indices[0][0].item()
        ]
        top_prob = top5_probs[0][0].item()

    if not predictions:
        return ("", get_error_html("No model loaded."), "")

    display_name = top_class.replace(
        "__"," "
    ).replace("_"," ")
    confidence_pct = top_prob * 100

    info = DISEASE_INFO.get(top_class, {
        "status": "Unknown",
        "disease_name": "Unknown",
        "scientific_name": "Unknown",
        "description": "No information available.",
        "treatment": "Consult an agricultural expert.",
        "severity": "Unknown",
        "severity_score": 0,
        "prevention": "Regular monitoring recommended.",
        "farmer_tips": "Consult local agricultural officer.",
        "affected_parts": "Unknown",
        "spread_risk": "Unknown",
        "economic_impact": "Unknown"
    })

    last_prediction = {
        'top_class': top_class,
        'confidence': confidence_pct,
        'info': info,
        'predictions': predictions
    }

    result_html = get_result_html(
        display_name, confidence_pct,
        info, predictions
    )
    title = f"{display_name} — {confidence_pct:.2f}%"

    return title, result_html, ""


def get_empty_result_html():
    return """
<div style="
    background:#161b22;border:1px solid #21262d;
    border-radius:16px;padding:64px 32px;
    text-align:center;
">
    <div style="font-size:48px;margin-bottom:16px;">🍎</div>
    <div style="color:#e5e7eb;font-size:16px;
                font-weight:600;margin-bottom:8px;">
        Upload a fruit or leaf image
    </div>
    <div style="color:#6b7280;font-size:13px;
                line-height:1.6;max-width:320px;
                margin:0 auto;">
        Our AI will detect diseases instantly and
        provide a complete analysis report with
        treatment recommendations.
    </div>
</div>
"""


def get_error_html(msg):
    return f"""
<div style="
    background:#161b22;border:1px solid #30363d;
    border-radius:16px;padding:40px 32px;
    text-align:center;
">
    <div style="
        width:52px;height:52px;
        background:rgba(239,68,68,0.1);
        border:1px solid rgba(239,68,68,0.2);
        border-radius:14px;
        display:flex;align-items:center;
        justify-content:center;
        margin:0 auto 16px;font-size:24px;
    ">⚠️</div>
    <div style="color:#f87171;font-size:15px;
                font-weight:600;margin-bottom:10px;">
        Invalid Image
    </div>
    <div style="color:#6b7280;font-size:13px;
                line-height:1.7;white-space:pre-line;">
        {msg}
    </div>
</div>
"""


def get_result_html(display_name, confidence,
                    info, predictions):
    status = info.get('status', 'Unknown')
    severity_score = info.get('severity_score', 0)
    sev_pct = severity_score * 10

    if status == 'Healthy':
        sc = '#22c55e'
        sb = 'rgba(34,197,94,0.08)'
        sbd = 'rgba(34,197,94,0.18)'
        si = '✓'
        sl = 'Healthy'
    else:
        sc = '#ef4444'
        sb = 'rgba(239,68,68,0.08)'
        sbd = 'rgba(239,68,68,0.18)'
        si = '!'
        sl = 'Diseased'

    if severity_score >= 7:
        sev_color = '#ef4444'
    elif severity_score >= 4:
        sev_color = '#f59e0b'
    else:
        sev_color = '#22c55e'

    # Predictions list
    pred_items = ""
    labels = ["Primary","Secondary","Tertiary",
               "Alternative","Unlikely"]
    for i, (cls, prob) in enumerate(predictions[:5]):
        op = 1.0 - i * 0.17
        is_first = i == 0
        pred_items += f"""
<div style="display:flex;align-items:center;
            justify-content:space-between;
            padding:10px 0;
            border-bottom:1px solid {'#21262d' if i<4 else 'transparent'};
            opacity:{op};">
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="
            width:24px;height:24px;
            background:{'#6366f1' if is_first else '#161b22'};
            border:{'none' if is_first else '1px solid #30363d'};
            border-radius:7px;
            display:flex;align-items:center;
            justify-content:center;
            font-size:10px;
            color:{'#fff' if is_first else '#6b7280'};
            font-weight:700;flex-shrink:0;
        ">{i+1}</div>
        <div>
            <div style="color:#e5e7eb;font-size:12.5px;
                        font-weight:{'600' if is_first else '400'};">
                {cls}
            </div>
            <div style="color:#4b5563;font-size:10.5px;">
                {labels[i]}
            </div>
        </div>
    </div>
    <div style="color:{'#a5b4fc' if is_first else '#6b7280'};
                font-size:12.5px;
                font-weight:{'700' if is_first else '400'};">
        {prob*100:.1f}%
    </div>
</div>"""

    conf_b64 = create_confidence_chart(predictions)
    met_b64 = create_metrics_chart()

    return f"""
<!-- Status Banner -->
<div style="
    background:{sb};border:1px solid {sbd};
    border-radius:14px;padding:16px 20px;
    display:flex;align-items:center;gap:14px;
    margin-bottom:14px;
">
    <div style="
        width:38px;height:38px;background:{sc};
        border-radius:50%;
        display:flex;align-items:center;
        justify-content:center;
        color:#fff;font-size:18px;font-weight:700;
        flex-shrink:0;
    ">{si}</div>
    <div style="flex:1;">
        <div style="color:{sc};font-size:14px;
                    font-weight:700;margin-bottom:2px;">
            {sl} — {display_name}
        </div>
        <div style="color:#9ca3af;font-size:12px;">
            Detected with {confidence:.2f}% confidence
            using Naan-Nee Architecture
        </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
        <div style="font-size:26px;font-weight:700;
                    color:#f9fafb;letter-spacing:-1px;
                    line-height:1;">{confidence:.1f}%</div>
        <div style="color:#4b5563;font-size:10px;
                    font-weight:600;letter-spacing:0.5px;
                    text-transform:uppercase;">
            Confidence
        </div>
    </div>
</div>

<!-- 2 Column Grid -->
<div style="display:grid;grid-template-columns:1fr 1fr;
            gap:12px;margin-bottom:12px;">

    <!-- Disease Profile -->
    <div style="background:#161b22;border:1px solid #21262d;
                border-radius:14px;padding:18px;">
        <div style="color:#4b5563;font-size:10px;
                    font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;
                    margin-bottom:14px;">
            Disease Profile
        </div>
        <div style="margin-bottom:11px;">
            <div style="color:#6b7280;font-size:10.5px;
                        margin-bottom:3px;">Disease Name</div>
            <div style="color:#e5e7eb;font-size:13px;
                        font-weight:600;">
                {info.get('disease_name','Unknown')}
            </div>
        </div>
        <div style="margin-bottom:11px;">
            <div style="color:#6b7280;font-size:10.5px;
                        margin-bottom:3px;">Scientific Name</div>
            <div style="color:#9ca3af;font-size:12px;
                        font-style:italic;">
                {info.get('scientific_name','Unknown')}
            </div>
        </div>
        <div style="margin-bottom:11px;">
            <div style="color:#6b7280;font-size:10.5px;
                        margin-bottom:6px;">
                Severity — {severity_score}/10
            </div>
            <div style="background:#0d1117;
                        border-radius:100px;height:5px;
                        overflow:hidden;">
                <div style="width:{sev_pct}%;height:100%;
                            background:{sev_color};
                            border-radius:100px;"></div>
            </div>
        </div>
        <div style="display:grid;
                    grid-template-columns:1fr 1fr;gap:8px;">
            <div>
                <div style="color:#6b7280;font-size:10px;
                            margin-bottom:2px;">
                    Affected Parts
                </div>
                <div style="color:#d1d5db;font-size:11px;">
                    {info.get('affected_parts','—')}
                </div>
            </div>
            <div>
                <div style="color:#6b7280;font-size:10px;
                            margin-bottom:2px;">
                    Spread Risk
                </div>
                <div style="color:#d1d5db;font-size:11px;">
                    {info.get('spread_risk','—')}
                </div>
            </div>
        </div>
    </div>

    <!-- Differential Diagnosis -->
    <div style="background:#161b22;border:1px solid #21262d;
                border-radius:14px;padding:18px;">
        <div style="color:#4b5563;font-size:10px;
                    font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;
                    margin-bottom:10px;">
            Differential Diagnosis
        </div>
        {pred_items}
    </div>
</div>

<!-- Description -->
<div style="background:#161b22;border:1px solid #21262d;
            border-radius:14px;padding:18px;
            margin-bottom:12px;">
    <div style="color:#4b5563;font-size:10px;
                font-weight:700;letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:10px;">
        Clinical Description
    </div>
    <div style="color:#9ca3af;font-size:13px;
                line-height:1.78;">
        {info.get('description','—')}
    </div>
</div>

<!-- Treatment & Prevention -->
<div style="display:grid;grid-template-columns:1fr 1fr;
            gap:12px;margin-bottom:12px;">
    <div style="background:#161b22;border:1px solid #21262d;
                border-radius:14px;padding:18px;">
        <div style="color:#4b5563;font-size:10px;
                    font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;
                    margin-bottom:10px;">Treatment</div>
        <div style="color:#9ca3af;font-size:12.5px;
                    line-height:1.78;">
            {info.get('treatment','—')}
        </div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;
                border-radius:14px;padding:18px;">
        <div style="color:#4b5563;font-size:10px;
                    font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;
                    margin-bottom:10px;">Prevention</div>
        <div style="color:#9ca3af;font-size:12.5px;
                    line-height:1.78;">
            {info.get('prevention','—')}
        </div>
    </div>
</div>

<!-- Farmer Advisory -->
<div style="
    background:rgba(99,102,241,0.05);
    border:1px solid rgba(99,102,241,0.14);
    border-radius:14px;padding:18px;
    margin-bottom:12px;
">
    <div style="color:#a5b4fc;font-size:10px;
                font-weight:700;letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:10px;">Farmer Advisory</div>
    <div style="color:#c4b5fd;font-size:12.5px;
                line-height:1.78;">
        {info.get('farmer_tips','—')}
    </div>
</div>

<!-- Charts -->
<div style="display:grid;grid-template-columns:1fr 1fr;
            gap:12px;margin-bottom:12px;">
    <div style="background:#161b22;border:1px solid #21262d;
                border-radius:14px;padding:14px;
                overflow:hidden;">
        <img src="data:image/png;base64,{conf_b64}"
             style="width:100%;border-radius:8px;"
             alt="Confidence Chart">
    </div>
    <div style="background:#161b22;border:1px solid #21262d;
                border-radius:14px;padding:14px;
                overflow:hidden;">
        <img src="data:image/png;base64,{met_b64}"
             style="width:100%;border-radius:8px;"
             alt="Metrics Chart">
    </div>
</div>

<!-- Metrics Strip -->
<div style="background:#161b22;border:1px solid #21262d;
            border-radius:14px;padding:14px 20px;
            display:flex;align-items:center;
            justify-content:space-between;flex-wrap:wrap;
            gap:12px;">
    <div style="color:#4b5563;font-size:10px;
                font-weight:700;letter-spacing:0.8px;">
        NAAN-NEE MODEL METRICS
    </div>
    <div style="display:flex;gap:22px;flex-wrap:wrap;">
        <div style="text-align:center;">
            <div style="color:#6366f1;font-size:14px;
                        font-weight:700;">99.74%</div>
            <div style="color:#4b5563;font-size:10px;">
                Accuracy</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#8b5cf6;font-size:14px;
                        font-weight:700;">100%</div>
            <div style="color:#4b5563;font-size:10px;">
                Precision</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#06b6d4;font-size:14px;
                        font-weight:700;">100%</div>
            <div style="color:#4b5563;font-size:10px;">
                Recall</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#10b981;font-size:14px;
                        font-weight:700;">100%</div>
            <div style="color:#4b5563;font-size:10px;">
                F1-Score</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#f59e0b;font-size:14px;
                        font-weight:700;">1.4L+</div>
            <div style="color:#4b5563;font-size:10px;">
                Images</div>
        </div>
    </div>
</div>
"""


def download_report():
    global last_prediction
    if not last_prediction:
        return None
    return generate_pdf_report(
        predicted_class=last_prediction['top_class'],
        confidence=last_prediction['confidence'],
        disease_info=last_prediction['info'],
        predictions=last_prediction['predictions'],
        save_path="results/disease_report.pdf"
    )


logo_b64 = get_logo_base64()
if logo_b64:
    logo_tag = (
        f'<img src="data:image/png;base64,{logo_b64}"'
        f' style="height:36px;width:auto;'
        f'object-fit:contain;" alt="Vel Tech">'
    )
else:
    logo_tag = (
        '<div style="height:36px;width:36px;'
        'background:#6366f1;border-radius:8px;'
        'display:flex;align-items:center;'
        'justify-content:center;color:#fff;'
        'font-size:10px;font-weight:700;">VT</div>'
    )

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: 'Inter', -apple-system,
        BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background: #010409 !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
}

/* Image upload area */
.gr-image {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

.gr-image > div {
    background: #0d1117 !important;
}

/* Upload button text */
.gr-image .upload-text {
    color: #8b949e !important;
}

/* Analyze button */
.gr-button-primary {
    background: #6366f1 !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    width: 100% !important;
    letter-spacing: -0.2px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.gr-button-primary:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.38) !important;
}

/* Secondary buttons */
.gr-button-secondary {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #8b949e !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.15s ease !important;
    padding: 10px 16px !important;
}

.gr-button-secondary:hover {
    border-color: #30363d !important;
    color: #e5e7eb !important;
    background: #161b22 !important;
}

/* Text output */
.gr-textbox, textarea {
    background: #0d1117 !important;
    color: #e5e7eb !important;
    border: 1px solid #21262d !important;
    border-radius: 14px !important;
    font-size: 13px !important;
    line-height: 1.85 !important;
    padding: 16px !important;
}

/* Labels */
label, .gr-label {
    color: #6b7280 !important;
    font-size: 10.5px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* HTML output panels */
.gr-html {
    background: transparent !important;
    padding: 0 !important;
}

/* File output */
.gr-file {
    background: #0d1117 !important;
    border: 1px dashed #30363d !important;
    border-radius: 12px !important;
}

/* Row/column backgrounds */
.gap-4, .gr-form, .gr-block,
.gr-box, .gr-panel {
    background: transparent !important;
    border: none !important;
}

footer { display: none !important; }
"""

navbar_html = f"""
<div style="
    width:100%;
    background:rgba(1,4,9,0.96);
    backdrop-filter:blur(20px);
    border-bottom:1px solid #21262d;
    padding:0 28px;
    height:58px;
    display:flex;
    align-items:center;
    justify-content:space-between;
">
    <div style="display:flex;align-items:center;gap:11px;">
        {logo_tag}
        <div style="width:1px;height:24px;
                    background:#21262d;"></div>
        <span style="color:#6b7280;font-size:12.5px;
                     font-weight:400;">
            Vel Tech R&amp;D Institute
        </span>
    </div>

    <div style="
        color:#e5e7eb;font-size:13.5px;
        font-weight:600;letter-spacing:-0.3px;
        position:absolute;left:50%;
        transform:translateX(-50%);
    ">Fruit Disease Detection — Naan-Nee</div>

    <div style="display:flex;align-items:center;gap:10px;">
        <div style="display:flex;align-items:center;gap:5px;">
            <div style="width:6px;height:6px;
                        background:#22c55e;
                        border-radius:50%;
                        box-shadow:0 0 6px rgba(34,197,94,0.6);">
            </div>
            <span style="color:#6b7280;font-size:11.5px;">
                Live
            </span>
        </div>
        <div style="
            background:#161b22;
            border:1px solid #21262d;
            border-radius:7px;
            padding:4px 11px;
            color:#8b949e;
            font-size:11.5px;
            font-weight:500;
        ">Batch 2022–26</div>
    </div>
</div>
"""

hero_html = """
<div style="
    background:#010409;
    padding:72px 28px 60px;
    text-align:center;
    position:relative;overflow:hidden;
    border-bottom:1px solid #21262d;
">
    <div style="position:absolute;top:-100px;left:50%;
                transform:translateX(-50%);
                width:700px;height:500px;
                background:radial-gradient(ellipse,
                    rgba(99,102,241,0.1) 0%,
                    transparent 65%);
                pointer-events:none;"></div>
    <div style="position:absolute;bottom:-60px;left:15%;
                width:320px;height:320px;
                background:radial-gradient(ellipse,
                    rgba(139,92,246,0.07) 0%,
                    transparent 70%);
                pointer-events:none;"></div>
    <div style="position:absolute;bottom:-60px;right:15%;
                width:320px;height:320px;
                background:radial-gradient(ellipse,
                    rgba(6,182,212,0.07) 0%,
                    transparent 70%);
                pointer-events:none;"></div>

    <div style="position:relative;z-index:1;">
        <div style="
            display:inline-flex;align-items:center;
            gap:8px;
            background:rgba(99,102,241,0.1);
            border:1px solid rgba(99,102,241,0.2);
            border-radius:100px;
            padding:5px 18px;margin-bottom:24px;
        ">
            <div style="width:5px;height:5px;
                        background:#6366f1;
                        border-radius:50%;"></div>
            <span style="color:#a5b4fc;font-size:11px;
                         font-weight:600;letter-spacing:0.8px;
                         text-transform:uppercase;">
                AI-Powered Precision Agriculture
            </span>
        </div>

        <h1 style="
            font-size:clamp(30px,5vw,56px);
            font-weight:700;color:#f0f6fc;
            line-height:1.08;letter-spacing:-2px;
            margin-bottom:18px;
        ">
            Detect Fruit Disease<br>
            <span style="
                background:linear-gradient(135deg,
                    #6366f1 0%,#8b5cf6 40%,#06b6d4 100%);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                background-clip:text;
            ">with Precision AI.</span>
        </h1>

        <p style="
            color:#8b949e;font-size:16px;
            font-weight:400;line-height:1.7;
            max-width:460px;margin:0 auto 40px;
            letter-spacing:-0.2px;
        ">
            Upload any fruit or leaf image.
            Instant disease detection with complete
            treatment recommendations — Powered by
             Naan-Nee — Hybrid ResNet50 + Vision Transformer
        </p>

        <div style="
            display:inline-flex;
            background:#0d1117;
            border:1px solid #21262d;
            border-radius:18px;overflow:hidden;
        ">
            <div style="padding:17px 26px;
                        border-right:1px solid #21262d;
                        text-align:center;">
                <div style="font-size:21px;font-weight:700;
                            color:#f0f6fc;letter-spacing:-0.8px;
                            line-height:1;margin-bottom:5px;">
                    98.98%</div>
                <div style="color:#484f58;font-size:9.5px;
                            font-weight:600;letter-spacing:0.8px;
                            text-transform:uppercase;">
                    Fruit Acc.</div>
            </div>
            <div style="padding:17px 26px;
                        border-right:1px solid #21262d;
                        text-align:center;">
                <div style="font-size:21px;font-weight:700;
                            color:#f0f6fc;letter-spacing:-0.8px;
                            line-height:1;margin-bottom:5px;">
                    99.74%</div>
                <div style="color:#484f58;font-size:9.5px;
                            font-weight:600;letter-spacing:0.8px;
                            text-transform:uppercase;">
                    Leaf Acc.</div>
            </div>
            <div style="padding:17px 26px;
                        border-right:1px solid #21262d;
                        text-align:center;">
                <div style="font-size:21px;font-weight:700;
                            color:#f0f6fc;letter-spacing:-0.8px;
                            line-height:1;margin-bottom:5px;">
                    1.4L+</div>
                <div style="color:#484f58;font-size:9.5px;
                            font-weight:600;letter-spacing:0.8px;
                            text-transform:uppercase;">
                    Images</div>
            </div>
            <div style="padding:17px 26px;
                        border-right:1px solid #21262d;
                        text-align:center;">
                <div style="font-size:21px;font-weight:700;
                            color:#f0f6fc;letter-spacing:-0.8px;
                            line-height:1;margin-bottom:5px;">
                    66</div>
                <div style="color:#484f58;font-size:9.5px;
                            font-weight:600;letter-spacing:0.8px;
                            text-transform:uppercase;">
                    Diseases</div>
            </div>
            <div style="padding:17px 26px;
                        text-align:center;">
                <div style="font-size:21px;font-weight:700;
                            color:#f0f6fc;letter-spacing:-0.8px;
                            line-height:1;margin-bottom:5px;">
                    2</div>
                <div style="color:#484f58;font-size:9.5px;
                            font-weight:600;letter-spacing:0.8px;
                            text-transform:uppercase;">
                    AI Models</div>
            </div>
        </div>
    </div>
</div>

<div style="
    background:#010409;
    border-bottom:1px solid #21262d;
    padding:11px 28px;
    display:flex;align-items:center;
    justify-content:space-between;flex-wrap:wrap;gap:8px;
">
    <span style="color:#484f58;font-size:11.5px;">
        B.Tech Computer Science &amp; Engineering
        &nbsp;&bull;&nbsp; Batch 2022–2026
        &nbsp;&bull;&nbsp; Final Year Major Project
    </span>
    <div style="display:flex;gap:7px;">
        <span style="
            background:rgba(99,102,241,0.08);
            border:1px solid rgba(99,102,241,0.15);
            color:#a5b4fc;padding:3px 12px;
            border-radius:7px;font-size:11.5px;
            font-weight:600;">P. Kavya Sri</span>
        <span style="
            background:rgba(99,102,241,0.08);
            border:1px solid rgba(99,102,241,0.15);
            color:#a5b4fc;padding:3px 12px;
            border-radius:7px;font-size:11.5px;
            font-weight:600;">V. Alluri Rohit Reddy</span>
    </div>
</div>
"""

supported_html = """
<div style="
    background:#0d1117;border:1px solid #21262d;
    border-radius:14px;padding:16px;margin-top:10px;
">
    <div style="color:#484f58;font-size:9.5px;
                font-weight:700;letter-spacing:1px;
                text-transform:uppercase;
                margin-bottom:12px;">Supported</div>
    <div style="display:flex;flex-wrap:wrap;
                gap:6px;margin-bottom:12px;">
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Apple</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Banana</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Mango</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Orange</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Grape</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Tomato</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Potato</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">Strawberry</span>
        <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.18);color:#a5b4fc;padding:3px 10px;border-radius:7px;font-size:11px;font-weight:500;">+ More</span>
    </div>
    <div style="
        background:rgba(239,68,68,0.07);
        border:1px solid rgba(239,68,68,0.14);
        border-radius:9px;padding:8px 12px;
        display:flex;align-items:center;gap:7px;
    ">
        <div style="width:5px;height:5px;
                    background:#ef4444;
                    border-radius:50%;flex-shrink:0;"></div>
        <span style="color:#fca5a5;font-size:11px;
                     font-weight:500;">
            Non-fruit images rejected automatically
        </span>
    </div>
</div>
"""

with gr.Blocks(css=css) as app:
    gr.HTML(navbar_html)
    gr.HTML(hero_html)

    with gr.Row():
        with gr.Column(scale=4, min_width=300):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=270
            )
            submit_btn = gr.Button(
                "Analyze Image",
                variant="primary",
                size="lg"
            )
            with gr.Row():
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    size="sm"
                )
                download_btn = gr.Button(
                    "Download PDF",
                    variant="secondary",
                    size="sm"
                )
            pdf_output = gr.File(
                label="PDF Report",
                visible=True
            )
            gr.HTML(supported_html)

        with gr.Column(scale=6, min_width=400):
            result_title = gr.Textbox(
                label="Detection Result",
                lines=1,
                interactive=False,
                visible=False
            )
            result_html = gr.HTML(
                get_empty_result_html()
            )

    submit_btn.click(
        fn=predict_fruit,
        inputs=[image_input],
        outputs=[result_title, result_html,
                 result_title]
    )

    clear_btn.click(
        fn=lambda: (None, get_empty_result_html(), ""),
        outputs=[image_input, result_html, result_title]
    )

    download_btn.click(
        fn=download_report,
        inputs=[],
        outputs=[pdf_output]
    )

if __name__ == "__main__":
    app.launch(share=True)
