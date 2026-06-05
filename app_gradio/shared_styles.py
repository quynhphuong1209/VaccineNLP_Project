"""
shared_styles.py — VaccineNLP shared constants
===============================================
Chứa CSS_STYLE, LABEL_MAPS, LABEL_ICONS, LABEL_COLORS,
SAMPLE_TEXTS, GRADIO_EXAMPLES dùng chung cho app.py và gradio_app.py.

KHÔNG chỉnh sửa trực tiếp trong app.py / gradio_app.py nữa.
Mọi thay đổi CSS/label đều thực hiện ở đây.
"""

CSS_STYLE = """
/* ============================================================
   VaccineNLP — Premium Theme v4.5
   Design: Hybrid Light White / Dark Navy theme
   Inspired by: quynhphuong1209-rehab-ai-monitor-2026.hf.space
   ============================================================ */

/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ===== CSS VARIABLES (Light Mode by Default) ===== */
:root {
    /* ===== Z-INDEX SCALE ===== */
    --z-sidebar:        9999;
    --z-sidebar-toggle: 10001;
    --z-dropdown:       999999;
    --bg-color: #ffffff;
    --bg-gradient: none;
    --text-color: #000000;
    --card-bg: #ffffff;
    --card-border: #e2e8f0;
    --header-bg: #ffffff;
    --header-text: #000000;
    --footer-bg: #ffffff;
    --footer-text: #000000;
    --input-bg: #ffffff;
    --input-text: #000000;
    --input-border: #cbd5e1;
    --accordion-bg: #f8fafc;
    --tab-button-bg: #f1f5f9;
    --tab-button-text: #475569;
    --accent-color: #00b894;
    --accent-bright: #00d4aa;
    --accent-bg: rgba(0, 184, 148, 0.07);
    --shadow-color: rgba(0, 0, 0, 0.06);
    --glow-color: rgba(0, 184, 148, 0.15);
    --card-text-muted: #64748b;
    --card-text-primary: #000000;
    --card-text-secondary: #334155;
    --progress-bar-bg: #e2e8f0;
    --dropdown-bg: #ffffff;
    --custom-card-bg: #ffffff;
    --custom-card-border: #cbd5e1;
    --custom-text-neon: #00b894;
    --custom-text-muted: #475569;
    --custom-text-normal: #000000;
    --saliency-pos-color: 0, 184, 148;
    --custom-phobert-bg: #f0fdf4;
    --custom-xlmr-bg: #eff6ff;
    --custom-gemma-bg: #fffbeb;
    --custom-phobert-border: #00b894;
    --custom-phobert-text: #00b894;

    /* Theme specific variables */
    --sidebar-bg: #ffffff;
    --sidebar-border: #cbd5e1;
    --hero-bg: #ffffff;
    --hero-border: #e2e8f0;
    --hero-title-color: #000000;
    --sidebar-title-color: #000000;
    --hero-subtitle-color: #475569;
    --footer-bg-gradient: #ffffff;
    --footer-border: #cbd5e1;
    --footer-top-accent: #00b894;
    --footer-shadow: 0 -15px 50px rgba(0,0,0,0.04);
    --card-bg-gradient: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
}

/* ===== DARK MODE — Premium Navy ===== */
:root.dark, body.dark, .dark {
    --bg-color: #04091a;
    --bg-gradient: linear-gradient(160deg, #04091a 0%, #070e1a 40%, #050d1f 100%);
    --text-color: #ccd6f6;
    --card-bg: rgba(8, 18, 40, 0.75);
    --card-border: rgba(0, 212, 170, 0.18);
    --header-bg: rgba(4, 9, 26, 0.95);
    --header-text: #e6f1ff;
    --footer-bg: rgba(4, 9, 26, 0.97);
    --footer-text: #8892b0;
    --input-bg: rgba(10, 20, 45, 0.85);
    --input-text: #e6f1ff;
    --input-border: rgba(0, 212, 170, 0.22);
    --accordion-bg: rgba(10, 20, 45, 0.5);
    --tab-button-bg: rgba(10, 20, 45, 0.6);
    --tab-button-text: #8892b0;
    --accent-color: #00d4aa;
    --accent-bright: #00ffcc;
    --accent-bg: rgba(0, 212, 170, 0.09);
    --shadow-color: rgba(0, 0, 0, 0.55);
    --glow-color: rgba(0, 212, 170, 0.3);
    --card-text-muted: #8892b0;
    --card-text-primary: #ccd6f6;
    --card-text-secondary: #a8b2d8;
    --progress-bar-bg: rgba(4, 9, 26, 0.7);
    --dropdown-bg: #0a1628;
    --custom-card-bg: rgba(8, 18, 40, 0.65);
    --custom-card-border: rgba(0, 212, 170, 0.35);
    --custom-text-neon: #00d4aa;
    --custom-text-muted: #8892b0;
    --custom-text-normal: #ccd6f6;
    --saliency-pos-color: 0, 212, 170;
    --custom-phobert-bg: rgba(0, 212, 170, 0.06);
    --custom-xlmr-bg: rgba(0, 123, 255, 0.06);
    --custom-gemma-bg: rgba(255, 165, 0, 0.06);
    --custom-phobert-border: #00d4aa;
    --custom-phobert-text: #00d4aa;

    /* Theme specific variables */
    --sidebar-bg: linear-gradient(180deg, #050f1f 0%, #04091a 100%);
    --sidebar-border: rgba(0, 212, 170, 0.12);
    --hero-bg: linear-gradient(135deg, rgba(4,9,26,0.97) 0%, rgba(5,15,38,0.98) 50%, rgba(4,9,26,0.97) 100%);
    --hero-border: rgba(0, 212, 170, 0.22);
    --hero-title-color: #ffffff;
    --sidebar-title-color: #ffffff;
    --hero-subtitle-color: #8892b0;
    --footer-bg-gradient: linear-gradient(135deg, rgba(4,9,26,0.97) 0%, rgba(5,14,35,0.98) 100%);
    --footer-border: rgba(0, 212, 170, 0.18);
    --footer-top-accent: #00d4aa;
    --footer-shadow: 0 -15px 50px rgba(0,0,0,0.4);
    --card-bg-gradient: linear-gradient(145deg, rgba(8, 18, 40, 0.88) 0%, rgba(5, 12, 30, 0.93) 100%);
}

/* ===== KEYFRAME ANIMATIONS ===== */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(0,212,170,0.3), 0 0 20px rgba(0,212,170,0.1); }
    50%       { box-shadow: 0 0 20px rgba(0,212,170,0.6), 0 0 45px rgba(0,212,170,0.25); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ===== BASE ===== */
* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    text-shadow: none !important;
}
body, html {
    background-color: var(--bg-color) !important;
    background: var(--bg-gradient) !important;
    background-attachment: fixed !important;
    color: var(--text-color) !important;
    margin: 0;
    padding: 0;
    height: auto !important;
    min-height: 100vh;
}

/* ===== TYPOGRAPHY ===== */
body, html, p, li, table, tr, td, th { font-size: 1.06rem !important; line-height: 1.65 !important; }
code, pre, kbd, samp { font-family: 'Fira Code','JetBrains Mono',Consolas,monospace !important; }
h1 { font-size: 2.3rem !important; font-weight: 800 !important; letter-spacing: -0.025em !important; line-height: 1.25 !important; }
h2 { font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.015em !important; line-height: 1.35 !important; }
h3 { font-size: 1.38rem !important; font-weight: 700 !important; line-height: 1.4 !important; }
h4 { font-size: 1.18rem !important; font-weight: 600 !important; }
h5, h6 { font-size: 1.06rem !important; font-weight: 600 !important; }
label { font-size: 1.02rem !important; font-weight: 500 !important; }

/* ===== REMOVE GRADIO LABEL BADGES ===== */
.gradio-container .block-label,
.gradio-container [data-testid="block-label"],
.gradio-container label span,
.gradio-container label > span {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-color) !important;
    padding: 0 !important;
    margin: 0 0 6px 0 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}
.dark .gradio-container .block-label,
.dark .gradio-container [data-testid="block-label"],
.dark .gradio-container label span,
.dark .gradio-container label > span {
    color: var(--text-color) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.28); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,170,0.55); }
::-webkit-scrollbar-button { display: none !important; width: 0 !important; height: 0 !important; }

/* ===== GRADIO CONTAINER ===== */
.gradio-container {
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
    padding: 0 !important;
    overflow: visible !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
.gradio-container .contain { max-width: 100% !important; width: 100% !important; }

/* ===== DROPDOWN FIX ===== */
.gradio-container .border-none { background-color: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-container .options,
.gradio-container .select-options,
.gradio-container .dropdown-menu {
    z-index: var(--z-dropdown) !important;
    background-color: var(--dropdown-bg) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-color) !important;
    box-shadow: 0 16px 45px var(--shadow-color), 0 0 35px var(--glow-color) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    position: absolute !important;
    top: 100% !important;
    bottom: auto !important;
    transform: translateY(6px) !important;
    animation: dropdownFadeIn 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

@keyframes dropdownFadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(6px); }
}

.gradio-container .options .option,
.gradio-container .options .item,
.gradio-container .select-options .option,
.gradio-container .select-options .item,
.gradio-container .dropdown-menu .option,
.gradio-container .dropdown-menu .item {
    color: var(--text-color) !important;
    padding: 11px 15px !important;
    margin: 4px 0 !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Hover effect */
.gradio-container .options .option:hover,
.gradio-container .options .item:hover,
.gradio-container .select-options .option:hover,
.gradio-container .select-options .item:hover,
.gradio-container .dropdown-menu .option:hover,
.gradio-container .dropdown-menu .item:hover {
    background-color: rgba(0, 212, 170, 0.1) !important;
    color: var(--accent-color) !important;
    padding-left: 19px !important; /* Slide right animation */
}

/* Selected state */
.gradio-container .options .option.selected,
.gradio-container .options .item.selected,
.gradio-container .select-options .option.selected,
.gradio-container .select-options .item.selected,
.gradio-container .dropdown-menu .option.selected,
.gradio-container .dropdown-menu .item.selected {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(0, 212, 170, 0.25) !important;
}
.dark .gradio-container .options .option.selected,
.dark .gradio-container .options .item.selected,
.dark .gradio-container .select-options .option.selected,
.dark .gradio-container .select-options .item.selected,
.dark .gradio-container .dropdown-menu .option.selected,
.dark .gradio-container .dropdown-menu .item.selected {
    color: #04091a !important;
    box-shadow: 0 4px 14px rgba(0, 212, 170, 0.45) !important;
}

/* ===== TABS ===== */
.tabs { border-bottom: 1px solid rgba(0,212,170,0.12) !important; background: transparent !important; }
.tab-nav {
    display: flex;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: visible !important;
    gap: 4px !important;
    background: rgba(240,240,240,0.6) !important;
    border-bottom: 1px solid var(--card-border) !important;
    padding: 8px 10px !important;
    border-radius: 12px 12px 0 0 !important;
    -webkit-overflow-scrolling: touch !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
}
.dark .tab-nav {
    background: rgba(4,9,26,0.55) !important;
    border-bottom: 1px solid rgba(0,212,170,0.12) !important;
}
.tab-nav button {
    background: transparent !important;
    color: var(--tab-button-text) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    font-size: 0.81rem !important;
    text-transform: none !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.tab-nav button:hover { color: var(--accent-color) !important; background: rgba(0,212,170,0.06) !important; }
.tab-nav button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(0,212,170,0.25) !important;
}
.dark .tab-nav button.selected {
    color: #04091a !important;
    box-shadow: 0 4px 18px rgba(0,212,170,0.45), 0 0 25px rgba(0,212,170,0.18) !important;
}
@media (max-width: 768px) {
    .tab-nav { gap: 4px !important; padding: 5px !important; }
    .tab-nav button { padding: 6px 10px !important; font-size: 0.76rem !important; border-radius: 7px !important; }
}

/* ===== BUTTONS — Primary ===== */
button.primary, button.gr-button-primary {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 60%, #00a884 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,184,148,0.2) !important;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
    padding: 12px 28px !important;
}
.dark button.primary, .dark button.gr-button-primary {
    color: #04091a !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.38), 0 0 30px rgba(0,212,170,0.12) !important;
}
button.primary:hover, button.gr-button-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(0,184,148,0.35) !important;
    filter: brightness(1.08) !important;
}
.dark button.primary:hover, .dark button.gr-button-primary:hover {
    box-shadow: 0 8px 30px rgba(0,212,170,0.55), 0 0 45px rgba(0,212,170,0.22) !important;
}
button.primary:active, button.gr-button-primary:active { transform: translateY(0) !important; }

/* ===== BUTTONS — Secondary ===== */
button.secondary, button.gr-button-secondary {
    background-color: var(--tab-button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    transition: all 0.28s ease !important;
}
button.secondary:hover, button.gr-button-secondary:hover {
    border-color: var(--accent-color) !important;
    color: var(--accent-color) !important;
    background-color: rgba(0,184,148,0.06) !important;
    box-shadow: 0 4px 18px rgba(0,184,148,0.1) !important;
}

/* ===== INPUTS ===== */
.gradio-container input[type="text"]:not(.border-none):not(.dropdown input):not(.select-wrap input):not(.wrap input),
.gradio-container textarea {
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    transition: all 0.28s ease !important;
}
.gradio-container input[type="text"]:not(.border-none):focus,
.gradio-container textarea:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 3px rgba(0,184,148,0.12) !important;
}

/* ===== CARDS & PANELS ===== */
.gr-box, .gr-panel, .block {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px var(--shadow-color) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

/* ===== SIDEBAR BLOCK FIX ===== */
#sidebar-col .block, #sidebar-col .wrap, #sidebar-col .gap,
#sidebar-col .form > .block, #sidebar-col fieldset { overflow: visible !important; }
#sidebar-col .block, #sidebar-col .gr-block {
    background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
}
#sidebar-col .block:focus-within {
    z-index: 1000 !important;
    position: relative !important;
}

/* ===== ACCORDION ===== */
.gr-accordion { background-color: var(--accordion-bg) !important; border: 1px solid var(--input-border) !important; border-radius: 12px !important; margin-bottom: 12px !important; transition: all 0.3s ease !important; }
.gr-accordion:hover { border-color: rgba(0,184,148,0.3) !important; box-shadow: 0 4px 20px rgba(0,184,148,0.05) !important; }

/* ===== RESULT CARDS ===== */
.result-card-hover {
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    animation: fadeInUp 0.55s cubic-bezier(0.165, 0.84, 0.44, 1) forwards;
}
.result-card-hover:hover {
    transform: translateY(-8px) scale(1.018) !important;
    box-shadow: 0 24px 50px var(--shadow-color) !important;
}
.dark .result-card-hover:hover {
    box-shadow: 0 24px 50px var(--shadow-color), 0 0 40px rgba(0,212,170,0.2) !important;
}
.resource-card { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; }
.resource-card:hover { border-color: var(--accent-color) !important; box-shadow: 0 10px 35px rgba(0,184,148,0.12) !important; transform: translateY(-5px) !important; }

/* ===== THEME TOGGLE ===== */
.theme-dark-btn, .theme-light-btn { border: 1px solid var(--input-border) !important; border-radius: 8px !important; font-weight: 500 !important; cursor: pointer !important; transition: all 0.3s ease !important; }
body.dark .theme-dark-btn { background: linear-gradient(135deg, var(--accent-color) 0%, #00b894 100%) !important; color: #020617 !important; font-weight: bold !important; border-color: var(--accent-color) !important; }
body.dark .theme-light-btn { background-color: var(--tab-button-bg) !important; color: var(--text-color) !important; }
body:not(.dark) .theme-light-btn { background: linear-gradient(135deg, var(--accent-color) 0%, #00b894 100%) !important; color: #ffffff !important; font-weight: bold !important; border-color: var(--accent-color) !important; }
body:not(.dark) .theme-dark-btn { background-color: var(--tab-button-bg) !important; color: var(--text-color) !important; }

/* ===== TOGGLE SWITCH ===== */
.switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 22px;
}
.switch input {
    opacity: 0;
    width: 0;
    height: 0;
}
.slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #cbd5e1 !important; /* light gray when off */
    transition: .4s;
    border-radius: 34px !important;
}
.dark .slider {
    background-color: #334155 !important; /* dark gray when off */
}
.slider:before {
    position: absolute;
    content: "";
    height: 16px;
    width: 16px;
    left: 3px;
    bottom: 3px;
    background-color: white !important;
    transition: .4s;
    border-radius: 50% !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
}
.switch input:checked + .slider {
    background-color: #22c55e !important; /* emerald/green when on */
}
.switch input:checked + .slider:before {
    transform: translateX(22px) !important;
}

/* ===== SIDEBAR LAYOUT ===== */
#sidebar-col {
    align-self: flex-start !important;
    position: sticky !important;
    top: 0 !important;
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    padding: 20px !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease !important;
    display: block !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: 100vh !important;
    max-height: 100vh !important;
    box-sizing: border-box !important;
    width: 290px !important;
    min-width: 290px !important;
    max-width: 290px !important;
    opacity: 1 !important;
    z-index: var(--z-sidebar) !important;
    transform: translateX(0) !important;
}
.dark #sidebar-col, body.dark #sidebar-col {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
}
#sidebar-col::-webkit-scrollbar { width: 4px !important; display: block !important; }
#sidebar-col::-webkit-scrollbar-track { background: transparent !important; }
#sidebar-col::-webkit-scrollbar-thumb { background-color: rgba(0,212,170,0.2) !important; border-radius: 4px !important; }
#sidebar-col::-webkit-scrollbar-thumb:hover { background-color: rgba(0,212,170,0.4) !important; }
#sidebar-col::-webkit-scrollbar-button { display: none !important; width: 0 !important; height: 0 !important; }

#main-layout-row { flex-wrap: nowrap !important; width: 100% !important; display: flex !important; overflow: visible !important; position: relative !important; align-items: stretch !important; }
#sidebar-col.collapsed { width: 0px !important; min-width: 0px !important; max-width: 0px !important; padding: 0px !important; opacity: 0 !important; border-right: none !important; transform: translateX(-290px) !important; pointer-events: none !important; overflow: hidden !important; }
#content-col { position: relative !important; padding-top: 50px !important; padding-left: 20px !important; padding-right: 20px !important; transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; flex: 1 1 auto !important; display: flex !important; flex-direction: column !important; padding-bottom: 30px !important; }
#sidebar-col.collapsed ~ #content-col { width: 100% !important; max-width: 100% !important; }
#sidebar-col:not(.collapsed) ~ #content-col { width: calc(100% - 290px) !important; max-width: calc(100% - 290px) !important; }

/* ===== SIDEBAR TOGGLE ===== */
#sidebar-toggle-btn {
    position: fixed !important; z-index: var(--z-sidebar-toggle) !important; top: 16px !important;
    width: 40px !important; min-width: 40px !important; max-width: 40px !important;
    height: 40px !important; padding: 0 !important; border-radius: 10px !important;
    font-size: 18px !important; font-weight: bold !important;
    background: var(--bg-color) !important; color: var(--accent-color) !important;
    border: 1px solid var(--sidebar-border) !important; cursor: pointer !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.08) !important;
    transition: all 0.25s ease !important; display: flex !important;
    align-items: center !important; justify-content: center !important; left: 258px !important;
}
.dark #sidebar-toggle-btn {
    background: rgba(4,9,26,0.92) !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.3), 0 0 15px rgba(0,212,170,0.1) !important;
}
#sidebar-toggle-btn.sidebar-is-collapsed { left: 16px !important; }
#sidebar-toggle-btn::before {
    content: ""; display: block; width: 18px; height: 18px;
    background-color: var(--accent-color);
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='black'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke-width='2.5' stroke='black'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5'/%3E%3C/svg%3E");
    -webkit-mask-size: contain; mask-size: contain;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
#sidebar-toggle-btn.sidebar-is-collapsed::before { transform: rotate(180deg); }
#sidebar-toggle-btn span { display: none !important; }
#sidebar-toggle-btn:hover {
    background: rgba(0,184,148,0.1) !important;
    border-color: var(--accent-color) !important;
    box-shadow: 0 4px 18px rgba(0,184,148,0.2) !important;
    transform: scale(1.08) !important;
}

/* ===== MOBILE ===== */
@media (max-width: 768px) {
    .gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 8px !important; margin: 0 !important; }
    #sidebar-col {
        position: fixed !important; top: 0 !important; left: 0 !important; height: 100vh !important;
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
        box-shadow: 5px 0 30px rgba(0,0,0,0.1) !important; z-index: var(--z-sidebar) !important;
        transform: translateX(-290px) !important; opacity: 0 !important;
        width: 290px !important; min-width: 290px !important; max-width: 290px !important;
    }
    .dark #sidebar-col {
        box-shadow: 5px 0 30px rgba(0,0,0,0.5) !important;
    }
    #sidebar-col:not(.collapsed) { transform: translateX(0) !important; opacity: 1 !important; pointer-events: auto !important; }
    #sidebar-col.collapsed { transform: translateX(-290px) !important; opacity: 0 !important; width: 0px !important; min-width: 0px !important; max-width: 0px !important; padding: 0px !important; }
    #content-col { width: 100% !important; max-width: 100% !important; padding-top: 60px !important; padding-left: 8px !important; padding-right: 8px !important; }
    #sidebar-col.collapsed ~ #content-col, #sidebar-col:not(.collapsed) ~ #content-col { width: 100% !important; max-width: 100% !important; }
}

/* ===== PLOTLY ===== */
.js-plotly-plot { background-color: transparent !important; width: 100% !important; }
.js-plotly-plot .bg { fill: transparent !important; }
.js-plotly-plot text, .js-plotly-plot tspan, .js-plotly-plot .xtick text, .js-plotly-plot .ytick text,
.js-plotly-plot .gtitle, .js-plotly-plot .xtitle, .js-plotly-plot .ytitle,
.js-plotly-plot .legendtext { fill: var(--text-color) !important; }
.dark .js-plotly-plot text, .dark .js-plotly-plot tspan { fill: #ffffff !important; }
.js-plotly-plot .gridlayer path, .js-plotly-plot .zerolinelayer path,
.js-plotly-plot .axis line { stroke: rgba(128,128,128,0.15) !important; }
.dark .js-plotly-plot .gridlayer path, .dark .js-plotly-plot .zerolinelayer path,
.dark .js-plotly-plot .axis line { stroke: rgba(255,255,255,0.1) !important; }
.js-plotly-plot .sankey-node text { fill: var(--text-color) !important; }

/* Radar polar styles */
.js-plotly-plot .polarbg { fill: #ffffff !important; }
.dark .js-plotly-plot .polarbg { fill: #000000 !important; }
.js-plotly-plot .polargrid path { stroke: rgba(0,0,0,0.1) !important; }
.dark .js-plotly-plot .polargrid path { stroke: rgba(255,255,255,0.75) !important; }
.gr-plot, .gradio-plot, .plot-container, [data-testid="plot"], .js-plotly-plot,
.plotly, .svg-container, .main-svg { width: 100% !important; max-width: 100% !important; }
.gr-plot > div, .plot-container > div, .js-plotly-plot > div { width: 100% !important; max-width: 100% !important; }

/* ===== MISC UTILITIES ===== */
.model-color-phobert { color: var(--custom-text-neon) !important; }
.model-color-xlmr { color: #3b82f6 !important; }
.model-color-gemma { color: #FFA500 !important; }
.dropdown-menu { background-color: var(--input-bg) !important; border: 1px solid var(--input-border) !important; }
.reset-btn-layout { background: transparent !important; border: 1.5px solid var(--input-border) !important; color: var(--text-color) !important; transition: all 0.2s ease !important; }
.reset-btn-layout:hover { border-color: var(--accent-color) !important; color: var(--accent-color) !important; background-color: var(--accent-bg) !important; }
#sidebar-col .form { display: flex !important; flex-direction: column !important; gap: 18px !important; height: auto !important; overflow: visible !important; }
#sidebar-col .form > * { width: 100% !important; box-sizing: border-box !important; flex-shrink: 0 !important; }
#sidebar-col hr { margin: 28px 0 20px 0 !important; opacity: 0.85; }
#sidebar-col h5 { margin-top: 4px !important; margin-bottom: 10px !important; }
.sidebar-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, rgba(0,184,148,0.2) 50%, transparent 100%) !important; margin: 28px 0 20px 0 !important; border: none !important; }
.dark .sidebar-divider { background: linear-gradient(90deg, rgba(0,212,170,0) 0%, rgba(0,212,170,0.3) 50%, rgba(0,212,170,0) 100%) !important; }
.sidebar-scroll-btn { display: none !important; }
footer, .gradio-container > footer { display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; }

/* ===== HERO BANNER ===== */
#hero-banner {
    background: var(--hero-bg) !important;
    border: 1px solid var(--hero-border) !important;
    border-radius: 20px !important;
    padding: 48px 28px !important;
    text-align: center !important;
    margin-bottom: 28px !important;
    box-shadow: 0 12px 50px rgba(0,0,0,0.04) !important;
    position: relative !important;
    overflow: hidden !important;
}
.dark #hero-banner {
    box-shadow: 0 12px 50px rgba(0,0,0,0.55), 0 0 80px rgba(0,212,170,0.07),
                inset 0 1px 0 rgba(0,212,170,0.12) !important;
}
#hero-banner::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 15% 50%, rgba(0,184,148,0.04) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 50%, rgba(0,100,255,0.03) 0%, transparent 55%);
    pointer-events: none;
}
.dark #hero-banner::before {
    background: radial-gradient(ellipse at 15% 50%, rgba(0,212,170,0.07) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 50%, rgba(0,100,255,0.05) 0%, transparent 55%);
}
.hero-accent-line {
    position: absolute !important; top: 0 !important; left: 0 !important; right: 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, transparent 0%, #00d4aa 25%, #00ffcc 50%, #00d4aa 75%, transparent 100%) !important;
    background-size: 200% auto !important;
    animation: shimmer 3s linear infinite !important;
}
.hero-emojis {
    font-size: 2.5rem !important; margin-bottom: 10px !important;
    filter: drop-shadow(0 0 14px rgba(0,184,148,0.2)) !important;
    animation: float 4s ease-in-out infinite !important;
    display: inline-block !important;
}
.dark .hero-emojis {
    filter: drop-shadow(0 0 14px rgba(0,212,170,0.45)) !important;
}
.hero-title {
    margin: 6px 0 16px 0 !important;
    font-size: clamp(1.6rem, 3.5vw, 2.6rem) !important;
    font-weight: 800 !important;
    color: var(--hero-title-color) !important;
    line-height: 1.3 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}
.hero-divider {
    width: 130px !important; height: 3px !important;
    background: var(--hero-title-gradient, linear-gradient(135deg, #e6f1ff 0%, #00d4aa 35%, #00ffcc 55%, #ccd6f6 100%)) !important;
    background-size: 200% auto !important;
    animation: shimmer 2.5s linear infinite !important;
    margin: 16px auto !important; border-radius: 2px !important;
}
.hero-subtitle {
    margin: 0 auto !important;
    font-size: clamp(0.88rem, 1.8vw, 1.1rem) !important;
    color: var(--hero-subtitle-color) !important;
    font-weight: 500 !important;
    max-width: 820px !important;
    line-height: 1.55 !important;
}

/* ===== REMOVE XAI TAB BOX BORDERS ===== */
#xai-tabs {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
#xai-tabs > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#xai-tabs .tabitem {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 15px 0 !important;
}

/* ===== REMOVE BORDERS AND BACKGROUNDS FROM GRADIO MARKDOWN & HTML BLOCKS ===== */
.type-markdown, .block:has(.prose), .block:has(.gradio-markdown),
.type-html, .gradio-html, .gradio-html.block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 8px 0 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* ===== COMPACT DROPDOWNS & SELECTION BOXES ===== */
.gradio-dropdown, .select-wrap {
    min-height: 28px !important;
    font-size: 0.80rem !important;
}
.select-wrap input,
.select-wrap .token,
.select-wrap .single-select,
.select-wrap .item,
.select-wrap .control {
    font-size: 0.80rem !important;
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    min-height: 26px !important;
}
/* Dropdown list items: flush to left edge */
.gradio-container .options .option,
.gradio-container .options .item,
.gradio-container .select-options .option,
.gradio-container .select-options .item,
.gradio-container .dropdown-menu .option,
.gradio-container .dropdown-menu .item,
ul.options > li,
.options li {
    padding: 3px 4px 3px 6px !important;
    font-size: 0.78rem !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* ===== MOBILE DROPDOWN FIX ===== */
@media (max-width: 768px) {
    .gradio-container .options,
    .gradio-container .select-options,
    .gradio-container .dropdown-menu {
        max-width: calc(100vw - 32px) !important;
        left: 0 !important;
    }
    /* Charts row stack vertically */
    .gradio-container .gr-row {
        flex-direction: column !important;
    }
}
"""

LABEL_MAPS = {
    "misinfo":   {0: "Tin giả",  1: "Tin thật"},
    "stance":    {0: "Ủng hộ",   1: "Phản đối", 2: "Trung lập"},
    "sentiment": {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"},
}

LABEL_ICONS = {
    "misinfo":   {0: "🚨", 1: "✅"},
    "stance":    {0: "👍", 1: "👎", 2: "🤝"},
    "sentiment": {0: "😠", 1: "😐", 2: "😊"},
}
LABEL_COLORS = {
    "misinfo":   {0: "#e8504a", 1: "#3db882"},
    "stance":    {0: "#3db882", 1: "#e8504a", 2: "#4a9eed"},
    "sentiment": {0: "#e8504a", 1: "#4a9eed", 2: "#3db882"},
}

SAMPLE_TEXTS = {
    "🚨 Tin giả - Chống vaccine cực đoan": "Ko tiêm mũi nào hết. Ko biết bạn thuộc thế hệ nào, chứ bạn nhìn xem thế hệ 8x trở về trước ko có ai tiêm bất cứ mũi gì vẫn khoẻ mạnh đó thôi. Cha mẹ thời nay bị doạ cho sợ hãi, đem con đi tiêm vì bị bóng ma sợ hãi nó đè, chứ thực chất chả có tác dụng gì còn gây hại cho cơ thể nữa.",
    "🚨 Tin giả - Vô sinh": "Cảnh báo: vắc xin COVID có thể gây vô sinh ở phụ nữ và biến đổi gen ở trẻ em. Mọi người nên tìm hiểu kỹ trước khi làm chuột bạch cho các tập đoàn dược phẩm.",
    "🟢 Ủng hộ tiêm chủng": "Em cũng đang tiêm từng mũi 1 cho con, con e 5 tháng, mới tiêm tới phế cầu, 3 tháng đầu chỉ tiêm 6in1 và uống rota. Chậm mà đủ và an toàn cho con là được. Trộm vía bé e chưa sốt, chưa hành mũi nào ❤️",
    "🟡 Nghi ngại": "Cún mình chỉ tiêm mũi ở viện nhà là ko tiêm gì nữa. Bây giờ 2 tuổi rồi. Ai hỏi t vẫn nói tiêm đủ.",
    "✅ Thông tin chuẩn": "Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi vaccine cơ bản theo Chương trình Tiêm chủng Mở rộng để phòng các bệnh truyền nhiễm nguy hiểm.",
    "🔵 Câu hỏi tư vấn": "Trâm Trần ví dụ như Ko có tiêm 6in1 hay 5in1, mà tiêm từng mũi từng bệnh phải không ạ?",
    "💬 Tin giả - Từ lóng MXH": "K có vacxin thì hệ miễn dịch khỏe sẽ rất ít khi bị ốm bị bệnh \nNhưng tiêm vắc xin thì là tiêm thuốc độc vào người \n\nCàng tiêm nhiều càng bệnh nhiều \n\nBạn xem thời xưa có ai phải tiêm đâu sao ai cũng khỏe mạnh\n\nMuốn thải độc vx , kim loại nặng thì nên cho uống nc lá mùi đun lên \n\nMuốn hạ sốt ( sốt nóng ) cho con uống nc chanh ấm có đường \nLấy chanh xoa toàn thân",
}

GRADIO_EXAMPLES = [
    ["Vắc xin COVID gây vô sinh ở phụ nữ trẻ và biến đổi gen ở trẻ em.", "PhoBERT-v2"],
    ["Em đã tiêm đủ 5 mũi cho con theo lịch tiêm chủng mở rộng. Bé khỏe mạnh, không sốt.", "PhoBERT-v2"],
    ["Bộ Y tế công bố lịch tiêm chủng mới cho trẻ em năm 2026.", "PhoBERT-v2"],
    ["Vaccine là chip 5G theo dõi người dân, không nên tin tưởng.", "XLM-R-v1"],
]
