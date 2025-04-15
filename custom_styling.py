"""
Enhanced CSS styling for the Confidence Intervals Explorer application
Ensures proper display of mathematical formulas in both light and dark modes
"""

def get_custom_css():
    return """
<style>
    /* General layout improvements */
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { margin-top: 0.8rem; margin-bottom: 0.8rem; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
    .stAlert { margin-top: 1rem; margin-bottom: 1rem; }
    
    /* Force visibility of LaTeX elements in ALL contexts */
    /* This uses !important to override any conflicting styles */
    .katex, 
    .katex-display,
    .katex-html {
        color: currentColor !important;
    }
    
    /* Specific LaTeX element styling */
    .katex .mord,
    .katex .mbin,
    .katex .mrel, 
    .katex .mop,
    .katex .mopen,
    .katex .mclose,
    .katex .mpunct,
    .katex .minner,
    .katex .mfrac,
    .katex .msqrt,
    .katex .mfrac * {
        color: currentColor !important;
    }
    
    /* Add a subtle background to math blocks for better contrast */
    .math-block {
        background-color: rgba(245, 245, 245, 0.1) !important;
        border: 1px solid rgba(200, 200, 200, 0.2) !important;
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
        overflow-x: auto;
    }
    
    /* Style for specialized containers with mode-appropriate styling */
    .definition, .example, .proof, .key-equation {
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
    }
    
    /* Light mode container styles */
    html:not([data-theme="dark"]) .definition {
        background-color: rgba(230, 240, 255, 0.8);
        border-left: 4px solid rgba(25, 118, 210, 0.7);
    }
    
    html:not([data-theme="dark"]) .example {
        background-color: rgba(230, 250, 230, 0.8);
        border-left: 4px solid rgba(76, 175, 80, 0.7);
    }
    
    html:not([data-theme="dark"]) .proof {
        background-color: rgba(250, 240, 230, 0.8);
        border-left: 4px solid rgba(220, 150, 80, 0.7);
    }
    
    html:not([data-theme="dark"]) .key-equation {
        background-color: rgba(240, 240, 255, 0.8);
        border-left: 4px solid rgba(75, 75, 200, 0.7);
    }
    
    /* Dark mode container styles - with high specificity selectors */
    [data-theme="dark"] .definition,
    html[data-theme="dark"] .definition,
    body[data-theme="dark"] .definition,
    .stApp[data-theme="dark"] .definition {
        background-color: rgba(25, 118, 210, 0.2) !important;
        border-left: 4px solid rgba(40, 130, 220, 0.8) !important;
    }
    
    [data-theme="dark"] .example,
    html[data-theme="dark"] .example,
    body[data-theme="dark"] .example,
    .stApp[data-theme="dark"] .example {
        background-color: rgba(76, 175, 80, 0.2) !important;
        border-left: 4px solid rgba(86, 195, 90, 0.8) !important;
    }
    
    [data-theme="dark"] .proof,
    html[data-theme="dark"] .proof,
    body[data-theme="dark"] .proof,
    .stApp[data-theme="dark"] .proof {
        background-color: rgba(220, 150, 80, 0.2) !important;
        border-left: 4px solid rgba(240, 165, 90, 0.8) !important;
    }
    
    [data-theme="dark"] .key-equation,
    html[data-theme="dark"] .key-equation,
    body[data-theme="dark"] .key-equation,
    .stApp[data-theme="dark"] .key-equation {
        background-color: rgba(75, 75, 200, 0.2) !important;
        border-left: 4px solid rgba(95, 95, 220, 0.8) !important;
    }
</style>
"""

def get_footer():
    return """
<div style='text-align: center; color: grey; padding: 10px;'>
    <p>Designed and developed by Vishal Bharti © 2025 | PhD-Level Confidence Intervals Explorer</p>
</div>
"""