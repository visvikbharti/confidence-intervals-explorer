"""
Enhanced helper functions for improved LaTeX formula rendering in the Confidence Intervals Explorer
with special attention to dark mode compatibility
"""

import streamlit as st

def render_latex(formula, block=False):
    """
    Renders LaTeX formula with improved visibility in both light and dark modes
    
    Args:
        formula (str): The LaTeX formula to render
        block (bool): Whether to render as a block (display mode) or inline
    
    Returns:
        None: Renders directly to the Streamlit app
    """
    if block:
        # For block/display mode equations
        st.markdown(f"""
        <div class="math-block">
            <div class="latex-formula-block">
                $$
                {formula}
                $$
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # For inline equations
        st.markdown(f"""
        <span class="latex-formula-inline">
            ${formula}$
        </span>
        """, unsafe_allow_html=True)

def render_definition(content):
    """
    Renders content with the definition styling
    
    Args:
        content (str): The content to render, can include LaTeX
    
    Returns:
        None: Renders directly to the Streamlit app
    """
    st.markdown(f"""
    <div class="definition">
        {content}
    </div>
    """, unsafe_allow_html=True)

def render_example(content):
    """
    Renders content with the example styling
    
    Args:
        content (str): The content to render, can include LaTeX
    
    Returns:
        None: Renders directly to the Streamlit app
    """
    st.markdown(f"""
    <div class="example">
        {content}
    </div>
    """, unsafe_allow_html=True)

def render_proof(content):
    """
    Renders content with the proof styling
    
    Args:
        content (str): The content to render, can include LaTeX
    
    Returns:
        None: Renders directly to the Streamlit app
    """
    st.markdown(f"""
    <div class="proof">
        {content}
    </div>
    """, unsafe_allow_html=True)

def render_key_equation(content):
    """
    Renders content with the key equation styling
    
    Args:
        content (str): The content to render, can include LaTeX
    
    Returns:
        None: Renders directly to the Streamlit app
    """
    st.markdown(f"""
    <div class="key-equation">
        {content}
    </div>
    """, unsafe_allow_html=True)

def render_latex_table(headers, rows, caption=None):
    """
    Renders a table with LaTeX formatting support in cells
    
    Args:
        headers (list): List of column headers
        rows (list): List of rows, where each row is a list of cell values
        caption (str, optional): Table caption
    
    Returns:
        None: Renders directly to the Streamlit app
    """
    html = '<div class="latex-table-container"><table class="latex-table">'
    
    # Add caption if provided
    if caption:
        html += f'<caption>{caption}</caption>'
    
    # Add headers
    html += '<thead><tr>'
    for header in headers:
        html += f'<th>{header}</th>'
    html += '</tr></thead>'
    
    # Add rows
    html += '<tbody>'
    for row in rows:
        html += '<tr>'
        for cell in row:
            html += f'<td>${cell}$</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    
    st.markdown(html, unsafe_allow_html=True)

def add_latex_styles():
    """
    Adds custom CSS styles for better LaTeX rendering in both light and dark modes
    Call this at the beginning of your app or page
    """
    st.markdown("""
    <style>
    /* Specific styling for LaTeX formulas */
    .latex-formula-inline {
        display: inline-block;
        background-color: rgba(128, 128, 128, 0.1);
        padding: 0 4px;
        border-radius: 3px;
        margin: 0 2px;
    }
    
    .latex-formula-block {
        display: block;
        width: 100%;
    }
    
    /* LaTeX table styling */
    .latex-table-container {
        margin: 20px 0;
        overflow-x: auto;
    }
    
    .latex-table {
        width: 100%;
        border-collapse: collapse;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }
    
    .latex-table th,
    .latex-table td {
        padding: 8px 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        text-align: center;
    }
    
    .latex-table th {
        background-color: rgba(128, 128, 128, 0.1);
        font-weight: bold;
    }
    
    .latex-table caption {
        margin-bottom: 10px;
        font-style: italic;
    }
    
    /* Dark mode tweaks */
    [data-theme="dark"] .latex-formula-inline,
    .stApp[data-theme="dark"] .latex-formula-inline {
        background-color: rgba(255, 255, 255, 0.15);
    }
    
    [data-theme="dark"] .latex-table th,
    .stApp[data-theme="dark"] .latex-table th {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    [data-theme="dark"] .latex-table,
    [data-theme="dark"] .latex-table th,
    [data-theme="dark"] .latex-table td,
    .stApp[data-theme="dark"] .latex-table,
    .stApp[data-theme="dark"] .latex-table th,
    .stApp[data-theme="dark"] .latex-table td {
        border-color: rgba(255, 255, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)