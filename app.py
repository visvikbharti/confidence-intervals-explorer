import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, minimize_scalar
from custom_styling import get_custom_css, get_footer
from latex_helper import render_latex, render_definition, render_example, render_proof, render_key_equation
from force_visible_math import force_visible_math_mode, inline_math_fix

# Set page configuration
st.set_page_config(
    page_title="Advanced Confidence Intervals Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for better formula display in both light and dark modes
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Apply the force_visible_math and inline_math_fix functions
force_visible_math_mode()
inline_math_fix()

# Home page content
def main():
    st.title("Advanced Confidence Intervals Explorer")
    st.markdown("*A comprehensive tool for understanding, visualizing, and applying confidence interval concepts*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This application is designed to deepen your understanding of confidence intervals through:
        
        - **Rigorous mathematical foundations** with formal proofs
        - **Interactive visualizations** to build intuition
        - **Simulations** demonstrating sampling behavior and coverage properties
        - **Advanced techniques** beyond traditional methods
        - **Real-world examples** relevant to various research domains
        - **Challenging problem sets** with detailed solutions
        
        Use the navigation panel on the left to explore different modules.
        """)
        
        st.info("This tool is intended for PhD-level statistics students and researchers. Some sections assume familiarity with mathematical statistics, measure theory, and statistical computing.")
    
    with col2:
        # Create an illustrative figure
        fig = go.Figure()
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x)
        
        # Normal distribution
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name='Population', 
                      line=dict(color='blue', width=2))
        )
        
        # 95% CI region
        ci_x = np.linspace(-1.96, 1.96, 500)
        ci_y = stats.norm.pdf(ci_x)
        fig.add_trace(
            go.Scatter(x=ci_x, y=ci_y, fill='tozeroy', mode='none', 
                      name='95% Confidence', 
                      fillcolor='rgba(255, 0, 0, 0.2)')
        )
        
        fig.update_layout(
            title='95% Confidence Region',
            xaxis_title='Parameter Value',
            yaxis_title='Density',
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Formula section with both t and z formulas
        st.markdown("### Key Formulas")
        
        # Formula 1: t-distribution (unknown variance)
        st.markdown("""
        <div class="key-equation">
        <strong>Confidence Interval for Mean (Unknown Variance):</strong>

        $$CI_{1-\\alpha}(\\mu) = \\bar{X} \\pm t_{\\alpha/2, n-1} \\cdot \\frac{s}{\\sqrt{n}}$$

        This provides a $(1-\\alpha)$ confidence level for the population mean $\\mu$ when the population variance is unknown.
        </div>
        """, unsafe_allow_html=True)
        
        # Formula 2: z-distribution (known variance)
        st.markdown("""
        <div class="key-equation">
        <strong>Confidence Interval for Mean (Known Variance):</strong>

        $$CI_{1-\\alpha}(\\mu) = \\bar{X} \\pm z_{\\alpha/2} \\cdot \\frac{\\sigma}{\\sqrt{n}}$$

        This provides a $(1-\\alpha)$ confidence level for the population mean $\\mu$ when the population variance $\\sigma^2$ is known.
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation buttons
        st.markdown("### Quick Access")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Simulations", key="sim_button"):
                st.switch_page("pages/02_Interactive_Simulations.py")
        with col2:
            if st.button("Problem Sets", key="prob_button"):
                st.switch_page("pages/04_Real_World_Applications.py")

if __name__ == "__main__":
    main()

# Add footer at the bottom of app.py
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)