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
from PIL import Image
import os
import io
import base64
import pathlib

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

# Add custom CSS for the professional banner
st.markdown("""
<style>
/* Professional banner styling */
.banner-container {
    width: 100%;
    background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
    color: white;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    border-radius: 0.5rem;
}

.banner-title {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 0.025em;
}

.banner-tagline {
    font-size: 1.2rem;
    font-weight: 400;
    font-style: italic;
    text-align: center;
    margin-top: 0.75rem;
    margin-bottom: 0;
    color: rgba(255, 255, 255, 0.9);
}

/* Fix for display issues */
.block-container {
    padding-top: 1rem;
    max-width: 100%;
}

/* Custom card styling */
.info-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    background-color: #f8f9fa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

[data-theme="dark"] .info-card {
    background-color: #2d3748;
    border-color: #4a5568;
}

/* Module card styling */
.module-card-blue {
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.module-card-green {
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #065F46 0%, #10B981 100%);
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.module-card-purple {
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #4F46E5 0%, #8B5CF6 100%);
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.module-button {
    background-color: rgba(255, 255, 255, 0.2);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 600;
    text-align: center;
    cursor: pointer;
    transition: background-color 0.3s ease;
    display: inline-block;
    margin-top: 10px;
}
.module-button:hover {
    background-color: rgba(255, 255, 255, 0.3);
}

/* Navigation button styling */
.nav-button {
    background-color: #4a5568;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    text-align: center;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.nav-button:hover {
    background-color: #2d3748;
}

/* Ensure content appears correctly */
.stApp > header {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# Function to convert PIL Image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Home page content
def main():
    # Hide the default title
    st.markdown("<style>h1:first-of-type {visibility: hidden; height: 0px;}</style>", unsafe_allow_html=True)
    
    # Create a professional-looking title banner
    st.markdown("""
    <div class="banner-container">
        <h1 class="banner-title">Advanced Confidence Intervals Explorer</h1>
        <p class="banner-tagline">A comprehensive tool for understanding, visualizing, and applying confidence interval concepts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main content columns with better proportions
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## About This Application
        
        This interactive tool is designed to deepen your understanding of confidence intervals through:
        
        - **Rigorous mathematical foundations** with formal proofs
        - **Interactive visualizations** to build intuition
        - **Simulations** demonstrating sampling behavior and coverage properties
        - **Advanced techniques** beyond traditional methods
        - **Real-world examples** relevant to various research domains
        - **Comprehensive references** with detailed explanations
        """)
        
        # Add a stylized info card
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top:0;">Getting Started</h3>
            <p>Use the navigation panel on the left to explore different modules. Begin with <b>Theoretical Foundations</b> for key concepts or jump directly to <b>Interactive Simulations</b> for hands-on learning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("This tool is intended for students and researchers in statistics. Some sections assume familiarity with mathematical statistics, but all concepts are explained with interactive visualizations.")
    
    with col2:
        # Try to load the custom image with proper error handling
        try:
            # Look for the image file in common locations
            image_paths = [
                "confidence_intervals_1.png",
                "images/confidence_intervals_1.png",
                "./images/confidence_intervals_1.png",
                "../images/confidence_intervals_1.png"
            ]
            
            image_found = False
            for img_path in image_paths:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    image_found = True
                    break
            
            # If image not found, create a visualization instead
            if not image_found:
                # Create visualization with a unique key
                fig1 = go.Figure()
                x = np.linspace(-4, 4, 1000)
                y = stats.norm.pdf(x)
                
                # Normal distribution
                fig1.add_trace(
                    go.Scatter(x=x, y=y, mode='lines', name='Population Distribution', 
                              line=dict(color='blue', width=2))
                )
                
                # 95% CI region
                ci_x = np.linspace(-1.96, 1.96, 500)
                ci_y = stats.norm.pdf(ci_x)
                fig1.add_trace(
                    go.Scatter(x=ci_x, y=ci_y, fill='tozeroy', mode='none', 
                              name='95% Confidence Region', 
                              fillcolor='rgba(255, 0, 0, 0.2)')
                )
                
                # Add vertical lines for critical values
                fig1.add_vline(x=-1.96, line=dict(color='red', width=2, dash='dash'))
                fig1.add_vline(x=1.96, line=dict(color='red', width=2, dash='dash'))
                
                # Improve layout with fixed legend positioning
                fig1.update_layout(
                    title='95% Confidence Region for Standard Normal',
                    xaxis_title='Z-score',
                    yaxis_title='Density',
                    showlegend=True,
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                # Add a unique key to avoid duplicate element error
                st.plotly_chart(fig1, use_container_width=True, key="normal_distribution_plot")
        except Exception as e:
            st.error(f"Error loading image or creating visualization: {str(e)}")
        
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
    
    # Create a more visual navigation section
    st.markdown("## Explore the Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card-blue">
            <h3 style="margin-top:0;">Theoretical Foundations</h3>
            <p>Definitions, properties, and interpretations of confidence intervals. Learn the mathematical basis and statistical principles.</p>
            <a href="javascript:document.getElementById('foundations_button').click();" class="module-button">Explore Foundations</a>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Explore Foundations", key="foundations_button", help="Navigate to Theoretical Foundations"):
            st.switch_page("pages/01_Theoretical_Foundations.py")
    
    with col2:
        st.markdown("""
        <div class="module-card-green">
            <h3 style="margin-top:0;">Interactive Simulations</h3>
            <p>Hands-on demonstrations of coverage properties, sample size effects, and confidence interval behavior.</p>
            <a href="javascript:document.getElementById('sim_button').click();" class="module-button">Try Simulations</a>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Try Simulations", key="sim_button", help="Navigate to Interactive Simulations"):
            st.switch_page("pages/02_Interactive_Simulations.py")
    
    with col3:
        st.markdown("""
        <div class="module-card-purple">
            <h3 style="margin-top:0;">Real-world Applications</h3>
            <p>Examples from clinical trials, A/B testing, environmental monitoring, and quality control.</p>
            <a href="javascript:document.getElementById('app_button').click();" class="module-button">View Applications</a>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Applications", key="app_button", help="Navigate to Real-world Applications"):
            st.switch_page("pages/04_Real_World_Applications.py")

if __name__ == "__main__":
    main()

# Add footer at the bottom of app.py
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)