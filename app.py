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
import math

# Set page configuration
st.set_page_config(
    page_title="Advanced Confidence Intervals Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to enhance the UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {margin-top: 0.8rem; margin-bottom: 0.8rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}
    .stAlert {margin-top: 1rem; margin-bottom: 1rem;}
    .math-block {overflow-x: auto; padding: 10px; margin: 10px 0;}
    .proof {margin-left: 20px; border-left: 3px solid #4CAF50; padding-left: 10px;}
    .definition {background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
    .example {background-color: #f0fff0; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

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
        
        # Quick navigation buttons
        st.markdown("### Quick Access")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Simulations", key="sim_button", on_click=lambda: st.switch_page("pages/02_Interactive_Simulations.py"))
        with col2:
            st.button("Problem Sets", key="prob_button", on_click=lambda: st.switch_page("pages/05_Mathematical_Proofs.py"))

if __name__ == "__main__":
    main()

# Add at the bottom of app.py
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 10px; margin-top: 30px; font-size: 0.8em;">
        <p>© 2025 Designed and developed by <b>Vishal Bharti</b> | Advanced Confidence Intervals Explorer for PhD-Level Statistics</p>
    </div>
    """, 
    unsafe_allow_html=True
)