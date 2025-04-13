import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.optimize import minimize
import base64
from io import BytesIO
import math
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import _tconfint_generic
import sympy as sp
from sympy import symbols, solve, Eq, latex

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

# Main title
st.title("Advanced Confidence Intervals Explorer")
st.markdown("*A comprehensive tool for understanding, visualizing, and applying confidence interval concepts*")

# App navigation
nav = st.sidebar.radio(
    "Navigation",
    ["Home", "Theoretical Foundations", "Interactive Simulations", 
     "Advanced Methods", "Real-world Applications", "Problem Sets", 
     "Mathematical Proofs", "References & Resources"]
)

# Home page
if nav == "Home":
    st.header("Welcome to the Advanced Confidence Intervals Explorer")
    
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
        # Placeholder for an illustrative figure
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
            st.button("Simulations", key="sim_button")
        with col2:
            st.button("Problem Sets", key="prob_button")

# Theoretical Foundations
elif nav == "Theoretical Foundations":
    st.header("Theoretical Foundations of Confidence Intervals")
    
    tabs = st.tabs(["Definitions & Properties", "Statistical Theory", "Derivations", "Optimality", "Interpretation"])
    
    with tabs[0]:  # Definitions & Properties
        st.subheader("Formal Definitions")
        
        st.markdown(r"""
        <div class="definition">
        <strong>Definition 1:</strong> A confidence interval for a parameter $\theta$ is a pair of statistics $L(X)$ and $U(X)$ such that:
        
        $$P_{\theta}(L(X) \leq \theta \leq U(X)) \geq 1-\alpha \quad \forall \theta \in \Theta$$
        
        where $1-\alpha$ is the confidence level, and $\Theta$ is the parameter space.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(r"""
        <div class="definition">
        <strong>Definition 2:</strong> A random interval $[L(X), U(X)]$ is an <em>exact</em> confidence interval if:
        
        $$P_{\theta}(L(X) \leq \theta \leq U(X)) = 1-\alpha \quad \forall \theta \in \Theta$$
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Essential Properties")
        
        property_descriptions = {
            "Coverage Probability": "The fundamental property that defines confidence intervals. An interval has correct coverage if the true parameter is contained within the random interval with frequency at least $1-\alpha$ under repeated sampling.",
            "Precision": "Measured by the expected width of the confidence interval. Narrower intervals provide more precise information about the parameter.",
            "Equivariance": "If a confidence interval for $\theta$ is $[L,U]$, then a confidence interval for $g(\theta)$ is $[g(L), g(U)]$ for monotone transformations $g$.",
            "Consistency": "As sample size increases, the width of the confidence interval should approach zero, and the coverage should approach the nominal level.",
            "Efficiency": "Among all intervals with the same coverage, efficient intervals have minimum expected width."
        }
        
        for prop, desc in property_descriptions.items():
            st.markdown(f"**{prop}**: {desc}")
        
        # Visualization of coverage
        st.subheader("Coverage Visualization")
        
        # Sample size and confidence level selectors
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of random samples", 10, 200, 50, key="n_samples_def")
        with col2:
            conf_level = st.select_slider("Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95, key="conf_level_def")
        
        # Generate visualization of coverage
        if st.button("Generate Coverage Visualization", key="gen_coverage"):
            # Set true parameter value
            true_mu = 50
            true_sigma = 10
            
            # Generate multiple samples and compute CIs
            samples = []
            lower_bounds = []
            upper_bounds = []
            contains_param = []
            
            for i in range(n_samples):
                # Generate sample
                sample = np.random.normal(true_mu, true_sigma, 30)
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                
                # Compute CI
                margin = stats.t.ppf((1 + conf_level) / 2, 29) * sample_std / np.sqrt(30)
                lower = sample_mean - margin
                upper = sample_mean + margin
                
                # Store results
                samples.append(sample_mean)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                contains_param.append(lower <= true_mu <= upper)
            
            # Create visualization
            fig = go.Figure()
            
            # Add horizontal line for true parameter
            fig.add_hline(y=true_mu, line=dict(color='red', width=2, dash='dash'), 
                         annotation=dict(text="True μ = 50", showarrow=False, 
                                         xref="paper", yref="y", 
                                         x=1.02, y=true_mu))
            
            # Add confidence intervals
            for i in range(n_samples):
                color = 'rgba(0, 128, 0, 0.5)' if contains_param[i] else 'rgba(255, 0, 0, 0.5)'
                fig.add_trace(
                    go.Scatter(x=[i, i], y=[lower_bounds[i], upper_bounds[i]], 
                              mode='lines', line=dict(color=color, width=2))
                )
                fig.add_trace(
                    go.Scatter(x=[i], y=[samples[i]], mode='markers', 
                              marker=dict(color='blue', size=6))
                )
            
            # Calculate coverage
            actual_coverage = sum(contains_param) / n_samples * 100
            
            fig.update_layout(
                title=f'Coverage of {conf_level*100:.0f}% Confidence Intervals<br>'
                      f'Actual coverage: {actual_coverage:.1f}% ({sum(contains_param)} out of {n_samples})',
                xaxis_title='Sample Number',
                yaxis_title='Parameter Value',
                showlegend=False,
                height=500,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            This visualization shows {n_samples} different 95% confidence intervals constructed from different random samples.
            
            - Each **vertical line** represents a confidence interval
            - **Green intervals** contain the true parameter value (μ = 50)
            - **Red intervals** miss the true parameter value
            - **Blue dots** represent the sample means
            
            Theoretically, {conf_level*100:.0f}% of intervals should contain the true parameter.
            In this simulation, {actual_coverage:.1f}% ({sum(contains_param)} out of {n_samples}) intervals contain the true parameter.
            """)

    with tabs[1]:  # Statistical Theory
        st.subheader("Statistical Theory of Confidence Intervals")
        
        st.markdown(r"""
        ### Pivotal Quantities
        
        A pivotal quantity is a function of both the data and the parameter whose distribution does not depend on any unknown parameter. This property makes pivotal quantities ideal for constructing confidence intervals.
        
        <div class="definition">
        <strong>Definition:</strong> A statistic $Q(X, \theta)$ is a pivotal quantity if its distribution is the same for all $\theta \in \Theta$.
        </div>
        
        <div class="example">
        <strong>Example:</strong> For a normal sample with unknown mean $\mu$ and known variance $\sigma^2$, the quantity
        $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$$
        follows a standard normal distribution regardless of the value of $\mu$, making it a pivotal quantity.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(r"""
        ### Relationship with Hypothesis Testing
        
        There is a fundamental duality between confidence intervals and hypothesis testing:
        
        - A $(1-\alpha)$ confidence interval for $\theta$ contains precisely those values that would not be rejected by a level-$\alpha$ test of $H_0: \theta = \theta_0$
        
        This relationship can be expressed formally as:
        
        $$\theta_0 \in CI_{1-\alpha}(X) \iff \text{The test of } H_0: \theta = \theta_0 \text{ is not rejected at level } \alpha$$
        """, unsafe_allow_html=True)
        
        st.subheader("Theoretical visualization")
        
        # Create interactive visualization of the duality
        st.markdown("##### Interactive demonstration of the duality between confidence intervals and hypothesis tests")
        
        col1, col2 = st.columns(2)
        with col1:
            mu_0 = st.slider("Hypothesized mean (μ₀)", 0.0, 10.0, 5.0, 0.1)
        with col2:
            sample_mean = st.slider("Observed sample mean", 0.0, 10.0, 7.0, 0.1)
            
        # Fixed parameters
        n = 30
        sigma = 2
        alpha = 0.05
        
        # Critical values
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # CI bounds
        ci_lower = sample_mean - z_crit * sigma / np.sqrt(n)
        ci_upper = sample_mean + z_crit * sigma / np.sqrt(n)
        
        # Test statistic
        z_stat = (sample_mean - mu_0) / (sigma / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Create visualization
        x = np.linspace(0, 10, 1000)
        sampling_dist = stats.norm.pdf(x, mu_0, sigma / np.sqrt(n))
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Hypothesis Test Perspective", "Confidence Interval Perspective"),
                           vertical_spacing=0.15)
        
        # Hypothesis test subplot
        fig.add_trace(
            go.Scatter(x=x, y=sampling_dist, name="Sampling Distribution",
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add critical regions
        reject_region_left = x[x <= mu_0 - z_crit * sigma / np.sqrt(n)]
        reject_y_left = stats.norm.pdf(reject_region_left, mu_0, sigma / np.sqrt(n))
        
        reject_region_right = x[x >= mu_0 + z_crit * sigma / np.sqrt(n)]
        reject_y_right = stats.norm.pdf(reject_region_right, mu_0, sigma / np.sqrt(n))
        
        fig.add_trace(
            go.Scatter(x=reject_region_left, y=reject_y_left, fill='tozeroy',
                      mode='none', name="Rejection Region",
                      fillcolor='rgba(255, 0, 0, 0.3)'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=reject_region_right, y=reject_y_right, fill='tozeroy',
                      mode='none', showlegend=False,
                      fillcolor='rgba(255, 0, 0, 0.3)'),
            row=1, col=1
        )
        
        # Add observed statistic
        fig.add_trace(
            go.Scatter(x=[sample_mean], y=[0], mode='markers',
                      marker=dict(color='red', size=10, symbol='triangle-up'),
                      name="Observed Mean"),
            row=1, col=1
        )
        
        # Add vertical line for hypothesized value
        fig.add_vline(x=mu_0, line=dict(color='green', width=2, dash='dash'),
                     annotation=dict(text=f"μ₀ = {mu_0}", showarrow=False), row=1, col=1)
        
        # Confidence interval subplot
        all_means = np.linspace(0, 10, 1000)
        ci_centers = []
        ci_contains = []
        
        for mean in all_means:
            curr_lower = mean - z_crit * sigma / np.sqrt(n)
            curr_upper = mean + z_crit * sigma / np.sqrt(n)
            ci_centers.append(mean)
            ci_contains.append(mu_0 >= curr_lower and mu_0 <= curr_upper)
        
        # Convert to numpy arrays for filtering
        ci_centers = np.array(ci_centers)
        ci_contains = np.array(ci_contains)
        
        # Add CIs that contain mu_0
        fig.add_trace(
            go.Scatter(x=ci_centers[ci_contains], y=[1]*sum(ci_contains),
                      mode='markers', marker=dict(color='green', size=5),
                      name="CIs containing μ₀"),
            row=2, col=1
        )
        
        # Add CIs that don't contain mu_0
        fig.add_trace(
            go.Scatter(x=ci_centers[~ci_contains], y=[1]*sum(~ci_contains),
                      mode='markers', marker=dict(color='red', size=5),
                      name="CIs not containing μ₀"),
            row=2, col=1
        )
        
        # Add current CI
        fig.add_trace(
            go.Scatter(x=[ci_lower, ci_upper], y=[0.9, 0.9], mode='lines',
                      line=dict(color='blue', width=3),
                      name=f"Current {(1-alpha)*100:.0f}% CI"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[sample_mean], y=[0.9], mode='markers',
                      marker=dict(color='blue', size=10),
                      name="Current Sample Mean"),
            row=2, col=1
        )
        
        # Add vertical line for hypothesized value in second plot
        fig.add_vline(x=mu_0, line=dict(color='green', width=2, dash='dash'),
                     annotation=dict(text=f"μ₀ = {mu_0}", showarrow=False), row=2, col=1)
        
        fig.update_xaxes(title_text="Sample Mean", row=1, col=1)
        fig.update_xaxes(title_text="Sample Mean", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="", showticklabels=False, row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text=f"Duality of Hypothesis Tests and Confidence Intervals<br>"
                     f"(p-value: {p_value:.4f}, {'Reject H₀' if p_value < alpha else 'Fail to Reject H₀'}, "
                     f"CI: [{ci_lower:.2f}, {ci_upper:.2f}], μ₀ {'in' if mu_0 >= ci_lower and mu_0 <= ci_upper else 'not in'} CI)",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the visualization
        if mu_0 >= ci_lower and mu_0 <= ci_upper:
            conclusion = "fail to reject H₀ and the hypothesized value falls within the confidence interval"
        else:
            conclusion = "reject H₀ and the hypothesized value falls outside the confidence interval"
            
        st.markdown(f"""
        This visualization demonstrates the duality between hypothesis testing and confidence intervals:
        
        - **Top panel**: Shows the sampling distribution under H₀: μ = {mu_0}. The red regions represent the rejection regions for a two-sided hypothesis test at α = {alpha}.
        
        - **Bottom panel**: Shows which sample means would produce confidence intervals that contain μ₀ = {mu_0} (green dots) and which would not (red dots).
        
        With your current selections:
        - Observed mean: {sample_mean}
        - Hypothesized mean: {mu_0}
        - Test statistic z = {z_stat:.4f}
        - p-value: {p_value:.4f}
        - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
        
        We {conclusion}, illustrating the duality relationship.
        """)

    with tabs[2]:  # Derivations
        st.subheader("Derivation of Common Confidence Intervals")
        
        interval_type = st.selectbox(
            "Select confidence interval type",
            ["Normal Mean (Known Variance)", "Normal Mean (Unknown Variance)", 
             "Binomial Proportion", "Difference of Means", "Variance"]
        )
        
        if interval_type == "Normal Mean (Known Variance)":
            st.markdown(r"""
            ### Confidence Interval for Normal Mean (Known Variance)
            
            For a random sample $X_1, X_2, \ldots, X_n$ from a normal distribution with unknown mean $\mu$ and known variance $\sigma^2$:
            
            <div class="proof">
            <strong>Step 1:</strong> Identify a pivotal quantity.
            
            The standardized sample mean follows a standard normal distribution:
            
            $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$$
            
            <strong>Step 2:</strong> Find critical values such that $P(-z_{\alpha/2} \leq Z \leq z_{\alpha/2}) = 1-\alpha$.
            
            $$P\left(-z_{\alpha/2} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}\right) = 1-\alpha$$
            
            <strong>Step 3:</strong> Solve for $\mu$.
            
            $$P\left(\bar{X} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) = 1-\alpha$$
            
            <strong>Step 4:</strong> The $(1-\alpha)$ confidence interval for $\mu$ is:
            
            $$\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$$
            </div>
            
            where $z_{\alpha/2}$ is the $(1-\alpha/2)$ quantile of the standard normal distribution.
            """, unsafe_allow_html=True)
            
            # Interactive demonstration
            st.subheader("Interactive Demonstration")
            
            col1, col2 = st.columns(2)
            with col1:
                demo_mean = st.slider("True population mean (μ)", -10.0, 10.0, 0.0, 0.1)
                demo_sigma = st.slider("Population standard deviation (σ)", 0.1, 5.0, 1.0, 0.1)
            with col2:
                demo_n = st.slider("Sample size (n)", 5, 100, 30)
                demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
            
            if st.button("Generate Sample and Confidence Interval", key="gen_normal_known"):
                # Generate sample
                np.random.seed(None)  # Use a different seed each time
                sample = np.random.normal(demo_mean, demo_sigma, demo_n)
                sample_mean = np.mean(sample)
                
                # Calculate CI
                z_critical = stats.norm.ppf(1 - demo_alpha/2)
                margin = z_critical * demo_sigma / np.sqrt(demo_n)
                ci_lower = sample_mean - margin
                ci_upper = sample_mean + margin
                
                # Display results
                st.markdown(f"""
                **Sample Mean**: {sample_mean:.4f}
                
                **{(1-demo_alpha)*100:.0f}% Confidence Interval**: [{ci_lower:.4f}, {ci_upper:.4f}]
                
                **Margin of Error**: {margin:.4f}
                
                **Critical value** $z_{{{demo_alpha/2}}}$: {z_critical:.4f}
                """)
                
                # Visualization
                x = np.linspace(demo_mean - 4*demo_sigma/np.sqrt(demo_n), 
                               demo_mean + 4*demo_sigma/np.sqrt(demo_n), 1000)
                sampling_dist = stats.norm.pdf(x, demo_mean, demo_sigma/np.sqrt(demo_n))
                
                fig = go.Figure()
                
                # Add sampling distribution
                fig.add_trace(
                    go.Scatter(x=x, y=sampling_dist, mode='lines', 
                              name='Sampling Distribution',
                              line=dict(color='blue'))
                )
                
                # Add vertical lines for sample mean and CI
                fig.add_vline(x=sample_mean, line=dict(color='red', width=2),
                             annotation=dict(text=f"Sample Mean", showarrow=False))
                
                fig.add_vline(x=ci_lower, line=dict(color='green', width=2, dash='dash'),
                             annotation=dict(text=f"Lower CI", showarrow=False))
                
                fig.add_vline(x=ci_upper, line=dict(color='green', width=2, dash='dash'),
                             annotation=dict(text=f"Upper CI", showarrow=False))
                
                # Add vertical line for true mean
                fig.add_vline(x=demo_mean, line=dict(color='purple', width=2, dash='dot'),
                             annotation=dict(text=f"True Mean", showarrow=False))
                
                fig.update_layout(
                    title=f"Sampling Distribution and {(1-demo_alpha)*100:.0f}% Confidence Interval",
                    xaxis_title="Mean",
                    yaxis_title="Density",
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                contains = ci_lower <= demo_mean <= ci_upper
                st.markdown(f"""
                **Interpretation**:
                
                - The {(1-demo_alpha)*100:.0f}% confidence interval is [{ci_lower:.4f}, {ci_upper:.4f}]
                - This interval {"does" if contains else "does not"} contain the true population mean ({demo_mean})
                - If we were to repeat this experiment many times, approximately {(1-demo_alpha)*100:.0f}% of the resulting confidence intervals would contain the true mean
                """)
                
                # Add demonstration of repeated sampling
                st.subheader("Demonstration of Repeated Sampling")
                
                n_repetitions = st.slider("Number of repetitions", 10, 200, 50)
                
                if st.button("Generate Multiple Intervals", key="gen_multiple_normal"):
                    # Generate multiple samples and compute CIs
                    means = []
                    lower_bounds = []
                    upper_bounds = []
                    contains_param = []
                    
                    for i in range(n_repetitions):
                        # Generate sample
                        sample = np.random.normal(demo_mean, demo_sigma, demo_n)
                        curr_mean = np.mean(sample)
                        
                        # Compute CI
                        margin = z_critical * demo_sigma / np.sqrt(demo_n)
                        lower = curr_mean - margin
                        upper = curr_mean + margin
                        
                        # Store results
                        means.append(curr_mean)
                        lower_bounds.append(lower)
                        upper_bounds.append(upper)
                        contains_param.append(lower <= demo_mean <= upper)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add horizontal line for true parameter
                    fig.add_hline(y=demo_mean, line=dict(color='purple', width=2, dash='dot'), 
                                 annotation=dict(text=f"True μ = {demo_mean}", showarrow=False))
                    
                    # Add confidence intervals
                    for i in range(n_repetitions):
                        color = 'rgba(0, 128, 0, 0.5)' if contains_param[i] else 'rgba(255, 0, 0, 0.5)'
                        fig.add_trace(
                            go.Scatter(x=[i, i], y=[lower_bounds[i], upper_bounds[i]], 
                                      mode='lines', line=dict(color=color, width=2),
                                      showlegend=False)
                        )
                        fig.add_trace(
                            go.Scatter(x=[i], y=[means[i]], mode='markers', 
                                      marker=dict(color='blue', size=6),
                                      showlegend=False)
                        )
                    
                    # Calculate coverage
                    actual_coverage = sum(contains_param) / n_repetitions * 100
                    
                    fig.update_layout(
                        title=f'Coverage of {(1-demo_alpha)*100:.0f}% Confidence Intervals<br>'
                              f'Actual coverage: {actual_coverage:.1f}% ({sum(contains_param)} out of {n_repetitions})',
                        xaxis_title='Sample Number',
                        yaxis_title='Parameter Value',
                        height=500,
                    )
                    
                    # Add legend traces
                    fig.add_trace(
                        go.Scatter(x=[None], y=[None], mode='lines', 
                                  line=dict(color='rgba(0, 128, 0, 0.5)', width=2),
                                  name="Interval contains μ")
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=[None], y=[None], mode='lines', 
                                  line=dict(color='rgba(255, 0, 0, 0.5)', width=2),
                                  name="Interval does not contain μ")
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=[None], y=[None], mode='markers', 
                                  marker=dict(color='blue', size=6),
                                  name="Sample Mean")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    This visualization shows {n_repetitions} different {(1-demo_alpha)*100:.0f}% confidence intervals constructed from different random samples.
                    
                    - Each **vertical line** represents a confidence interval
                    - **Green intervals** contain the true parameter value (μ = {demo_mean})
                    - **Red intervals** miss the true parameter value
                    - **Blue dots** represent the sample means
                    
                    Theoretically, {(1-demo_alpha)*100:.0f}% of intervals should contain the true parameter.
                    In this simulation, {actual_coverage:.1f}% ({sum(contains_param)} out of {n_repetitions}) intervals contain the true parameter.
                    """)
                
        elif interval_type == "Normal Mean (Unknown Variance)":
            st.markdown(r"""
            ### Confidence Interval for Normal Mean (Unknown Variance)
            
            For a random sample $X_1, X_2, \ldots, X_n$ from a normal distribution with unknown mean $\mu$ and unknown variance $\sigma^2$:
            
            <div class="proof">
            <strong>Step 1:</strong> Identify a pivotal quantity.
            
            Since $\sigma$ is unknown, we replace it with the sample standard deviation $S$. The resulting statistic follows a t-distribution:
            
            $$T = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$$
            
            where $t_{n-1}$ is the t-distribution with $n-1$ degrees of freedom.
            
            <strong>Step 2:</strong> Find critical values such that $P(-t_{\alpha/2, n-1} \leq T \leq t_{\alpha/2, n-1}) = 1-\alpha$.
            
            $$P\left(-t_{\alpha/2, n-1} \leq \frac{\bar{X} - \mu}{S/\sqrt{n}} \leq t_{\alpha/2, n-1}\right) = 1-\alpha$$
            
            <strong>Step 3:</strong> Solve for $\mu$.
            
            $$P\left(\bar{X} - t_{\alpha/2, n-1}\frac{S}{\sqrt{n}} \leq \mu \leq \bar{X} + t_{\alpha/2, n-1}\frac{S}{\sqrt{n}}\right) = 1-\alpha$$
            
            <strong>Step 4:</strong> The $(1-\alpha)$ confidence interval for $\mu$ is:
            
            $$\bar{X} \pm t_{\alpha/2, n-1}\frac{S}{\sqrt{n}}$$
            </div>
            
            where $t_{\alpha/2, n-1}$ is the $(1-\alpha/2)$ quantile of the t-distribution with $n-1$ degrees of freedom.
            """, unsafe_allow_html=True)
            
            # Interactive demonstration
            st.subheader("Interactive Demonstration")
            
            col1, col2 = st.columns(2)
            with col1:
                demo_mean = st.slider("True population mean (μ)", -10.0, 10.0, 0.0, 0.1)
                demo_sigma = st.slider("Population standard deviation (σ)", 0.1, 5.0, 1.0, 0.1)
            with col2:
                demo_n = st.slider("Sample size (n)", 2, 100, 10)
                demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
            
            if st.button("Generate Sample and Confidence Interval", key="gen_t_dist"):
                # Generate sample
                np.random.seed(None)  # Use a different seed each time
                sample = np.random.normal(demo_mean, demo_sigma, demo_n)
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                
                # Calculate CI
                t_critical = stats.t.ppf(1 - demo_alpha/2, demo_n - 1)
                margin = t_critical * sample_std / np.sqrt(demo_n)
                ci_lower = sample_mean - margin
                ci_upper = sample_mean + margin
                
                # Compare with z-interval
                z_critical = stats.norm.ppf(1 - demo_alpha/2)
                z_margin = z_critical * demo_sigma / np.sqrt(demo_n)
                z_ci_lower = sample_mean - z_margin
                z_ci_upper = sample_mean + z_margin
                
                # Display results
                st.markdown(f"""
                **Sample Statistics**:
                - Sample Mean: {sample_mean:.4f}
                - Sample Standard Deviation: {sample_std:.4f}
                
                **{(1-demo_alpha)*100:.0f}% Confidence Interval (t-distribution)**:
                - Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
                - Margin of Error: {margin:.4f}
                - Critical value $t_{{{demo_alpha/2}, {demo_n-1}}}$: {t_critical:.4f}
                
                **Comparison with known variance interval**:
                - Z-interval (if σ were known): [{z_ci_lower:.4f}, {z_ci_upper:.4f}]
                - Z-margin: {z_margin:.4f}
                - Critical value $z_{{{demo_alpha/2}}}$: {z_critical:.4f}
                """)
                
                # Visualization comparing t and z
                x = np.linspace(-4, 4, 1000)
                t_dist = stats.t.pdf(x, demo_n - 1)
                z_dist = stats.norm.pdf(x)
                
                fig = go.Figure()
                
                # Add t and normal distributions
                fig.add_trace(
                    go.Scatter(x=x, y=t_dist, mode='lines', 
                              name=f't-distribution (df={demo_n-1})',
                              line=dict(color='blue'))
                )
                
                fig.add_trace(
                    go.Scatter(x=x, y=z_dist, mode='lines', 
                              name='Standard Normal',
                              line=dict(color='red', dash='dash'))
                )
                
                # Add critical values
                fig.add_vline(x=t_critical, line=dict(color='blue', width=2, dash='dot'),
                             annotation=dict(text=f"t critical: {t_critical:.2f}", showarrow=False))
                
                fig.add_vline(x=z_critical, line=dict(color='red', width=2, dash='dot'),
                             annotation=dict(text=f"z critical: {z_critical:.2f}", showarrow=False))
                
                fig.add_vline(x=-t_critical, line=dict(color='blue', width=2, dash='dot'),
                             annotation=dict(text=f"-t critical", showarrow=False))
                
                fig.add_vline(x=-z_critical, line=dict(color='red', width=2, dash='dot'),
                             annotation=dict(text=f"-z critical", showarrow=False))
                
                fig.update_layout(
                    title=f"Comparison of t and Normal Distributions<br>Sample size n = {demo_n}",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show the effect of different sample sizes
                st.subheader("Effect of Sample Size on t-Distribution")
                
                # Create sample size selector
                sample_sizes = [2, 5, 10, 30, 100]
                
                fig = go.Figure()
                
                # Add t-distributions for different sample sizes
                for n in sample_sizes:
                    t_dist = stats.t.pdf(x, n - 1)
                    fig.add_trace(
                        go.Scatter(x=x, y=t_dist, mode='lines', 
                                  name=f't-distribution (df={n-1})',
                                  line=dict(width=2))
                    )
                
                # Add normal distribution
                fig.add_trace(
                    go.Scatter(x=x, y=z_dist, mode='lines', 
                              name='Standard Normal',
                              line=dict(color='black', width=3, dash='dash'))
                )
                
                fig.update_layout(
                    title="t-Distribution Approaches Normal Distribution as Sample Size Increases",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Key observations**:
                
                1. The t-distribution has heavier tails than the normal distribution, especially for small sample sizes
                2. This results in wider confidence intervals when using the t-distribution (unknown σ) compared to the z-distribution (known σ)
                3. As the sample size increases, the t-distribution approaches the normal distribution
                4. For practical purposes, when n > 30, the difference between t and z critical values becomes negligible
                
                This demonstrates why we need to use the t-distribution when estimating the population variance from the sample.
                """)
        
        elif interval_type == "Binomial Proportion":
            st.markdown(r"""
            ### Confidence Interval for Binomial Proportion
            
            For a binomial random variable with $n$ trials and unknown probability of success $p$:
            
            <div class="proof">
            <strong>Wald (Standard) Interval:</strong>
            
            This is the most commonly used interval, based on the normal approximation to the binomial distribution.
            
            For a sample proportion $\hat{p} = X/n$ where $X \sim \text{Binomial}(n, p)$:
            
            1. For large $n$, $\hat{p}$ is approximately normal with mean $p$ and variance $\frac{p(1-p)}{n}$
            
            2. The standardized statistic follows a standard normal distribution:
            
            $$Z = \frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}} \approx N(0,1)$$
            
            3. Since $p$ is unknown in the denominator, we replace it with $\hat{p}$:
            
            $$\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$
            
            <strong>Wilson Score Interval:</strong>
            
            The Wilson Score interval has better coverage properties than the Wald interval:
            
            $$\frac{\hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}}}{1 + \frac{z_{\alpha/2}^2}{n}}$$
            
            <strong>Clopper-Pearson (Exact) Interval:</strong>
            
            Based on the binomial CDF directly:
            
            Lower bound: The value of $p_L$ such that $P(X \geq x) = \alpha/2$ when $p = p_L$
            Upper bound: The value of $p_U$ such that $P(X \leq x) = \alpha/2$ when $p = p_U$
            
            This uses the relationship between the binomial and beta distributions:
            
            Lower: $\text{Beta}(\alpha/2; x, n-x+1)$
            Upper: $\text{Beta}(1-\alpha/2; x+1, n-x)$
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive demonstration
            st.subheader("Interactive Demonstration")
            
            col1, col2 = st.columns(2)
            with col1:
                demo_p = st.slider("True population proportion (p)", 0.01, 0.99, 0.5, 0.01)
                demo_n = st.slider("Sample size (n)", 5, 1000, 50)
            with col2:
                demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
                ci_method = st.radio("Confidence interval method", 
                                     ["Wald (Standard)", "Wilson Score", "Clopper-Pearson (Exact)"])
            
            if st.button("Generate Sample and Confidence Intervals", key="gen_binom"):
                # Generate sample
                np.random.seed(None)  # Use a different seed each time
                sample = np.random.binomial(1, demo_p, demo_n)
                successes = np.sum(sample)
                p_hat = successes / demo_n
                
                # Calculate CIs
                z_critical = stats.norm.ppf(1 - demo_alpha/2)
                
                # Wald interval
                if p_hat == 0 or p_hat == 1:
                    wald_lower = 0 if p_hat == 0 else 1 - (z_critical**2)/(2*demo_n)
                    wald_upper = (z_critical**2)/(2*demo_n) if p_hat == 0 else 1
                else:
                    wald_se = np.sqrt(p_hat * (1 - p_hat) / demo_n)
                    wald_lower = max(0, p_hat - z_critical * wald_se)
                    wald_upper = min(1, p_hat + z_critical * wald_se)
                
                # Wilson interval
                wilson_denominator = 1 + z_critical**2/demo_n
                wilson_center = (p_hat + z_critical**2/(2*demo_n)) / wilson_denominator
                wilson_halfwidth = z_critical * np.sqrt((p_hat*(1-p_hat) + z_critical**2/(4*demo_n)) / demo_n) / wilson_denominator
                wilson_lower = max(0, wilson_center - wilson_halfwidth)
                wilson_upper = min(1, wilson_center + wilson_halfwidth)
                
                # Clopper-Pearson interval
                if successes == 0:
                    clopper_lower = 0
                else:
                    clopper_lower = stats.beta.ppf(demo_alpha/2, successes, demo_n - successes + 1)
                
                if successes == demo_n:
                    clopper_upper = 1
                else:
                    clopper_upper = stats.beta.ppf(1 - demo_alpha/2, successes + 1, demo_n - successes)
                
                # Display results
                st.markdown(f"""
                **Sample Statistics**:
                - Number of trials (n): {demo_n}
                - Number of successes (x): {successes}
                - Sample proportion (p̂): {p_hat:.4f}
                
                **{(1-demo_alpha)*100:.0f}% Confidence Intervals**:
                
                1. **Wald (Standard) Interval**: [{wald_lower:.4f}, {wald_upper:.4f}]
                   - Width: {wald_upper - wald_lower:.4f}
                   - Contains true p: {"Yes" if wald_lower <= demo_p <= wald_upper else "No"}
                
                2. **Wilson Score Interval**: [{wilson_lower:.4f}, {wilson_upper:.4f}]
                   - Width: {wilson_upper - wilson_lower:.4f}
                   - Contains true p: {"Yes" if wilson_lower <= demo_p <= wilson_upper else "No"}
                
                3. **Clopper-Pearson (Exact) Interval**: [{clopper_lower:.4f}, {clopper_upper:.4f}]
                   - Width: {clopper_upper - clopper_lower:.4f}
                   - Contains true p: {"Yes" if clopper_lower <= demo_p <= clopper_upper else "No"}
                """)
                
                # Visualization of the different intervals
                fig = go.Figure()
                
                # Add intervals
                methods = ["Wald", "Wilson", "Clopper-Pearson"]
                lower_bounds = [wald_lower, wilson_lower, clopper_lower]
                upper_bounds = [wald_upper, wilson_upper, clopper_upper]
                colors = ['blue', 'green', 'red']
                
                for i, method in enumerate(methods):
                    fig.add_trace(
                        go.Scatter(x=[lower_bounds[i], upper_bounds[i]], 
                                   y=[i, i], 
                                   mode='lines+markers',
                                   name=method,
                                   line=dict(color=colors[i], width=3))
                    )
                
                # Add true proportion
                fig.add_vline(x=demo_p, line=dict(color='black', width=2, dash='dash'),
                             annotation=dict(text=f"True p = {demo_p}", showarrow=False))
                
                # Add sample proportion
                fig.add_vline(x=p_hat, line=dict(color='purple', width=2),
                             annotation=dict(text=f"Sample p̂ = {p_hat}", showarrow=False))
                
                fig.update_layout(
                    title=f"Comparison of {(1-demo_alpha)*100:.0f}% Confidence Intervals for Binomial Proportion",
                    xaxis_title="Proportion",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1, 2],
                        ticktext=methods
                    ),
                    height=400,
                    xaxis=dict(range=[max(0, min(lower_bounds)-0.1), min(1, max(upper_bounds)+0.1)])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation about the different methods
                st.markdown("""
                **Comparison of the methods**:
                
                1. **Wald Interval**:
                   - Simplest method based on normal approximation
                   - Often has poor coverage, especially for small n or p near 0 or 1
                   - May produce intervals outside [0,1]
                
                2. **Wilson Score Interval**:
                   - More complex formula but better coverage properties
                   - Performs well even for small n and extreme p values
                   - Always produces intervals within [0,1]
                
                3. **Clopper-Pearson (Exact) Interval**:
                   - Based directly on the binomial distribution
                   - Guaranteed to have at least the nominal coverage probability
                   - Usually wider than necessary (conservative)
                   - Computationally more intensive
                
                **Recommendations**:
                - For most applications, the Wilson interval offers a good balance of simplicity and accuracy
                - When sample size is large (n > 30) and p̂ is not extreme, the Wald interval is usually adequate
                - When conservatism is important, the Clopper-Pearson interval is preferred
                """)
                
                # Add coverage simulation
                st.subheader("Coverage Simulation")
                
                n_sims = st.slider("Number of simulations", 100, 2000, 500, 100)
                
                if st.button("Run Coverage Simulation", key="run_coverage_sim"):
                    # Run simulations
                    wald_coverage = 0
                    wilson_coverage = 0
                    clopper_coverage = 0
                    
                    for _ in range(n_sims):
                        # Generate sample
                        sample = np.random.binomial(1, demo_p, demo_n)
                        successes = np.sum(sample)
                        p_hat = successes / demo_n
                        
                        # Wald interval
                        if p_hat == 0 or p_hat == 1:
                            wald_lower = 0 if p_hat == 0 else 1 - (z_critical**2)/(2*demo_n)
                            wald_upper = (z_critical**2)/(2*demo_n) if p_hat == 0 else 1
                        else:
                            wald_se = np.sqrt(p_hat * (1 - p_hat) / demo_n)
                            wald_lower = max(0, p_hat - z_critical * wald_se)
                            wald_upper = min(1, p_hat + z_critical * wald_se)
                        
                        # Wilson interval
                        wilson_denominator = 1 + z_critical**2/demo_n
                        wilson_center = (p_hat + z_critical**2/(2*demo_n)) / wilson_denominator
                        wilson_halfwidth = z_critical * np.sqrt((p_hat*(1-p_hat) + z_critical**2/(4*demo_n)) / demo_n) / wilson_denominator
                        wilson_lower = max(0, wilson_center - wilson_halfwidth)
                        wilson_upper = min(1, wilson_center + wilson_halfwidth)
                        
                        # Clopper-Pearson interval
                        if successes == 0:
                            clopper_lower = 0
                        else:
                            clopper_lower = stats.beta.ppf(demo_alpha/2, successes, demo_n - successes + 1)
                        
                        if successes == demo_n:
                            clopper_upper = 1
                        else:
                            clopper_upper = stats.beta.ppf(1 - demo_alpha/2, successes + 1, demo_n - successes)
                        
                        # Check coverage
                        wald_coverage += (wald_lower <= demo_p <= wald_upper)
                        wilson_coverage += (wilson_lower <= demo_p <= wilson_upper)
                        clopper_coverage += (clopper_lower <= demo_p <= clopper_upper)
                    
                    # Calculate coverage percentages
                    wald_coverage_pct = wald_coverage / n_sims * 100
                    wilson_coverage_pct = wilson_coverage / n_sims * 100
                    clopper_coverage_pct = clopper_coverage / n_sims * 100
                    
                    # Create bar chart
                    coverage_data = {
                        'Method': ['Wald', 'Wilson', 'Clopper-Pearson'],
                        'Coverage': [wald_coverage_pct, wilson_coverage_pct, clopper_coverage_pct]
                    }
                    
                    fig = px.bar(
                        coverage_data, 
                        x='Method', 
                        y='Coverage',
                        text_auto='.1f',
                        color='Method',
                        color_discrete_sequence=['blue', 'green', 'red'],
                        title=f"Actual Coverage of {(1-demo_alpha)*100:.0f}% Confidence Intervals<br>({n_sims} simulations, p={demo_p}, n={demo_n})"
                    )
                    
                    # Add horizontal line for nominal coverage
                    fig.add_hline(y=(1-demo_alpha)*100, line=dict(color='black', width=2, dash='dash'),
                                 annotation=dict(text=f"Nominal {(1-demo_alpha)*100:.0f}%", 
                                               showarrow=False,
                                               yref="y",
                                               xref="paper",
                                               x=1.05))
                    
                    fig.update_layout(
                        yaxis=dict(
                            title='Actual Coverage (%)',
                            range=[min(wald_coverage_pct, wilson_coverage_pct, clopper_coverage_pct)*0.95, 
                                  max(wald_coverage_pct, wilson_coverage_pct, clopper_coverage_pct)*1.05]
                        ),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add average interval widths
                    st.markdown(f"""
                    **Coverage Results**:
                    
                    - **Wald**: {wald_coverage_pct:.1f}% (Target: {(1-demo_alpha)*100:.0f}%)
                    - **Wilson**: {wilson_coverage_pct:.1f}% (Target: {(1-demo_alpha)*100:.0f}%)
                    - **Clopper-Pearson**: {clopper_coverage_pct:.1f}% (Target: {(1-demo_alpha)*100:.0f}%)
                    
                    This simulation demonstrates the actual coverage probability of each method over repeated sampling. The Clopper-Pearson interval is generally conservative (above nominal coverage), while the Wald interval often falls below the nominal coverage, especially for small samples or extreme values of p.
                    """)
        
        elif interval_type == "Difference of Means":
            st.markdown(r"""
            ### Confidence Interval for Difference of Means
            
            For two independent random samples from normal distributions with unknown means $\mu_1$ and $\mu_2$:
            
            <div class="proof">
            <strong>Case 1: Equal and Known Variances $\sigma^2$</strong>
            
            The standardized difference follows a standard normal distribution:
            
            $$Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sigma\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim N(0,1)$$
            
            Leading to the CI:
            
            $$(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \cdot \sigma\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$
            
            <strong>Case 2: Equal but Unknown Variances</strong>
            
            We use the pooled variance estimate:
            
            $$S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}$$
            
            The test statistic follows a t-distribution:
            
            $$T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{S_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t_{n_1+n_2-2}$$
            
            Leading to the CI:
            
            $$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, n_1+n_2-2} \cdot S_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$
            
            <strong>Case 3: Unequal and Unknown Variances (Welch-Satterthwaite)</strong>
            
            The test statistic approximately follows a t-distribution:
            
            $$T' = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}$$
            
            with degrees of freedom approximated by:
            
            $$\nu \approx \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}$$
            
            Leading to the CI:
            
            $$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \nu} \cdot \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}$$
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive demonstration
            st.subheader("Interactive Demonstration")
            
            col1, col2 = st.columns(2)
            with col1:
                mean1 = st.slider("Group 1 true mean (μ₁)", -10.0, 10.0, 5.0, 0.1)
                sigma1 = st.slider("Group 1 true std dev (σ₁)", 0.1, 5.0, 2.0, 0.1)
                n1 = st.slider("Group 1 sample size (n₁)", 5, 100, 30)
            
            with col2:
                mean2 = st.slider("Group 2 true mean (μ₂)", -10.0, 10.0, 3.0, 0.1)
                sigma2 = st.slider("Group 2 true std dev (σ₂)", 0.1, 5.0, 2.0, 0.1)
                n2 = st.slider("Group 2 sample size (n₂)", 5, 100, 30)
            
            alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
            
            method = st.radio("Method", 
                           ["Pooled variance (equal variances)", 
                            "Welch-Satterthwaite (unequal variances)"])
            
            if st.button("Generate Samples and Confidence Interval", key="gen_diff_means"):
                # Generate samples
                np.random.seed(None)  # Use a different seed each time
                sample1 = np.random.normal(mean1, sigma1, n1)
                sample2 = np.random.normal(mean2, sigma2, n2)
                
                # Calculate sample statistics
                xbar1 = np.mean(sample1)
                xbar2 = np.mean(sample2)
                s1 = np.std(sample1, ddof=1)
                s2 = np.std(sample2, ddof=1)
                mean_diff = xbar1 - xbar2
                true_diff = mean1 - mean2
                
                # Calculate CIs based on method
                if method == "Pooled variance (equal variances)":
                    # Pooled variance
                    # Pooled variance
                    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
                    sp = np.sqrt(sp2)
                    
                    # Standard error
                    se = sp * np.sqrt(1/n1 + 1/n2)
                    
                    # Critical value
                    t_crit = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
                    
                    # CI
                    margin = t_crit * se
                    ci_lower = mean_diff - margin
                    ci_upper = mean_diff + margin
                    
                    method_description = f"Pooled variance t-interval with {n1 + n2 - 2} df"
                    
                else:  # Welch-Satterthwaite
                    # Standard error
                    se = np.sqrt(s1**2/n1 + s2**2/n2)
                    
                    # Degrees of freedom (Welch-Satterthwaite)
                    df_num = (s1**2/n1 + s2**2/n2)**2
                    df_denom = (s1**4/(n1**2 * (n1-1))) + (s2**4/(n2**2 * (n2-1)))
                    df = df_num / df_denom
                    
                    # Critical value
                    t_crit = stats.t.ppf(1 - alpha/2, df)
                    
                    # CI
                    margin = t_crit * se
                    ci_lower = mean_diff - margin
                    ci_upper = mean_diff + margin
                    
                    method_description = f"Welch-Satterthwaite t-interval with {df:.1f} df"
                
                # Display results
                st.markdown(f"""
                **Sample Statistics**:
                
                |  | Group 1 | Group 2 |
                |--|---------|---------|
                | Sample Size | {n1} | {n2} |
                | Sample Mean | {xbar1:.4f} | {xbar2:.4f} |
                | Sample Std Dev | {s1:.4f} | {s2:.4f} |
                
                **Difference in Means**:
                - Sample difference (x̄₁ - x̄₂): {mean_diff:.4f}
                - True difference (μ₁ - μ₂): {true_diff:.4f}
                
                **{(1-alpha)*100:.0f}% Confidence Interval**:
                - Method: {method_description}
                - Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
                - Margin of Error: {margin:.4f}
                - Critical value: {t_crit:.4f}
                - Standard Error: {se:.4f}
                
                The interval {"does" if ci_lower <= true_diff <= ci_upper else "does not"} contain the true difference.
                """)
                
                # Visualization
                fig = go.Figure()
                
                # Add density plots for both samples
                x1_range = np.linspace(min(sample1), max(sample1), 100)
                x2_range = np.linspace(min(sample2), max(sample2), 100)
                
                kde1 = stats.gaussian_kde(sample1)
                kde2 = stats.gaussian_kde(sample2)
                
                fig.add_trace(
                    go.Scatter(x=x1_range, y=kde1(x1_range), mode='lines',
                              name='Group 1 Distribution',
                              line=dict(color='blue', width=2))
                )
                
                fig.add_trace(
                    go.Scatter(x=x2_range, y=kde2(x2_range), mode='lines',
                              name='Group 2 Distribution',
                              line=dict(color='red', width=2))
                )
                
                # Add vertical lines for means
                fig.add_vline(x=xbar1, line=dict(color='blue', width=2, dash='dash'),
                             annotation=dict(text=f"x̄₁ = {xbar1:.2f}", showarrow=False))
                
                fig.add_vline(x=xbar2, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"x̄₂ = {xbar2:.2f}", showarrow=False))
                
                fig.update_layout(
                    title="Sample Distributions with Means",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualization of the difference
                diff_fig = go.Figure()
                
                # Create a range for the difference
                diff_range = np.linspace(ci_lower - margin, ci_upper + margin, 1000)
                
                # If using pooled variance
                if method == "Pooled variance (equal variances)":
                    # Distribution of the difference under t with pooled df
                    diff_dist = stats.t.pdf((diff_range - mean_diff) / se, n1 + n2 - 2) / se
                else:
                    # Distribution of the difference under t with Welch-Satterthwaite df
                    diff_dist = stats.t.pdf((diff_range - mean_diff) / se, df) / se
                
                diff_fig.add_trace(
                    go.Scatter(x=diff_range, y=diff_dist, mode='lines',
                              name='Sampling Distribution of Difference',
                              line=dict(color='purple', width=2))
                )
                
                # Add vertical lines for CI bounds
                diff_fig.add_vline(x=ci_lower, line=dict(color='green', width=2, dash='dash'),
                                 annotation=dict(text=f"Lower CI = {ci_lower:.2f}", showarrow=False))
                
                diff_fig.add_vline(x=ci_upper, line=dict(color='green', width=2, dash='dash'),
                                 annotation=dict(text=f"Upper CI = {ci_upper:.2f}", showarrow=False))
                
                # Add vertical line for observed difference
                diff_fig.add_vline(x=mean_diff, line=dict(color='purple', width=2),
                                 annotation=dict(text=f"Observed Diff = {mean_diff:.2f}", showarrow=False))
                
                # Add vertical line for true difference
                diff_fig.add_vline(x=true_diff, line=dict(color='black', width=2, dash='dot'),
                                 annotation=dict(text=f"True Diff = {true_diff:.2f}", showarrow=False))
                
                # Shade the confidence interval region
                ci_x = np.linspace(ci_lower, ci_upper, 100)
                if method == "Pooled variance (equal variances)":
                    ci_y = stats.t.pdf((ci_x - mean_diff) / se, n1 + n2 - 2) / se
                else:
                    ci_y = stats.t.pdf((ci_x - mean_diff) / se, df) / se
                
                diff_fig.add_trace(
                    go.Scatter(x=ci_x, y=ci_y, fill='tozeroy',
                              mode='none', name=f"{(1-alpha)*100:.0f}% CI",
                              fillcolor='rgba(0, 128, 0, 0.3)')
                )
                
                diff_fig.update_layout(
                    title=f"{(1-alpha)*100:.0f}% Confidence Interval for Difference in Means",
                    xaxis_title="Difference (Group 1 - Group 2)",
                    yaxis_title="Density",
                    height=400,
                )
                
                st.plotly_chart(diff_fig, use_container_width=True)
                
                # Explanation
                st.markdown(f"""
                **Interpretation**:
                
                1. The first plot shows the distributions of the two samples with their respective means.
                
                2. The second plot shows:
                   - The **purple curve** represents the sampling distribution of the difference in means
                   - The **green shaded area** represents the {(1-alpha)*100:.0f}% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]
                   - The **purple vertical line** shows the observed difference: {mean_diff:.4f}
                   - The **black dotted line** shows the true difference: {true_diff:.4f}
                
                3. Since the true difference {"is" if ci_lower <= true_diff <= ci_upper else "is not"} contained within the confidence interval, this particular interval {"correctly" if ci_lower <= true_diff <= ci_upper else "incorrectly"} captures the true parameter.
                
                **Key insights about difference of means CIs**:
                
                1. When variances are equal, the pooled variance method provides more precise intervals (narrower).
                
                2. When variances are unequal, especially with unequal sample sizes, the Welch-Satterthwaite method provides better coverage.
                
                3. The width of the interval depends on:
                   - Sample sizes (larger samples → narrower intervals)
                   - Sample variability (more variability → wider intervals)
                   - Confidence level (higher confidence → wider intervals)
                """)
                
                # Add simulation of coverage
                st.subheader("Coverage Simulation")
                
                n_sims = st.slider("Number of simulations", 100, 2000, 500, 100)
                
                if st.button("Run Coverage Simulation", key="run_diff_coverage_sim"):
                    # Run simulations
                    pooled_coverage = 0
                    welch_coverage = 0
                    pooled_widths = []
                    welch_widths = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(n_sims):
                        # Update progress
                        progress_bar.progress((i + 1) / n_sims)
                        status_text.text(f"Running simulation {i+1}/{n_sims}")
                        
                        # Generate samples
                        sample1 = np.random.normal(mean1, sigma1, n1)
                        sample2 = np.random.normal(mean2, sigma2, n2)
                        
                        # Calculate sample statistics
                        xbar1 = np.mean(sample1)
                        xbar2 = np.mean(sample2)
                        s1 = np.std(sample1, ddof=1)
                        s2 = np.std(sample2, ddof=1)
                        mean_diff = xbar1 - xbar2
                        
                        # Pooled method
                        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
                        sp = np.sqrt(sp2)
                        se_pooled = sp * np.sqrt(1/n1 + 1/n2)
                        t_crit_pooled = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
                        margin_pooled = t_crit_pooled * se_pooled
                        ci_lower_pooled = mean_diff - margin_pooled
                        ci_upper_pooled = mean_diff + margin_pooled
                        pooled_widths.append(ci_upper_pooled - ci_lower_pooled)
                        pooled_coverage += (ci_lower_pooled <= true_diff <= ci_upper_pooled)
                        
                        # Welch method
                        se_welch = np.sqrt(s1**2/n1 + s2**2/n2)
                        df_num = (s1**2/n1 + s2**2/n2)**2
                        df_denom = (s1**4/(n1**2 * (n1-1))) + (s2**4/(n2**2 * (n2-1)))
                        df_welch = df_num / df_denom
                        t_crit_welch = stats.t.ppf(1 - alpha/2, df_welch)
                        margin_welch = t_crit_welch * se_welch
                        ci_lower_welch = mean_diff - margin_welch
                        ci_upper_welch = mean_diff + margin_welch
                        welch_widths.append(ci_upper_welch - ci_lower_welch)
                        welch_coverage += (ci_lower_welch <= true_diff <= ci_upper_welch)
                    
                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Calculate coverage percentages
                    pooled_coverage_pct = pooled_coverage / n_sims * 100
                    welch_coverage_pct = welch_coverage / n_sims * 100
                    
                    # Calculate average widths
                    pooled_avg_width = np.mean(pooled_widths)
                    welch_avg_width = np.mean(welch_widths)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Method': ['Pooled Variance', 'Welch-Satterthwaite'],
                        'Coverage (%)': [pooled_coverage_pct, welch_coverage_pct],
                        'Average Width': [pooled_avg_width, welch_avg_width],
                        'Count': [pooled_coverage, welch_coverage]
                    })
                    
                    # Display results
                    st.dataframe(results_df.style.format({
                        'Coverage (%)': '{:.1f}',
                        'Average Width': '{:.4f}'
                    }))
                    
                    # Create bar chart
                    fig = px.bar(
                        results_df, 
                        x='Method', 
                        y='Coverage (%)',
                        text_auto='.1f',
                        color='Method',
                        title=f"Actual Coverage of {(1-alpha)*100:.0f}% CIs for Difference in Means<br>({n_sims} simulations)"
                    )
                    
                    # Add horizontal line for nominal coverage
                    fig.add_hline(y=(1-alpha)*100, line=dict(color='black', width=2, dash='dash'),
                                 annotation=dict(text=f"Nominal {(1-alpha)*100:.0f}%", 
                                               showarrow=False,
                                               yref="y",
                                               xref="paper",
                                               x=1.05))
                    
                    fig.update_layout(
                        yaxis=dict(
                            title='Actual Coverage (%)',
                            range=[min(pooled_coverage_pct, welch_coverage_pct)*0.95, 
                                  max(pooled_coverage_pct, welch_coverage_pct)*1.05]
                        ),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create width comparison
                    width_df = pd.DataFrame({
                        'Method': ['Pooled Variance', 'Welch-Satterthwaite'],
                        'Average Width': [pooled_avg_width, welch_avg_width]
                    })
                    
                    width_fig = px.bar(
                        width_df, 
                        x='Method', 
                        y='Average Width',
                        text_auto='.4f',
                        color='Method',
                        title=f"Average Width of {(1-alpha)*100:.0f}% CIs for Difference in Means<br>({n_sims} simulations)"
                    )
                    
                    width_fig.update_layout(
                        yaxis=dict(title='Average Interval Width'),
                        height=400
                    )
                    
                    st.plotly_chart(width_fig, use_container_width=True)
                    
                    # Explanation based on simulation results
                    equal_var = np.isclose(sigma1, sigma2, rtol=0.1)
                    
                    if equal_var:
                        comparison_text = """
                        With equal population variances, the pooled variance method typically provides narrower intervals while maintaining good coverage.
                        """
                    else:
                        comparison_text = """
                        With unequal population variances, the Welch-Satterthwaite method typically provides better coverage, though often at the cost of wider intervals.
                        """
                    
                    st.markdown(f"""
                    **Simulation Results**:
                    
                    1. **Coverage**:
                       - Pooled Variance: {pooled_coverage_pct:.1f}% (Target: {(1-alpha)*100:.0f}%)
                       - Welch-Satterthwaite: {welch_coverage_pct:.1f}% (Target: {(1-alpha)*100:.0f}%)
                    
                    2. **Interval Width**:
                       - Pooled Variance: {pooled_avg_width:.4f} (average)
                       - Welch-Satterthwaite: {welch_avg_width:.4f} (average)
                    
                    {comparison_text}
                    
                    **Recommendations**:
                    - When variances are equal or nearly equal: Use the pooled variance method for more precision
                    - When variances are clearly unequal, especially with unequal sample sizes: Use the Welch-Satterthwaite method
                    - In practice, the Welch-Satterthwaite method is often used by default as it is more robust
                    """)
        
        elif interval_type == "Variance":
            st.markdown(r"""
            ### Confidence Interval for Variance
            
            For a random sample $X_1, X_2, \ldots, X_n$ from a normal distribution with unknown variance $\sigma^2$:
            
            <div class="proof">
            <strong>Step 1:</strong> Identify a pivotal quantity.
            
            Under normality, the standardized sample variance follows a chi-square distribution:
            
            $$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$$
            
            <strong>Step 2:</strong> Find critical values such that:
            
            $$P\left(\chi^2_{\alpha/2, n-1} \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, n-1}\right) = 1-\alpha$$
            
            <strong>Step 3:</strong> Solve for $\sigma^2$.
            
            $$P\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}} \leq \sigma^2 \leq \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right) = 1-\alpha$$
            
            <strong>Step 4:</strong> The $(1-\alpha)$ confidence interval for $\sigma^2$ is:
            
            $$\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}, \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right]$$
            
            For the standard deviation $\sigma$, take the square root of the endpoints.
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive demonstration
            st.subheader("Interactive Demonstration")
            
            col1, col2 = st.columns(2)
            with col1:
                demo_mean = st.slider("True population mean (μ)", -10.0, 10.0, 0.0, 0.1)
                demo_sigma = st.slider("True population standard deviation (σ)", 0.1, 10.0, 5.0, 0.1)
            with col2:
                demo_n = st.slider("Sample size (n)", 5, 100, 20)
                demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
            
            parameter = st.radio("Parameter of interest", ["Variance (σ²)", "Standard Deviation (σ)"])
            
            if st.button("Generate Sample and Confidence Interval", key="gen_var"):
                # Generate sample
                np.random.seed(None)  # Use a different seed each time
                sample = np.random.normal(demo_mean, demo_sigma, demo_n)
                
                # Calculate sample statistics
                sample_mean = np.mean(sample)
                sample_var = np.var(sample, ddof=1)
                sample_std = np.sqrt(sample_var)
                
                # Calculate chi-square critical values
                chi2_lower = stats.chi2.ppf(demo_alpha/2, demo_n - 1)
                chi2_upper = stats.chi2.ppf(1 - demo_alpha/2, demo_n - 1)
                
                # Calculate CI for variance
                var_lower = (demo_n - 1) * sample_var / chi2_upper
                var_upper = (demo_n - 1) * sample_var / chi2_lower
                
                # Calculate CI for standard deviation
                std_lower = np.sqrt(var_lower)
                std_upper = np.sqrt(var_upper)
                
                # Display results
                st.markdown(f"""
                **Sample Statistics**:
                - Sample size (n): {demo_n}
                - Sample mean (x̄): {sample_mean:.4f}
                - Sample variance (s²): {sample_var:.4f}
                - Sample standard deviation (s): {sample_std:.4f}
                
                **Chi-square critical values**:
                - Lower critical value (χ²<sub>{demo_alpha/2},{demo_n-1}</sub>): {chi2_lower:.4f}
                - Upper critical value (χ²<sub>{1-demo_alpha/2},{demo_n-1}</sub>): {chi2_upper:.4f}
                
                **{(1-demo_alpha)*100:.0f}% Confidence Intervals**:
                - For variance (σ²): [{var_lower:.4f}, {var_upper:.4f}]
                - For standard deviation (σ): [{std_lower:.4f}, {std_upper:.4f}]
                """, unsafe_allow_html=True)
                
                # Determine if interval contains true parameter
                var_contains = var_lower <= demo_sigma**2 <= var_upper
                std_contains = std_lower <= demo_sigma <= std_upper
                
                param_text = "variance" if parameter == "Variance (σ²)" else "standard deviation"
                true_value = demo_sigma**2 if parameter == "Variance (σ²)" else demo_sigma
                lower_bound = var_lower if parameter == "Variance (σ²)" else std_lower
                upper_bound = var_upper if parameter == "Variance (σ²)" else std_upper
                contains = var_contains if parameter == "Variance (σ²)" else std_contains
                
                st.markdown(f"""
                The {(1-demo_alpha)*100:.0f}% confidence interval for the {param_text} [{lower_bound:.4f}, {upper_bound:.4f}] {"contains" if contains else "does not contain"} the true {param_text} ({true_value:.4f}).
                """)
                
                # Visualization of chi-square distribution
                x = np.linspace(max(0.01, stats.chi2.ppf(0.001, demo_n - 1)), 
                               stats.chi2.ppf(0.999, demo_n - 1), 1000)
                chi2_pdf = stats.chi2.pdf(x, demo_n - 1)
                
                fig = go.Figure()
                
                # Add chi-square distribution
                fig.add_trace(
                    go.Scatter(x=x, y=chi2_pdf, mode='lines', 
                              name=f'χ²({demo_n-1}) Distribution',
                              line=dict(color='blue', width=2))
                )
                
                # Add critical values
                fig.add_vline(x=chi2_lower, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"χ²<sub>{demo_alpha/2}</sub> = {chi2_lower:.2f}", showarrow=False))
                
                fig.add_vline(x=chi2_upper, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"χ²<sub>{1-demo_alpha/2}</sub> = {chi2_upper:.2f}", showarrow=False))
                
                # Add observed value
                observed_chi2 = (demo_n - 1) * sample_var / (demo_sigma**2)
                
                fig.add_vline(x=observed_chi2, line=dict(color='green', width=2),
                             annotation=dict(text=f"Observed: {observed_chi2:.2f}", showarrow=False))
                
                # Shade the middle area
                middle_x = x[(x >= chi2_lower) & (x <= chi2_upper)]
                middle_y = stats.chi2.pdf(middle_x, demo_n - 1)
                
                fig.add_trace(
                    go.Scatter(x=middle_x, y=middle_y, fill='tozeroy',
                              mode='none', name=f"{(1-demo_alpha)*100:.0f}% Probability",
                              fillcolor='rgba(0, 128, 0, 0.3)')
                )
                
                fig.update_layout(
                    title=f"Chi-Square Distribution (df = {demo_n-1}) with Critical Values",
                    xaxis_title="Chi-Square Value",
                    yaxis_title="Density",
                    height=400,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create visualization for the chosen parameter
                if parameter == "Variance (σ²)":
                    param_range = np.linspace(var_lower * 0.5, var_upper * 1.5, 1000)
                    param_name = "Variance (σ²)"
                    true_param = demo_sigma**2
                    estimated_param = sample_var
                    ci_lower = var_lower
                    ci_upper = var_upper
                else:
                    param_range = np.linspace(std_lower * 0.5, std_upper * 1.5, 1000)
                    param_name = "Standard Deviation (σ)"
                    true_param = demo_sigma
                    estimated_param = sample_std
                    ci_lower = std_lower
                    ci_upper = std_upper
                
                # Create the figure
                param_fig = go.Figure()
                
                # Add shaded CI region
                param_fig.add_vrect(
                    x0=ci_lower, x1=ci_upper,
                    fillcolor="rgba(0, 128, 0, 0.2)",
                    layer="below", line_width=0,
                    annotation=dict(text=f"{(1-demo_alpha)*100:.0f}% CI", showarrow=False),
                    annotation_position="top left"
                )
                
                # Add vertical lines
                param_fig.add_vline(x=estimated_param, line=dict(color='blue', width=2),
                                  annotation=dict(text=f"Sample Estimate: {estimated_param:.4f}", showarrow=False))
                
                param_fig.add_vline(x=true_param, line=dict(color='red', width=2, dash='dash'),
                                  annotation=dict(text=f"True Value: {true_param:.4f}", showarrow=False))
                
                param_fig.add_vline(x=ci_lower, line=dict(color='green', width=2, dash='dot'),
                                  annotation=dict(text=f"Lower Bound: {ci_lower:.4f}", showarrow=False))
                
                param_fig.add_vline(x=ci_upper, line=dict(color='green', width=2, dash='dot'),
                                  annotation=dict(text=f"Upper Bound: {ci_upper:.4f}", showarrow=False))
                
                param_fig.update_layout(
                    title=f"{(1-demo_alpha)*100:.0f}% Confidence Interval for {param_name}",
                    xaxis_title=param_name,
                    yaxis=dict(showticklabels=False),
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(param_fig, use_container_width=True)
                
                # Add coverage simulation
                st.subheader("Coverage Simulation")
                
                n_sims = st.slider("Number of simulations", 100, 2000, 500, 100)
                
                if st.button("Run Coverage Simulation", key="run_var_coverage_sim"):
                    # Run simulations
                    var_coverage = 0
                    std_coverage = 0
                    interval_widths_var = []
                    interval_widths_std = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(n_sims):
                        # Update progress
                        progress_bar.progress((i + 1) / n_sims)
                        status_text.text(f"Running simulation {i+1}/{n_sims}")
                        
                        # Generate sample
                        sample = np.random.normal(demo_mean, demo_sigma, demo_n)
                        
                        # Calculate sample statistics
                        sample_var = np.var(sample, ddof=1)
                        
                        # Calculate CI for variance
                        var_lower = (demo_n - 1) * sample_var / chi2_upper
                        var_upper = (demo_n - 1) * sample_var / chi2_lower
                        
                        # Calculate CI for standard deviation
                        std_lower = np.sqrt(var_lower)
                        std_upper = np.sqrt(var_upper)
                        
                        # Check coverage
                        var_coverage += (var_lower <= demo_sigma**2 <= var_upper)
                        std_coverage += (std_lower <= demo_sigma <= std_upper)
                        
                        # Calculate interval widths
                        interval_widths_var.append(var_upper - var_lower)
                        interval_widths_std.append(std_upper - std_lower)
                    
                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Calculate coverage percentages
                    var_coverage_pct = var_coverage / n_sims * 100
                    std_coverage_pct = std_coverage / n_sims * 100
                    
                    # Calculate average widths
                    var_avg_width = np.mean(interval_widths_var)
                    std_avg_width = np.mean(interval_widths_std)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Parameter': ['Variance (σ²)', 'Standard Deviation (σ)'],
                        'Coverage (%)': [var_coverage_pct, std_coverage_pct],
                        'Average Width': [var_avg_width, std_avg_width],
                        'True Value': [demo_sigma**2, demo_sigma]
                    })
                    
                    # Display results
                    st.dataframe(results_df.style.format({
                        'Coverage (%)': '{:.1f}',
                        'Average Width': '{:.4f}',
                        'True Value': '{:.4f}'
                    }))
                    
                    # Create bar chart
                    fig = px.bar(
                        results_df, 
                        x='Parameter', 
                        y='Coverage (%)',
                        text_auto='.1f',
                        color='Parameter',
                        title=f"Actual Coverage of {(1-demo_alpha)*100:.0f}% Confidence Intervals<br>({n_sims} simulations)"
                    )
                    
                    # Add horizontal line for nominal coverage
                    fig.add_hline(y=(1-demo_alpha)*100, line=dict(color='black', width=2, dash='dash'),
                                 annotation=dict(text=f"Nominal {(1-demo_alpha)*100:.0f}%", 
                                               showarrow=False,
                                               yref="y",
                                               xref="paper",
                                               x=1.05))
                    
                    fig.update_layout(
                        yaxis=dict(
                            title='Actual Coverage (%)',
                            range=[min(var_coverage_pct, std_coverage_pct)*0.95, 
                                  max(var_coverage_pct, std_coverage_pct)*1.05]
                        ),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display analysis of the simulation
                    st.markdown(f"""
                    **Simulation Results**:
                    
                    1. **Coverage**:
                       - Variance (σ²): {var_coverage_pct:.1f}% (Target: {(1-demo_alpha)*100:.0f}%)
                       - Standard Deviation (σ): {std_coverage_pct:.1f}% (Target: {(1-demo_alpha)*100:.0f}%)
                    
                    2. **Average Interval Width**:
                       - Variance (σ²): {var_avg_width:.4f}
                       - Standard Deviation (σ): {std_avg_width:.4f}
                    
                    **Key insights**:
                    
                    1. The confidence interval for variance (σ²) is asymmetric around the sample variance.
                    
                    2. The confidence interval for standard deviation (σ) is not simply the square root of the endpoints of the variance interval.
                    
                    3. These intervals are generally wider relative to the parameter value than intervals for the mean, reflecting the greater uncertainty in estimating variance parameters.
                    
                    4. Coverage properties are typically good when the underlying distribution is normal, but may deteriorate for non-normal distributions.
                    """)
                    
                    # Add non-normality note
                    st.info("""
                    **Note on robustness to non-normality**: 
                    
                    The chi-square-based confidence interval for variance relies heavily on the normality assumption. When data is non-normal (especially with heavy tails or skewness), these intervals can have poor coverage properties. In such cases, bootstrap methods or transformations may be more appropriate for variance estimation.
                    """)

    with tabs[3]:  # Optimality
        st.subheader("Optimality of Confidence Intervals")
        
        st.markdown(r"""
        Optimality criteria for confidence intervals help us determine which of many possible intervals is "best." Several common criteria include:
        
        ### 1. Minimum Width
        
        For a fixed confidence level, we seek the interval with the shortest expected length.
        
        <div class="definition">
        <strong>Definition:</strong> A confidence interval $[L(X), U(X)]$ has minimum width if, for any other interval $[L'(X), U'(X)]$ with the same coverage probability:
        
        $$E_{\theta}[U(X) - L(X)] \leq E_{\theta}[U'(X) - L'(X)]$$
        </div>
        
        ### 2. Equal-Tailed Intervals
        
        These intervals allocate the error probability equally to both tails.
        
        <div class="definition">
        <strong>Definition:</strong> A $(1-\alpha)$ confidence interval $[L(X), U(X)]$ is equal-tailed if:
        
        $$P_{\theta}(\theta < L(X)) = P_{\theta}(\theta > U(X)) = \alpha/2$$
        </div>
        
        ### 3. Unbiased Confidence Intervals
        
        <div class="definition">
        <strong>Definition:</strong> A confidence interval $[L(X), U(X)]$ is unbiased if, for $\theta_1 < \theta_2$:
        
        $$P_{\theta_1}(\theta_2 \in [L(X), U(X)]) \leq P_{\theta_2}(\theta_2 \in [L(X), U(X)])$$
        $$P_{\theta_2}(\theta_1 \in [L(X), U(X)]) \leq P_{\theta_1}(\theta_1 \in [L(X), U(X)])$$
        </div>
        
        ### 4. Uniformly Most Accurate (UMA)
        
        <div class="definition">
        <strong>Definition:</strong> A confidence interval $[L(X), U(X)]$ is UMA if, for any other interval $[L'(X), U'(X)]$ with the same coverage probability and for all $\theta^* \neq \theta$:
        
        $$P_{\theta}(\theta^* \in [L(X), U(X)]) \leq P_{\theta}(\theta^* \in [L'(X), U'(X)])$$
        </div>
        
        ### Example: Normal Mean
        
        For a normal sample with unknown mean $\mu$ and known variance $\sigma^2$, the interval:
        
        $$\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$$
        
        is optimal under all of the criteria mentioned above. It has:
        
        1. Minimum width among all intervals with the same coverage
        2. Equal tail probabilities of $\alpha/2$
        3. Unbiased coverage probabilities
        4. Uniformly most accurate for any $\mu^* \neq \mu$
        """, unsafe_allow_html=True)
        
        # Visualization of optimality
        st.subheader("Visualization of Equal-Tailed vs. Shortest Interval")
        
        # For a skewed distribution, compare equal-tailed and shortest intervals
        dist_type = st.radio("Distribution type", ["Normal", "Chi-square", "Beta"])
        
        if dist_type == "Normal":
            mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
            sigma = st.slider("Standard deviation (σ)", 0.1, 5.0, 1.0, 0.1)
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            y = stats.norm.pdf(x, mu, sigma)
            equal_lower = stats.norm.ppf(0.025, mu, sigma)
            equal_upper = stats.norm.ppf(0.975, mu, sigma)
            # For normal, shortest = equal-tailed
            shortest_lower = equal_lower
            shortest_upper = equal_upper
            
        elif dist_type == "Chi-square":
            df = st.slider("Degrees of freedom", 1, 30, 5)
            x = np.linspace(0, stats.chi2.ppf(0.999, df), 1000)
            y = stats.chi2.pdf(x, df)
            equal_lower = stats.chi2.ppf(0.025, df)
            equal_upper = stats.chi2.ppf(0.975, df)
            
            # Find the shortest interval
            def chi2_interval_length(lower_prob):
                lower = stats.chi2.ppf(lower_prob, df)
                upper = stats.chi2.ppf(lower_prob + 0.95, df)
                return upper - lower
            
            # Minimize interval length
            result = minimize(chi2_interval_length, 0.025, bounds=[(0, 0.05)])
            optimal_lower_prob = result.x[0]
            shortest_lower = stats.chi2.ppf(optimal_lower_prob, df)
            shortest_upper = stats.chi2.ppf(optimal_lower_prob + 0.95, df)
            
        elif dist_type == "Beta":
            alpha = st.slider("Alpha parameter", 0.1, 10.0, 2.0, 0.1)
            beta = st.slider("Beta parameter", 0.1, 10.0, 5.0, 0.1)
            x = np.linspace(0, 1, 1000)
            y = stats.beta.pdf(x, alpha, beta)
            equal_lower = stats.beta.ppf(0.025, alpha, beta)
            equal_upper = stats.beta.ppf(0.975, alpha, beta)
            
            # Find the shortest interval
            def beta_interval_length(lower_prob):
                lower = stats.beta.ppf(lower_prob, alpha, beta)
                upper = stats.beta.ppf(lower_prob + 0.95, alpha, beta)
                return upper - lower
            
            # Minimize interval length
            result = minimize(beta_interval_length, 0.025, bounds=[(0, 0.05)])
            optimal_lower_prob = result.x[0]
            shortest_lower = stats.beta.ppf(optimal_lower_prob, alpha, beta)
            shortest_upper = stats.beta.ppf(optimal_lower_prob + 0.95, alpha, beta)
        
        # Create visualization
        fig = go.Figure()
        
        # Add PDF curve
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name='Distribution',
                      line=dict(color='blue', width=2))
        )
        
        # Add equal-tailed interval
        equal_x = x[(x >= equal_lower) & (x <= equal_upper)]
        equal_y = y[(x >= equal_lower) & (x <= equal_upper)]
        
        fig.add_trace(
            go.Scatter(x=equal_x, y=equal_y, fill='tozeroy',
                      mode='none', name='Equal-tailed 95% CI',
                      fillcolor='rgba(255, 0, 0, 0.3)')
        )
        
        fig.add_vline(x=equal_lower, line=dict(color='red', width=2, dash='dash'),
                     annotation=dict(text=f"Equal-tailed lower: {equal_lower:.3f}", showarrow=False))
        
        fig.add_vline(x=equal_upper, line=dict(color='red', width=2, dash='dash'),
                     annotation=dict(text=f"Equal-tailed upper: {equal_upper:.3f}", showarrow=False))
        
        # Add shortest interval if different
        if not np.isclose(shortest_lower, equal_lower) or not np.isclose(shortest_upper, equal_upper):
            shortest_x = x[(x >= shortest_lower) & (x <= shortest_upper)]
            shortest_y = y[(x >= shortest_lower) & (x <= shortest_upper)]
            
            fig.add_trace(
                go.Scatter(x=shortest_x, y=shortest_y, fill='tozeroy',
                          mode='none', name='Shortest 95% CI',
                          fillcolor='rgba(0, 255, 0, 0.3)')
            )
            
            fig.add_vline(x=shortest_lower, line=dict(color='green', width=2, dash='dash'),
                         annotation=dict(text=f"Shortest lower: {shortest_lower:.3f}", showarrow=False))
            
            fig.add_vline(x=shortest_upper, line=dict(color='green', width=2, dash='dash'),
                         annotation=dict(text=f"Shortest upper: {shortest_upper:.3f}", showarrow=False))
        
        fig.update_layout(
            title=f"Comparison of Equal-Tailed and Shortest 95% Confidence Intervals<br>{dist_type} Distribution",
            xaxis_title="Value",
            yaxis_title="Density",
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interval widths
        equal_width = equal_upper - equal_lower
        shortest_width = shortest_upper - shortest_lower
        width_diff = equal_width - shortest_width
        
        if np.isclose(width_diff, 0):
            st.markdown(f"""
            For this {dist_type} distribution, the equal-tailed and shortest intervals are identical, with width = {equal_width:.4f}.
            
            **Insight**: For symmetric distributions like the normal, the equal-tailed interval is also the shortest possible interval.
            """)
        else:
            st.markdown(f"""
            **Interval Widths**:
            - Equal-tailed interval: {equal_width:.4f}
            - Shortest interval: {shortest_width:.4f}
            - Difference: {width_diff:.4f} ({width_diff/equal_width*100:.2f}% reduction in width)
            
            **Insight**: For skewed distributions like the {dist_type}, the shortest interval is not equal-tailed. The shortest interval shifts toward regions of higher density.
            """)
        
        # Add an explanation of why this matters
        st.markdown("""
        ### Why Optimality Matters
        
        The choice of optimality criterion depends on the specific application and goals:
        
        1. **Minimum width** is desirable when precision is paramount
        
        2. **Equal-tailed intervals** are often preferred for their simplicity of interpretation and symmetry in error allocation
        
        3. **Unbiased intervals** ensure that the coverage probability is maximized at the true parameter value
        
        4. **UMA intervals** minimize the probability of including incorrect parameter values
        
        In practice, standard intervals for many common parameters (like the normal mean) satisfy multiple optimality criteria simultaneously. However, for more complex scenarios or non-standard distributions, these criteria may lead to different interval constructions.
        """)

    with tabs[4]:  # Interpretation
        st.subheader("Correct Interpretation of Confidence Intervals")
        
        st.markdown(r"""
        ### Common Misconceptions
        
        Confidence intervals are frequently misinterpreted, even by researchers. Here are some clarifications:
        
        | Incorrect Interpretation | Correct Interpretation |
        |--------------------------|------------------------|
        | There is a 95% probability that the true parameter lies within this specific interval | 95% of the intervals constructed using this method would contain the true parameter value |
        | 95% of the data falls within the confidence interval | The confidence interval describes the uncertainty in the parameter estimate, not the spread of the data |
        | If two confidence intervals overlap, the parameters are not significantly different | Overlapping confidence intervals do not necessarily imply non-significance (the correct test depends on the standard error of the difference) |
        | A wider confidence interval means the estimate is "more confident" | A wider interval actually indicates less precision (more uncertainty) in the estimate |
        
        ### Frequentist vs. Bayesian Perspective
        
        The fundamental difference in interpretation:
        
        - **Frequentist CI**: A range that would contain the true parameter in 95% of repeated experiments
        - **Bayesian Credible Interval**: A range that has a 95% probability of containing the true parameter, given the observed data and prior
        
        In the frequentist framework, the parameter is fixed (not random), and the interval is random. In the Bayesian framework, the parameter is treated as a random variable with a posterior distribution.
        """)
        
        # Visualization of the confidence interval concept
        st.subheader("Interactive Visualization of Confidence Interval Concept")
        
        n_intervals = st.slider("Number of simulated samples", 20, 100, 30)
        true_param = st.slider("True parameter value", 0.0, 10.0, 5.0, 0.1)
        sample_size = st.slider("Sample size for each interval", 5, 100, 20)
        conf_level = st.select_slider("Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95)
        
        # Generate visualization on button click
        if st.button("Generate Confidence Interval Visualization", key="gen_ci_concept"):
            np.random.seed(None)  # Use a different seed each time
            
            # Generate multiple samples and compute CIs
            intervals = []
            sample_means = []
            contains_param = []
            
            for i in range(n_intervals):
                # Generate sample from normal distribution
                sample = np.random.normal(true_param, 2, sample_size)
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                
                # Compute CI
                margin = stats.t.ppf((1 + conf_level) / 2, sample_size - 1) * sample_std / np.sqrt(sample_size)
                lower = sample_mean - margin
                upper = sample_mean + margin
                
                # Store results
                sample_means.append(sample_mean)
                intervals.append((lower, upper))
                contains_param.append(lower <= true_param <= upper)
            
            # Create visualization
            fig = go.Figure()
            
            # Add vertical line for true parameter
            fig.add_vline(x=true_param, line=dict(color='red', width=3, dash='dash'), 
                         annotation=dict(text=f"True Parameter = {true_param}", showarrow=False))
            
            # Add confidence intervals
            for i in range(n_intervals):
                lower, upper = intervals[i]
                color = 'rgba(0, 128, 0, 0.5)' if contains_param[i] else 'rgba(255, 0, 0, 0.5)'
                label = "Contains true value" if contains_param[i] else "Misses true value"
                showlegend = (i == 0) or (i == next((j for j, x in enumerate(contains_param) if x != contains_param[0]), None))
                
                fig.add_trace(
                    go.Scatter(x=[lower, upper], y=[i, i], mode='lines', 
                              line=dict(color=color, width=3),
                              name=label, showlegend=showlegend)
                )
                
                fig.add_trace(
                    go.Scatter(x=[sample_means[i]], y=[i], mode='markers',
                              marker=dict(color='blue', size=8),
                              name="Sample Mean", showlegend=(i == 0))
                )
            
            # Calculate actual coverage
            actual_coverage = sum(contains_param) / n_intervals * 100
            
            fig.update_layout(
                title=f'Multiple {conf_level*100:.0f}% Confidence Intervals from Different Samples<br>'
                      f'Actual coverage: {actual_coverage:.1f}% ({sum(contains_param)} out of {n_intervals})',
                xaxis_title='Parameter Value',
                yaxis_title='Sample Number',
                yaxis=dict(showticklabels=False),
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.markdown(f"""
            **Correct Interpretation**:
            
            This visualization shows {n_intervals} different {conf_level*100:.0f}% confidence intervals, each constructed from a different random sample of size {sample_size}.
            
            - Each **horizontal line** represents a confidence interval from one sample
            - **Green intervals** contain the true parameter value (red dashed line)
            - **Red intervals** miss the true parameter value
            - **Blue dots** represent the sample means
            
            In this simulation, {actual_coverage:.1f}% ({sum(contains_param)} out of {n_intervals}) intervals contain the true parameter value. With more simulations, this percentage would approach {conf_level*100:.0f}%.
            
            **Key point**: Any single confidence interval either contains the true parameter or it doesn't. The confidence level ({conf_level*100:.0f}%) refers to the frequency with which the interval construction method captures the true parameter over many repeated samples.
            """)
            
            # Add comparison with Bayesian interpretation
            st.subheader("Comparison with Bayesian Interpretation")
            
            st.markdown(r"""
            **Bayesian Alternative**:
            
            A Bayesian would compute a **credible interval** instead, which has the interpretation many people mistakenly apply to confidence intervals.
            
            For a 95% credible interval $[a, b]$:
            
            $$P(a \leq \theta \leq b | \text{data}) = 0.95$$
            
            That is, given the observed data, there is a 95% probability that the true parameter lies in the interval $[a, b]$.
            
            The difference arises because Bayesian inference treats the parameter as a random variable with a probability distribution, while frequentist inference treats the parameter as fixed but unknown.
            """)
            
            # Create a simple Bayesian vs. Frequentist example
            st.markdown("### Example: Normal Mean with Known Variance")
            
            # Simulate a sample
            np.random.seed(42)  # Fixed seed for reproducibility
            example_sample = np.random.normal(true_param, 2, sample_size)
            example_mean = np.mean(example_sample)
            
            # Frequentist CI
            freq_margin = 2 * (2 / np.sqrt(sample_size))  # Using known variance = 4
            freq_lower = example_mean - freq_margin
            freq_upper = example_mean + freq_margin
            
            # Bayesian credible interval (assuming improper flat prior)
            # With flat prior, the posterior is N(xbar, sigma^2/n)
            bayes_lower = stats.norm.ppf(0.025, example_mean, 2/np.sqrt(sample_size))
            bayes_upper = stats.norm.ppf(0.975, example_mean, 2/np.sqrt(sample_size))
            
            st.markdown(f"""
            For a single sample with mean x̄ = {example_mean:.4f}:
            
            - **95% Frequentist CI**: [{freq_lower:.4f}, {freq_upper:.4f}]
              - Interpretation: If we repeated the experiment many times, 95% of such intervals would contain μ
            
            - **95% Bayesian Credible Interval** (with flat prior): [{bayes_lower:.4f}, {bayes_upper:.4f}]
              - Interpretation: Given the observed data, there is a 95% probability that μ lies in this interval
            
            In this case with a flat prior, the intervals are numerically identical, but their interpretations differ fundamentally.
            """)
            
            # Add practical advice
            st.markdown("""
            ### Practical Advice for Interpretation
            
            When reporting confidence intervals in research:
            
            1. **Be precise in language**: "We are 95% confident that the true parameter lies within..." rather than "There is a 95% probability that..."
            
            2. **Emphasize estimation over testing**: Report and interpret the interval bounds, not just whether zero is included
            
            3. **Consider the practical significance**: Is the range of plausible values narrow enough to be useful?
            
            4. **Remember the assumptions**: Validity of confidence intervals depends on the assumptions of the underlying model
            
            5. **Consider presenting Bayesian intervals**: If the probability interpretation is desired, compute and report credible intervals instead of (or alongside) confidence intervals
            """)
        
# Continuation of the app - more modules planned:

# Interactive Simulations Module
elif nav == "Interactive Simulations":
    st.header("Interactive Simulations")
    
    sim_type = st.selectbox(
        "Select simulation type",
        ["Coverage Properties", "Sample Size Effects", "Bootstrapping", 
         "Transformations", "Non-normality Impact"]
    )
    
    if sim_type == "Coverage Properties":
        st.subheader("Coverage Properties of Confidence Intervals")
        
        st.markdown("""
        This simulation demonstrates the actual coverage probability of confidence intervals under various conditions.
        
        Coverage probability is the proportion of times that a confidence interval contains the true parameter value, which should match the nominal confidence level (e.g., 95%).
        """)
        
        interval_type = st.radio(
            "Select interval type",
            ["Normal Mean", "Binomial Proportion", "Variance"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
            n_sims = st.slider("Number of simulations", 100, 5000, 1000, 100)
        
        with col2:
            if interval_type == "Normal Mean":
                true_mean = st.slider("True population mean", -10.0, 10.0, 0.0, 0.5)
                true_sd = st.slider("True population standard deviation", 0.1, 10.0, 1.0, 0.1)
                sample_size = st.slider("Sample size", 5, 100, 30)
                use_t = st.checkbox("Use t-distribution (unknown variance)", value=True)
            
            elif interval_type == "Binomial Proportion":
                true_prop = st.slider("True population proportion", 0.01, 0.99, 0.5, 0.01)
                sample_size = st.slider("Sample size", 5, 500, 50)
                method = st.radio("Method", ["Wald", "Wilson", "Clopper-Pearson"])
            
            elif interval_type == "Variance":
                true_mean = st.slider("True population mean", -10.0, 10.0, 0.0, 0.5)
                true_sd = st.slider("True population standard deviation", 0.1, 10.0, 1.0, 0.1)
                sample_size = st.slider("Sample size", 5, 100, 30)
                parameter = st.radio("Parameter", ["Variance", "Standard Deviation"])
        
        if st.button("Run Simulation", key="run_coverage_sim"):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run the simulation
            contains_true = 0
            interval_widths = []
            
            for i in range(n_sims):
                # Update progress
                progress_bar.progress((i + 1) / n_sims)
                status_text.text(f"Running simulation {i+1}/{n_sims}")
                
                if interval_type == "Normal Mean":
                    # Generate sample
                    sample = np.random.normal(true_mean, true_sd, sample_size)
                    sample_mean = np.mean(sample)
                    
                    if use_t:
                        # t-interval
                        sample_sd = np.std(sample, ddof=1)
                        t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                        margin = t_crit * sample_sd / np.sqrt(sample_size)
                    else:
                        # z-interval (known variance)
                        z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                        margin = z_crit * true_sd / np.sqrt(sample_size)
                    
                    lower = sample_mean - margin
                    upper = sample_mean + margin
                    
                    # Check if interval contains true parameter
                    contains_true += (lower <= true_mean <= upper)
                    interval_widths.append(upper - lower)
                
                elif interval_type == "Binomial Proportion":
                    # Generate sample
                    sample = np.random.binomial(1, true_prop, sample_size)
                    sample_prop = np.mean(sample)
                    
                    if method == "Wald":
                        # Wald interval
                        if sample_prop == 0 or sample_prop == 1:
                            # Edge case handling
                            if sample_prop == 0:
                                lower = 0
                                upper = 3 / sample_size  # Rule of 3
                            else:  # sample_prop == 1
                                lower = 1 - 3 / sample_size
                                upper = 1
                        else:
                            z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                            margin = z_crit * np.sqrt(sample_prop * (1 - sample_prop) / sample_size)
                            lower = max(0, sample_prop - margin)
                            upper = min(1, sample_prop + margin)
                    
                    elif method == "Wilson":
                        # Wilson score interval
                        z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                        denominator = 1 + z_crit**2/sample_size
                        center = (sample_prop + z_crit**2/(2*sample_size)) / denominator
                        margin = z_crit * np.sqrt((sample_prop*(1-sample_prop) + z_crit**2/(4*sample_size)) / sample_size) / denominator
                        lower = max(0, center - margin)
                        upper = min(1, center + margin)
                    
                    else:  # Clopper-Pearson
                        # Clopper-Pearson exact interval
                        k = np.sum(sample)
                        lower = stats.beta.ppf((1-conf_level)/2, k, sample_size-k+1) if k > 0 else 0
                        upper = stats.beta.ppf(1-(1-conf_level)/2, k+1, sample_size-k) if k < sample_size else 1
                    
                    # Check if interval contains true parameter
                    contains_true += (lower <= true_prop <= upper)
                    interval_widths.append(upper - lower)
                
                elif interval_type == "Variance":
                    # Generate sample
                    sample = np.random.normal(true_mean, true_sd, sample_size)
                    sample_var = np.var(sample, ddof=1)
                    
                    # Chi-square interval for variance
# Chi-square interval for variance
                    chi2_upper = stats.chi2.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                    chi2_lower = stats.chi2.ppf((1 - conf_level)/2, sample_size - 1)
                    
                    var_lower = (sample_size - 1) * sample_var / chi2_upper
                    var_upper = (sample_size - 1) * sample_var / chi2_lower
                    
                    if parameter == "Variance":
                        # Check if interval contains true variance
                        contains_true += (var_lower <= true_sd**2 <= var_upper)
                        interval_widths.append(var_upper - var_lower)
                    else:  # Standard Deviation
                        # Square root transformation for SD interval
                        sd_lower = np.sqrt(var_lower)
                        sd_upper = np.sqrt(var_upper)
                        contains_true += (sd_lower <= true_sd <= sd_upper)
                        interval_widths.append(sd_upper - sd_lower)
            
            # Clear progress indicators
            status_text.empty()
            progress_bar.empty()
            
            # Calculate actual coverage
            actual_coverage = contains_true / n_sims * 100
            avg_width = np.mean(interval_widths)
            
            # Display results
            st.success(f"Simulation complete! {n_sims} intervals generated.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Actual Coverage", 
                    value=f"{actual_coverage:.1f}%",
                    delta=f"{actual_coverage - conf_level*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Average Interval Width", 
                    value=f"{avg_width:.4f}"
                )
            
            # Create coverage visualization
            fig = go.Figure()
            
            # Add bar for actual coverage
            fig.add_trace(go.Bar(
                x=['Actual Coverage'],
                y=[actual_coverage],
                name='Actual Coverage',
                marker_color='blue'
            ))
            
            # Add bar for nominal coverage
            fig.add_trace(go.Bar(
                x=['Nominal Coverage'],
                y=[conf_level * 100],
                name='Nominal Coverage',
                marker_color='green'
            ))
            
            fig.update_layout(
                title='Actual vs. Nominal Coverage',
                yaxis=dict(title='Coverage (%)', range=[min(actual_coverage, conf_level*100)*0.95, 
                                                       max(actual_coverage, conf_level*100)*1.05]),
                height=400,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            if abs(actual_coverage - conf_level*100) < 2:
                coverage_quality = "excellent"
            elif abs(actual_coverage - conf_level*100) < 5:
                coverage_quality = "good"
            else:
                coverage_quality = "poor"
            
            st.markdown(f"""
            ### Interpretation
            
            The simulation results show that the {interval_type} confidence interval has {coverage_quality} coverage properties under the specified conditions:
            
            - **Nominal (target) coverage**: {conf_level*100:.1f}%
            - **Actual coverage**: {actual_coverage:.1f}%
            - **Difference**: {actual_coverage - conf_level*100:.1f} percentage points
            
            This means that out of {n_sims} different confidence intervals constructed from different random samples, {contains_true} intervals ({actual_coverage:.1f}%) contained the true parameter value.
            
            The average width of the confidence intervals was {avg_width:.4f}.
            """)
            
            # Add specific interpretations based on interval type
            if interval_type == "Normal Mean":
                if use_t:
                    method_text = "t-distribution"
                    assumptions_text = "normally distributed population and unknown variance"
                else:
                    method_text = "normal distribution"
                    assumptions_text = "normally distributed population and known variance"
                
                st.markdown(f"""
                **Additional insights for Normal Mean interval**:
                
                The confidence interval based on the {method_text} generally has good coverage properties when the underlying assumptions ({assumptions_text}) are met. 
                
                As the sample size increases, the t-interval approaches the z-interval, and both provide very accurate coverage for normal data.
                """)
                
            elif interval_type == "Binomial Proportion":
                if method == "Wald":
                    method_evaluation = "The Wald interval is known to have poor coverage for small sample sizes or extreme proportions (near 0 or 1). It tends to have actual coverage below the nominal level."
                elif method == "Wilson":
                    method_evaluation = "The Wilson score interval generally has better coverage properties than the Wald interval, especially for small samples and extreme proportions."
                else:  # Clopper-Pearson
                    method_evaluation = "The Clopper-Pearson interval is guaranteed to have coverage at least equal to the nominal level, but is often conservative (wider than necessary)."
                
                st.markdown(f"""
                **Additional insights for Binomial Proportion interval**:
                
                {method_evaluation}
                
                For your simulation with p = {true_prop} and n = {sample_size}, the {method} method showed {coverage_quality} performance.
                """)
                
            elif interval_type == "Variance":
                st.markdown(f"""
                **Additional insights for {'Variance' if parameter == 'Variance' else 'Standard Deviation'} interval**:
                
                The chi-square-based interval for {'variance' if parameter == 'Variance' else 'standard deviation'} is highly sensitive to departures from normality.
                
                For normal data with sample size {sample_size}, the coverage is {coverage_quality}.
                
                The transformation from variance to standard deviation can affect coverage properties, particularly for small samples.
                """)
    
    elif sim_type == "Sample Size Effects":
        st.subheader("Effects of Sample Size on Confidence Intervals")
        
        st.markdown("""
        This simulation demonstrates how sample size affects the width and coverage of confidence intervals.
        
        As sample size increases:
        1. Interval width should decrease (greater precision)
        2. Coverage should approach the nominal confidence level
        """)
        
        interval_type = st.radio(
            "Select interval type",
            ["Normal Mean", "Binomial Proportion"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
            n_sims = st.slider("Simulations per sample size", 100, 2000, 500, 100)
        
        with col2:
            if interval_type == "Normal Mean":
                true_mean = st.slider("True population mean", -10.0, 10.0, 0.0, 0.5)
                true_sd = st.slider("True population standard deviation", 0.1, 10.0, 1.0, 0.1)
                min_n = st.slider("Minimum sample size", 5, 50, 5)
                max_n = st.slider("Maximum sample size", 51, 500, 100)
            
            elif interval_type == "Binomial Proportion":
                true_prop = st.slider("True population proportion", 0.01, 0.99, 0.5, 0.01)
                min_n = st.slider("Minimum sample size", 5, 50, 10)
                max_n = st.slider("Maximum sample size", 51, 500, 200)
                method = st.radio("Method", ["Wald", "Wilson", "Clopper-Pearson"])
        
        if st.button("Run Simulation", key="run_sample_size_sim"):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define sample sizes to test
            if max_n <= 100:
                sample_sizes = np.arange(min_n, max_n + 1, 5)
            else:
                step = max(5, (max_n - min_n) // 20)
                sample_sizes = np.arange(min_n, max_n + 1, step)
            
            # Initialize results storage
            coverages = []
            avg_widths = []
            
            # Run simulations for each sample size
            total_iterations = len(sample_sizes)
            
            for i, n in enumerate(sample_sizes):
                progress_bar.progress((i + 0.5) / total_iterations)
                status_text.text(f"Simulating with sample size {n} ({i+1}/{total_iterations})")
                
                # Run simulations for this sample size
                contains_true = 0
                interval_widths = []
                
                for j in range(n_sims):
                    if interval_type == "Normal Mean":
                        # Generate sample
                        sample = np.random.normal(true_mean, true_sd, n)
                        sample_mean = np.mean(sample)
                        sample_sd = np.std(sample, ddof=1)
                        
                        # t-interval
                        t_crit = stats.t.ppf(1 - (1 - conf_level)/2, n - 1)
                        margin = t_crit * sample_sd / np.sqrt(n)
                        lower = sample_mean - margin
                        upper = sample_mean + margin
                        
                        # Check if interval contains true mean
                        contains_true += (lower <= true_mean <= upper)
                        interval_widths.append(upper - lower)
                    
                    elif interval_type == "Binomial Proportion":
                        # Generate sample
                        sample = np.random.binomial(1, true_prop, n)
                        sample_prop = np.mean(sample)
                        
                        if method == "Wald":
                            # Wald interval
                            if sample_prop == 0 or sample_prop == 1:
                                # Edge case handling
                                if sample_prop == 0:
                                    lower = 0
                                    upper = 3 / n  # Rule of 3
                                else:  # sample_prop == 1
                                    lower = 1 - 3 / n
                                    upper = 1
                            else:
                                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                                margin = z_crit * np.sqrt(sample_prop * (1 - sample_prop) / n)
                                lower = max(0, sample_prop - margin)
                                upper = min(1, sample_prop + margin)
                        
                        elif method == "Wilson":
                            # Wilson score interval
                            z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                            denominator = 1 + z_crit**2/n
                            center = (sample_prop + z_crit**2/(2*n)) / denominator
                            margin = z_crit * np.sqrt((sample_prop*(1-sample_prop) + z_crit**2/(4*n)) / n) / denominator
                            lower = max(0, center - margin)
                            upper = min(1, center + margin)
                        
                        else:  # Clopper-Pearson
                            # Clopper-Pearson exact interval
                            k = np.sum(sample)
                            lower = stats.beta.ppf((1-conf_level)/2, k, n-k+1) if k > 0 else 0
                            upper = stats.beta.ppf(1-(1-conf_level)/2, k+1, n-k) if k < n else 1
                        
                        # Check if interval contains true parameter
                        contains_true += (lower <= true_prop <= upper)
                        interval_widths.append(upper - lower)
                
                # Store results for this sample size
                coverage = contains_true / n_sims * 100
                avg_width = np.mean(interval_widths)
                
                coverages.append(coverage)
                avg_widths.append(avg_width)
                
                progress_bar.progress((i + 1) / total_iterations)
            
            # Clear progress indicators
            status_text.empty()
            progress_bar.empty()
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Sample Size': sample_sizes,
                'Coverage (%)': coverages,
                'Average Width': avg_widths
            })
            
            # Display results table
            st.subheader("Simulation Results")
            st.dataframe(results_df.style.format({
                'Coverage (%)': '{:.1f}',
                'Average Width': '{:.4f}'
            }))
            
            # Create visualization of coverage vs sample size
            fig1 = go.Figure()
            
            # Add coverage line
            fig1.add_trace(go.Scatter(
                x=sample_sizes,
                y=coverages,
                mode='lines+markers',
                name='Actual Coverage',
                line=dict(color='blue', width=2)
            ))
            
            # Add nominal coverage line
            fig1.add_hline(y=conf_level*100, line=dict(color='red', width=2, dash='dash'),
                          annotation=dict(text=f"Nominal {conf_level*100:.1f}%", showarrow=False))
            
            fig1.update_layout(
                title='Coverage Probability vs. Sample Size',
                xaxis_title='Sample Size',
                yaxis_title='Coverage (%)',
                height=400,
                yaxis=dict(
                    range=[min(min(coverages), conf_level*100)*0.95, 
                           max(max(coverages), conf_level*100)*1.05]
                )
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create visualization of width vs sample size
            fig2 = go.Figure()
            
            # Add width line
            fig2.add_trace(go.Scatter(
                x=sample_sizes,
                y=avg_widths,
                mode='lines+markers',
                name='Average Width',
                line=dict(color='green', width=2)
            ))
            
            # Add theoretical width line
            if interval_type == "Normal Mean":
                # Theoretical width for normal mean: 2 * t_crit * sigma / sqrt(n)
                theoretical_widths = [2 * stats.t.ppf(1 - (1 - conf_level)/2, n - 1) * true_sd / np.sqrt(n) for n in sample_sizes]
                
                fig2.add_trace(go.Scatter(
                    x=sample_sizes,
                    y=theoretical_widths,
                    mode='lines',
                    name='Theoretical Width',
                    line=dict(color='purple', width=2, dash='dash')
                ))
            
            elif interval_type == "Binomial Proportion" and method == "Wald":
                # Theoretical width for binomial proportion (Wald): 2 * z_crit * sqrt(p(1-p)/n)
                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                theoretical_widths = [2 * z_crit * np.sqrt(true_prop * (1 - true_prop) / n) for n in sample_sizes]
                
                fig2.add_trace(go.Scatter(
                    x=sample_sizes,
                    y=theoretical_widths,
                    mode='lines',
                    name='Theoretical Width (Wald)',
                    line=dict(color='purple', width=2, dash='dash')
                ))
            
            fig2.update_layout(
                title='Average Interval Width vs. Sample Size',
                xaxis_title='Sample Size',
                yaxis_title='Average Width',
                height=400
            )
            
            # Add reference lines showing the 1/sqrt(n) relationship
            reference_n = sample_sizes[0]
            reference_width = avg_widths[0]
            
            reference_widths = [reference_width * np.sqrt(reference_n / n) for n in sample_sizes]
            
            fig2.add_trace(go.Scatter(
                x=sample_sizes,
                y=reference_widths,
                mode='lines',
                name='1/√n Relationship',
                line=dict(color='orange', width=2, dash='dot')
            ))
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Add logarithmic plot to better show the relationship
            fig3 = go.Figure()
            
            # Add width line on log-log scale
            fig3.add_trace(go.Scatter(
                x=np.log(sample_sizes),
                y=np.log(avg_widths),
                mode='lines+markers',
                name='Log(Average Width)',
                line=dict(color='green', width=2)
            ))
            
            # Fit a line to confirm the -1/2 slope
            slope, intercept = np.polyfit(np.log(sample_sizes), np.log(avg_widths), 1)
            
            fig3.add_trace(go.Scatter(
                x=np.log(sample_sizes),
                y=intercept + slope * np.log(sample_sizes),
                mode='lines',
                name=f'Fitted Line (slope = {slope:.3f})',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig3.update_layout(
                title='Log-Log Plot: Width vs. Sample Size',
                xaxis_title='Log(Sample Size)',
                yaxis_title='Log(Average Width)',
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Add interpretation
            st.subheader("Interpretation")
            
            st.markdown(f"""
            ### Effects of Sample Size on Confidence Intervals
            
            **1. Coverage Probability:**
            
            The simulation shows that as sample size increases, the actual coverage probability {'approaches' if abs(coverages[-1] - conf_level*100) < abs(coverages[0] - conf_level*100) else 'deviates from'} the nominal level of {conf_level*100:.1f}%.
            
            - For small samples (n = {min_n}), the actual coverage was {coverages[0]:.1f}%
            - For large samples (n = {sample_sizes[-1]}), the actual coverage was {coverages[-1]:.1f}%
            
            **2. Interval Width:**
            
            The width of the confidence interval decreases as sample size increases:
            
            - For small samples (n = {min_n}), the average width was {avg_widths[0]:.4f}
            - For large samples (n = {sample_sizes[-1]}), the average width was {avg_widths[-1]:.4f}
            
            **3. Rate of Decrease:**
            
            The log-log plot confirms that confidence interval width decreases approximately at a rate proportional to 1/√n:
            
            - The fitted slope is {slope:.3f}, which is {'close to' if abs(slope + 0.5) < 0.1 else 'different from'} the theoretical value of -0.5
            
            This means that to halve the width of a confidence interval, you need to quadruple the sample size.
            """)
            
            if interval_type == "Normal Mean":
                st.markdown("""
                **Additional insights for Normal Mean interval:**
                
                The t-based confidence interval for the mean has excellent coverage properties across all sample sizes when the population is normally distributed.
                
                For very small samples, the t-distribution has heavier tails than the normal distribution, which compensates for the increased uncertainty in estimating the standard deviation from the sample.
                """)
            
            elif interval_type == "Binomial Proportion":
                if method == "Wald":
                    method_specific = "The Wald interval tends to have poor coverage for small samples and extreme proportions (near 0 or 1), which can be seen in the coverage plot."
                elif method == "Wilson":
                    method_specific = "The Wilson score interval maintains good coverage across different sample sizes, even for smaller samples."
                else:  # Clopper-Pearson
                    method_specific = "The Clopper-Pearson interval is conservative, ensuring that the actual coverage is at least the nominal level across all sample sizes."
                
                st.markdown(f"""
                **Additional insights for Binomial Proportion interval:**
                
                {method_specific}
                
                For proportions near 0.5 (maximum variance), the width of the confidence interval is larger compared to proportions near 0 or 1. However, this is where the Wald interval performs best in terms of coverage.
                """)
    
    elif sim_type == "Bootstrapping":
        st.subheader("Bootstrap Confidence Intervals")
        
        st.markdown("""
        This simulation demonstrates how bootstrap methods can be used to construct confidence intervals without making parametric assumptions.
        
        Bootstrap methods involve resampling with replacement from the observed data to estimate the sampling distribution of a statistic. This approach is particularly useful when:
        
        1. The underlying distribution is unknown or non-normal
        2. The parameter of interest has a complex functional form
        3. Analytical expressions for the standard error are not available
        """)
        
        # Select parameter and bootstrap method
        col1, col2 = st.columns(2)
        
        with col1:
            parameter = st.radio(
                "Parameter of interest",
                ["Mean", "Median", "Trimmed Mean (10%)", "Standard Deviation", "Correlation"]
            )
            
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
        
        with col2:
            bootstrap_method = st.radio(
                "Bootstrap method",
                ["Percentile", "Basic", "BCa (Bias-Corrected and Accelerated)"]
            )
            
            n_bootstrap = st.slider("Number of bootstrap samples", 1000, 10000, 2000, 1000)
        
        # Data generation options
        st.subheader("Data Generation")
        
        dist_type = st.radio(
            "Distribution type",
            ["Normal", "Skewed (Log-normal)", "Heavy-tailed (t)", "Mixture", "User-defined"]
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample size", 10, 200, 50)
        
        with col2:
            if dist_type == "Normal":
                mean = st.slider("Mean", -10.0, 10.0, 0.0, 0.5)
                std_dev = st.slider("Standard deviation", 0.1, 10.0, 1.0, 0.1)
            elif dist_type == "Skewed (Log-normal)":
                location = st.slider("Location (μ)", -1.0, 2.0, 0.0, 0.1)
                scale = st.slider("Scale (σ)", 0.1, 2.0, 0.5, 0.1)
            elif dist_type == "Heavy-tailed (t)":
                df = st.slider("Degrees of freedom", 1, 30, 3)
            elif dist_type == "Mixture":
                mix_prob = st.slider("Mixture probability", 0.0, 1.0, 0.7, 0.05)
        
        with col3:
            if dist_type == "User-defined":
                user_data = st.text_area("Enter comma-separated data values", 
                                         "1.2, 3.4, 2.5, 4.6, 5.7, 6.8, 7.9, 8.0, 9.1, 10.2")
        
        # Run simulation
        if st.button("Generate Bootstrap Confidence Intervals", key="run_bootstrap"):
            # Generate or process data
            if dist_type == "Normal":
                data = np.random.normal(mean, std_dev, sample_size)
                true_value = mean if parameter == "Mean" else np.median(stats.norm.ppf(np.linspace(0.001, 0.999, 10000), mean, std_dev)) if parameter == "Median" else std_dev if parameter == "Standard Deviation" else None
            
            elif dist_type == "Skewed (Log-normal)":
                data = np.random.lognormal(location, scale, sample_size)
                # True values for lognormal
                if parameter == "Mean":
                    true_value = np.exp(location + scale**2/2)
                elif parameter == "Median":
                    true_value = np.exp(location)
                elif parameter == "Standard Deviation":
                    true_value = np.sqrt((np.exp(scale**2) - 1) * np.exp(2*location + scale**2))
                else:
                    true_value = None
            
            elif dist_type == "Heavy-tailed (t)":
                data = stats.t.rvs(df, size=sample_size)
                # True values for t-distribution
                if parameter == "Mean" and df > 1:
                    true_value = 0
                elif parameter == "Median":
                    true_value = 0
                elif parameter == "Standard Deviation" and df > 2:
                    true_value = np.sqrt(df / (df - 2))
                else:
                    true_value = None
            
            elif dist_type == "Mixture":
                component1 = np.random.normal(0, 1, sample_size)
                component2 = np.random.normal(5, 2, sample_size)
                mask = np.random.random(sample_size) < mix_prob
                data = component1 * mask + component2 * (1 - mask)
                # True values for mixture are complex, set to None
                true_value = mix_prob * 0 + (1 - mix_prob) * 5 if parameter == "Mean" else None
            
            elif dist_type == "User-defined":
                try:
                    data = np.array([float(x.strip()) for x in user_data.split(',')])
                    true_value = None
                except:
                    st.error("Invalid data format. Please enter comma-separated numeric values.")
                  
            
            # Calculate the observed statistic
            if parameter == "Mean":
                observed_stat = np.mean(data)
            elif parameter == "Median":
                observed_stat = np.median(data)
            elif parameter == "Trimmed Mean (10%)":
                observed_stat = stats.trim_mean(data, 0.1)
            elif parameter == "Standard Deviation":
                observed_stat = np.std(data, ddof=1)
            elif parameter == "Correlation":
                if len(data) < 2:
                    st.error("Need at least 2 data points for correlation.")
               
                # Generate correlated data for demonstration
                noise = np.random.normal(0, 0.5, sample_size)
                data2 = data * 0.7 + noise
                observed_stat = np.corrcoef(data, data2)[0, 1]
                # Create paired data for bootstrap
                data = np.column_stack((data, data2))
            
            # Generate bootstrap samples
            bootstrap_stats = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(n_bootstrap):
                # Update progress every 10%
                if i % (n_bootstrap // 10) == 0:
                    progress = i / n_bootstrap
                    progress_bar.progress(progress)
                    status_text.text(f"Generating bootstrap sample {i}/{n_bootstrap}")
                
                # Generate bootstrap sample
                bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
                
                if parameter == "Correlation":
                    bootstrap_sample = data[bootstrap_indices]
                    bootstrap_stat = np.corrcoef(bootstrap_sample[:, 0], bootstrap_sample[:, 1])[0, 1]
                else:
                    bootstrap_sample = data[bootstrap_indices]
                    
                    if parameter == "Mean":
                        bootstrap_stat = np.mean(bootstrap_sample)
                    elif parameter == "Median":
                        bootstrap_stat = np.median(bootstrap_sample)
                    elif parameter == "Trimmed Mean (10%)":
                        bootstrap_stat = stats.trim_mean(bootstrap_sample, 0.1)
                    elif parameter == "Standard Deviation":
                        bootstrap_stat = np.std(bootstrap_sample, ddof=1)
                
                bootstrap_stats.append(bootstrap_stat)
            
            # Clear progress indicators
            status_text.empty()
            progress_bar.empty()
            
            # Calculate confidence intervals
            alpha = 1 - conf_level
            bootstrap_stats = np.array(bootstrap_stats)
            
            if bootstrap_method == "Percentile":
                lower = np.percentile(bootstrap_stats, 100 * alpha/2)
                upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
                method_name = "Percentile Bootstrap"
                
            elif bootstrap_method == "Basic":
                # Basic bootstrap (reflection method)
                lower = 2 * observed_stat - np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
                upper = 2 * observed_stat - np.percentile(bootstrap_stats, 100 * alpha/2)
                method_name = "Basic Bootstrap"
                
            elif bootstrap_method == "BCa (Bias-Corrected and Accelerated)":
                # BCa bootstrap (more accurate but more complex)
                # Calculate bias-correction factor
                z0 = stats.norm.ppf(np.mean(bootstrap_stats < observed_stat))
                
                # Calculate acceleration factor (jackknife estimation)
                jackknife_stats = []
                
                for i in range(len(data)):
                    # Leave one out
                    if parameter == "Correlation":
                        jackknife_sample = np.delete(data, i, axis=0)
                        jackknife_stat = np.corrcoef(jackknife_sample[:, 0], jackknife_sample[:, 1])[0, 1]
                    else:
                        jackknife_sample = np.delete(data, i)
                        
                        if parameter == "Mean":
                            jackknife_stat = np.mean(jackknife_sample)
                        elif parameter == "Median":
                            jackknife_stat = np.median(jackknife_sample)
                        elif parameter == "Trimmed Mean (10%)":
                            jackknife_stat = stats.trim_mean(jackknife_sample, 0.1)
                        elif parameter == "Standard Deviation":
                            jackknife_stat = np.std(jackknife_sample, ddof=1)
                    
                    jackknife_stats.append(jackknife_stat)
                
                jackknife_stats = np.array(jackknife_stats)
                jackknife_mean = np.mean(jackknife_stats)
                num = np.sum((jackknife_mean - jackknife_stats)**3)
                denom = 6 * np.sum((jackknife_mean - jackknife_stats)**2)**1.5
                
                # Avoid division by zero
                a = num / denom if denom != 0 else 0
                
                # Calculate adjusted percentiles
                z_alpha1 = z0 + stats.norm.ppf(alpha/2)
                z_alpha2 = z0 + stats.norm.ppf(1 - alpha/2)
                
                # Adjust for acceleration
                p_alpha1 = stats.norm.cdf(z0 + (z_alpha1) / (1 - a * (z_alpha1)))
                p_alpha2 = stats.norm.cdf(z0 + (z_alpha2) / (1 - a * (z_alpha2)))
                
                # Get BCa interval
                lower = np.percentile(bootstrap_stats, 100 * p_alpha1)
                upper = np.percentile(bootstrap_stats, 100 * p_alpha2)
                method_name = "BCa Bootstrap"
            
            # Display results
            st.success(f"Bootstrap complete! {n_bootstrap} resamples generated.")
            
            # Create result cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=f"Observed {parameter}",
                    value=f"{observed_stat:.4f}"
                )
                
                if true_value is not None:
                    st.metric(
                        label=f"True {parameter} (if known)",
                        value=f"{true_value:.4f}"
                    )
            
            with col2:
                st.metric(
                    label=f"{conf_level*100:.0f}% CI Lower Bound",
                    value=f"{lower:.4f}"
                )
                
                st.metric(
                    label=f"{conf_level*100:.0f}% CI Upper Bound",
                    value=f"{upper:.4f}"
                )
            
            # Add interval width and coverage (if true value is known)
            interval_width = upper - lower
            
            st.metric(
                label="Interval Width",
                value=f"{interval_width:.4f}"
            )
            
            if true_value is not None:
                contains_true = lower <= true_value <= upper
                
                st.metric(
                    label="Contains True Value",
                    value="Yes" if contains_true else "No"
                )
            
            # Create histogram of bootstrap distribution
            fig = go.Figure()
            
            # Add histogram of bootstrap statistics
            fig.add_trace(go.Histogram(
                x=bootstrap_stats,
                nbinsx=30,
                opacity=0.7,
                name="Bootstrap Distribution"
            ))
            
            # Add vertical lines for CI bounds
            fig.add_vline(x=lower, line=dict(color='red', width=2, dash='dash'),
                         annotation=dict(text=f"Lower: {lower:.4f}", showarrow=False))
            
            fig.add_vline(x=upper, line=dict(color='red', width=2, dash='dash'),
                         annotation=dict(text=f"Upper: {upper:.4f}", showarrow=False))
            
            # Add vertical line for observed statistic
            fig.add_vline(x=observed_stat, line=dict(color='blue', width=2),
                         annotation=dict(text=f"Observed: {observed_stat:.4f}", showarrow=False))
            
            # Add vertical line for true value if known
            if true_value is not None:
                fig.add_vline(x=true_value, line=dict(color='green', width=2, dash='dot'),
                             annotation=dict(text=f"True: {true_value:.4f}", showarrow=False))
            
            fig.update_layout(
                title=f'Bootstrap Distribution of {parameter} ({n_bootstrap} resamples)',
                xaxis_title=parameter,
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.subheader("Interpretation")
            
            bootstrap_mean = np.mean(bootstrap_stats)
            bootstrap_sd = np.std(bootstrap_stats, ddof=1)
            
            st.markdown(f"""
            ### {method_name} Confidence Interval
            
            The {conf_level*100:.0f}% bootstrap confidence interval for the {parameter} is [{lower:.4f}, {upper:.4f}].
            
            **Bootstrap Distribution Statistics:**
            - Mean of bootstrap statistics: {bootstrap_mean:.4f}
            - Standard deviation of bootstrap statistics: {bootstrap_sd:.4f}
            - This standard deviation is an estimate of the standard error of the {parameter}
            
            **Interpretation of the Confidence Interval:**
            This interval suggests that, with {conf_level*100:.0f}% confidence, the true population {parameter.lower()} is between {lower:.4f} and {upper:.4f}.
            """)
            
            # Add method-specific explanation
            if bootstrap_method == "Percentile":
                st.markdown("""
                **Percentile Method Details:**
                
                The percentile method simply takes the empirical percentiles of the bootstrap distribution. It assumes that the bootstrap distribution is centered around the true parameter value and symmetric.
                
                **Advantages:**
                - Simple to understand and implement
                - Works well for symmetric distributions
                
                **Limitations:**
                - Can be biased for skewed distributions
                - Doesn't account for the relationship between the sample statistic and the bootstrap distribution
                """)
                
            elif bootstrap_method == "Basic":
                st.markdown("""
                **Basic Bootstrap Method Details:**
                
                The basic bootstrap method (also called the reflection method) reflects the bootstrap distribution around the observed statistic.
                
                **Advantages:**
                - Accounts for bias in the bootstrap distribution
                - Often more accurate than the percentile method
                
                **Limitations:**
                - Can produce intervals outside the parameter space (e.g., negative values for a standard deviation)
                - Doesn't adjust for skewness
                """)
                
            elif bootstrap_method == "BCa (Bias-Corrected and Accelerated)":
                st.markdown(f"""
                **BCa Method Details:**
                
                The BCa method adjusts for both bias and skewness in the bootstrap distribution.
                
                **Key Parameters:**
                - Bias-correction factor (z₀): {z0:.4f}
                - Acceleration factor (a): {a:.4f}
                
                **Advantages:**
                - Most accurate bootstrap method in most situations
                - Adjusts for both bias and skewness
                - Second-order accurate
                
                **Limitations:**
                - Computationally intensive (requires jackknife resampling)
                - More complex to implement and understand
                """)
            
            # Add distribution-specific notes
            if dist_type == "Normal":
                st.markdown("""
                **Notes on Normal Distribution:**
                
                For normally distributed data, parametric confidence intervals are typically optimal. However, bootstrap methods should provide similar results, especially for larger sample sizes.
                
                Bootstrap methods are particularly valuable when we're unsure about the underlying distribution or when dealing with statistics that don't have simple parametric confidence intervals (like the median or trimmed mean).
                """)
                
            elif dist_type == "Skewed (Log-normal)":
                st.markdown("""
                **Notes on Skewed Distribution:**
                
                For skewed distributions like the log-normal, bootstrap methods often outperform parametric methods, especially for statistics like the mean where the sampling distribution may not be normal.
                
                The BCa method is particularly valuable here because it adjusts for the skewness in the bootstrap distribution.
                """)
                
            elif dist_type == "Heavy-tailed (t)":
                st.markdown("""
                **Notes on Heavy-tailed Distribution:**
                
                With heavy-tailed distributions, sample means have high variance, and traditional methods may perform poorly with small samples. Bootstrap methods can provide more robust confidence intervals.
                
                For t-distributions with low degrees of freedom, statistics like the median may be more reliable than the mean.
                """)
    
    elif sim_type == "Transformations":
        st.subheader("Confidence Intervals with Transformations")
        
        st.markdown("""
        This simulation demonstrates how transformations can be used to construct more accurate confidence intervals, particularly for skewed distributions or parameters with restricted ranges.
        
        Common transformations include:
        1. Logarithmic transformation for right-skewed data
        2. Logit transformation for proportions
        3. Fisher's z-transformation for correlation coefficients
        """)
        
        transform_type = st.radio(
            "Select transformation type",
            ["Log Transformation for Ratio", "Logit Transformation for Proportion", 
             "Fisher's Z Transformation for Correlation"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
            sample_size = st.slider("Sample size", 10, 200, 30)
        
        with col2:
            if transform_type == "Log Transformation for Ratio":
                true_ratio = st.slider("True ratio", 0.1, 10.0, 2.0, 0.1)
                cv = st.slider("Coefficient of variation", 0.1, 1.0, 0.3, 0.05)
            
            elif transform_type == "Logit Transformation for Proportion":
                true_prop = st.slider("True proportion", 0.01, 0.99, 0.2, 0.01)
            
            elif transform_type == "Fisher's Z Transformation for Correlation":
                true_corr = st.slider("True correlation", -0.99, 0.99, 0.7, 0.01)
        
        if st.button("Run Simulation", key="run_transform_sim"):
            if transform_type == "Log Transformation for Ratio":
                # Log transformation for ratio
                # For log-normal, the mean on the log scale is log(μ²/√(μ²+σ²))
                # and the SD on the log scale is √(log(1 + (σ/μ)²))
                # where μ is the mean and σ is the SD on the original scale
                
                # Set up parameters for log-normal
                sigma = true_ratio * cv
                mu_log = np.log(true_ratio**2 / np.sqrt(true_ratio**2 + sigma**2))
                sigma_log = np.sqrt(np.log(1 + (sigma/true_ratio)**2))
                
                # Generate sample
                np.random.seed(None)
                log_sample = np.random.normal(mu_log, sigma_log, sample_size)
                sample = np.exp(log_sample)
                
                # Calculate statistics
                sample_ratio = np.mean(sample)
                sample_sd = np.std(sample, ddof=1)
                sample_cv = sample_sd / sample_ratio
                
                # Method 1: Direct CI on original scale (may be skewed)
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                margin_direct = t_crit * sample_sd / np.sqrt(sample_size)
                ci_lower_direct = sample_ratio - margin_direct
                ci_upper_direct = sample_ratio + margin_direct
                
                # Method 2: Log transformation
                log_mean = np.mean(log_sample)
                log_sd = np.std(log_sample, ddof=1)
                margin_log = t_crit * log_sd / np.sqrt(sample_size)
                ci_lower_log = np.exp(log_mean - margin_log)
                ci_upper_log = np.exp(log_mean + margin_log)
                
                # Display results
                st.subheader("Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Sample Ratio",
                        value=f"{sample_ratio:.4f}"
                    )
                    
                    st.metric(
                        label="True Ratio",
                        value=f"{true_ratio:.4f}"
                    )
                
                with col2:
                    st.metric(
                        label="Sample CV",
                        value=f"{sample_cv:.4f}"
                    )
                    
                    st.metric(
                        label="Sample SD",
                        value=f"{sample_sd:.4f}"
                    )
                
                with col3:
                    st.metric(
                        label="Sample Size",
                        value=f"{sample_size}"
                    )
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    'Method': ['Direct', 'Log Transformation'],
                    'Lower Bound': [ci_lower_direct, ci_lower_log],
                    'Upper Bound': [ci_upper_direct, ci_upper_log],
                    'Width': [ci_upper_direct - ci_lower_direct, ci_upper_log - ci_lower_log],
                    'Contains True Value': [ci_lower_direct <= true_ratio <= ci_upper_direct,
                                          ci_lower_log <= true_ratio <= ci_upper_log]
                })
                
                st.subheader(f"{conf_level*100:.0f}% Confidence Intervals")
                st.dataframe(comparison_df.style.format({
                    'Lower Bound': '{:.4f}',
                    'Upper Bound': '{:.4f}',
                    'Width': '{:.4f}',
                }))
                
                # Visualize the sample and CIs
                fig = go.Figure()
                
                # Add histogram of sample
                fig.add_trace(go.Histogram(
                    x=sample,
                    nbinsx=20,
                    opacity=0.7,
                    name="Sample Distribution"
                ))
                
                # Add vertical lines
                fig.add_vline(x=sample_ratio, line=dict(color='blue', width=2),
                             annotation=dict(text=f"Sample Mean: {sample_ratio:.4f}", showarrow=False))
                
                fig.add_vline(x=true_ratio, line=dict(color='green', width=2, dash='dot'),
                             annotation=dict(text=f"True Ratio: {true_ratio:.4f}", showarrow=False))
                
                # Direct interval
                fig.add_vline(x=ci_lower_direct, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"Direct Lower: {ci_lower_direct:.4f}", showarrow=False))
                
                fig.add_vline(x=ci_upper_direct, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"Direct Upper: {ci_upper_direct:.4f}", showarrow=False))
                
                # Log-transformed interval
                fig.add_vline(x=ci_lower_log, line=dict(color='purple', width=2, dash='dash'),
                             annotation=dict(text=f"Log Lower: {ci_lower_log:.4f}", showarrow=False))
                
                fig.add_vline(x=ci_upper_log, line=dict(color='purple', width=2, dash='dash'),
                             annotation=dict(text=f"Log Upper: {ci_upper_log:.4f}", showarrow=False))
                
                fig.update_layout(
                    title='Sample Distribution with Confidence Intervals',
                    xaxis_title='Ratio',
                    yaxis_title='Frequency',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                st.subheader("Interpretation")
                
                st.markdown(f"""
                ### Log Transformation for Ratio
                
                **Why use log transformation for ratios?**
                
                Ratios (and other multiplicative quantities) often have skewed sampling distributions. For example, a ratio of 2 (100% increase) is comparable to a ratio of 0.5 (50% decrease), but on a linear scale they are asymmetrical.
                
                **Results Comparison:**
                
                1. **Direct CI**: [{ci_lower_direct:.4f}, {ci_upper_direct:.4f}]
                   - Width: {ci_upper_direct - ci_lower_direct:.4f}
                   - {'Contains' if ci_lower_direct <= true_ratio <= ci_upper_direct else 'Does not contain'} the true ratio
                
                2. **Log-transformed CI**: [{ci_lower_log:.4f}, {ci_upper_log:.4f}]
                   - Width: {ci_upper_log - ci_lower_log:.4f}
                   - {'Contains' if ci_lower_log <= true_ratio <= ci_upper_log else 'Does not contain'} the true ratio
                
                **Key Observations:**
                
                - The direct CI is {'' if ci_lower_direct > 0 else 'not '}bounded away from zero
                - The log-transformed CI is always bounded away from zero (appropriate for ratios)
                - The log-transformed CI is {'symmetric' if np.isclose(sample_ratio - ci_lower_log, ci_upper_log - sample_ratio, rtol=0.05) else 'asymmetric'} around the sample ratio on the original scale
                - The direct CI is symmetric around the sample ratio on the original scale
                
                **When to use log transformation:**
                
                - When the coefficient of variation is large (> 0.3)
                - When the parameter must be positive (e.g., ratios, variances)
                - When the data is right-skewed
                """)
                
                # Add example from medicine
                st.markdown("""
                **Example Application:**
                
                In medical studies, relative risks and odds ratios are typically log-transformed for confidence interval construction. For instance, if a treatment reduces risk by 50% (relative risk = 0.5) or doubles it (relative risk = 2), these effects are of equal magnitude on the log scale: log(0.5) = -0.693, log(2) = 0.693.
                """)
            
            elif transform_type == "Logit Transformation for Proportion":
                # Logit transformation for proportion
                
                # Generate sample
                np.random.seed(None)
                sample = np.random.binomial(1, true_prop, sample_size)
                sample_prop = np.mean(sample)
                
                # Method 1: Wald interval
                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                wald_se = np.sqrt(sample_prop * (1 - sample_prop) / sample_size)
                wald_lower = max(0, sample_prop - z_crit * wald_se)
                wald_upper = min(1, sample_prop + z_crit * wald_se)
                
                # Method 2: Logit transformation
                # Avoid 0 and 1
                adj_prop = (sample_prop * sample_size + 0.5) / (sample_size + 1)
                logit_p = np.log(adj_prop / (1 - adj_prop))
                logit_se = np.sqrt(1 / (sample_size * adj_prop * (1 - adj_prop)))
                
                logit_lower = logit_p - z_crit * logit_se
                logit_upper = logit_p + z_crit * logit_se
                
                # Back-transform to original scale
                logit_lower_prop = 1 / (1 + np.exp(-logit_lower))
                logit_upper_prop = 1 / (1 + np.exp(-logit_upper))
                
                # Method 3: Wilson score interval
                wilson_denominator = 1 + z_crit**2/sample_size
                wilson_center = (sample_prop + z_crit**2/(2*sample_size)) / wilson_denominator
                wilson_margin = z_crit * np.sqrt((sample_prop*(1-sample_prop) + z_crit**2/(4*sample_size)) / sample_size) / wilson_denominator
                wilson_lower = max(0, wilson_center - wilson_margin)
                wilson_upper = min(1, wilson_center + wilson_margin)
                
                # Display results
                st.subheader("Simulation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Sample Proportion",
                        value=f"{sample_prop:.4f}"
                    )
                    
                    st.metric(
                        label="Number of Successes",
                        value=f"{int(np.sum(sample))}"
                    )
                
                with col2:
                    st.metric(
                        label="True Proportion",
                        value=f"{true_prop:.4f}"
                    )
                    
                    st.metric(
                        label="Sample Size",
                        value=f"{sample_size}"
                    )
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    'Method': ['Wald', 'Logit Transformation', 'Wilson Score'],
                    'Lower Bound': [wald_lower, logit_lower_prop, wilson_lower],
                    'Upper Bound': [wald_upper, logit_upper_prop, wilson_upper],
                    'Width': [wald_upper - wald_lower, logit_upper_prop - logit_lower_prop, wilson_upper - wilson_lower],
                    'Contains True Value': [wald_lower <= true_prop <= wald_upper,
                                          logit_lower_prop <= true_prop <= logit_upper_prop,
                                          wilson_lower <= true_prop <= wilson_upper]
                })
                
                st.subheader(f"{conf_level*100:.0f}% Confidence Intervals")
                st.dataframe(comparison_df.style.format({
                    'Lower Bound': '{:.4f}',
                    'Upper Bound': '{:.4f}',
                    'Width': '{:.4f}',
                }))
                
                # Visualize the confidence intervals
                fig = go.Figure()
                
                # Create x-axis for proportion
                x = np.linspace(0, 1, 1000)
                
                # Add vertical lines
                fig.add_vline(x=sample_prop, line=dict(color='blue', width=2),
                             annotation=dict(text=f"Sample: {sample_prop:.4f}", showarrow=False))
                
                fig.add_vline(x=true_prop, line=dict(color='green', width=2, dash='dot'),
                             annotation=dict(text=f"True: {true_prop:.4f}", showarrow=False))
                
                # Add intervals as shaded regions
                methods = ['Wald', 'Logit', 'Wilson']
                colors = ['red', 'purple', 'orange']
                bounds = [
                    [wald_lower, wald_upper],
                    [logit_lower_prop, logit_upper_prop],
                    [wilson_lower, wilson_upper]
                ]
                
                # Add rectangles for intervals
                for i, method in enumerate(methods):
                    fig.add_shape(
                        type="rect",
                        x0=bounds[i][0],
                        x1=bounds[i][1],
                        y0=i+0.25,
                        y1=i+0.75,
                        fillcolor=colors[i],
                        opacity=0.5,
                        line=dict(width=0),
                    )
                    
                    # Add text labels
                    fig.add_annotation(
                        x=bounds[i][0] - 0.02,
                        y=i+0.5,
                        text=f"{bounds[i][0]:.4f}",
                        showarrow=False,
                        xanchor="right",
                        yanchor="middle"
                    )
                    
                    fig.add_annotation(
                        x=bounds[i][1] + 0.02,
                        y=i+0.5,
                        text=f"{bounds[i][1]:.4f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle"
                    )
                
                fig.update_layout(
                    title=f'{conf_level*100:.0f}% Confidence Intervals for Proportion',
                    xaxis_title='Proportion',
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[0.5, 1.5, 2.5],
                        ticktext=methods
                    ),
                    height=400,
                    showlegend=False,
                    xaxis=dict(range=[-0.1, 1.1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                st.subheader("Interpretation")
                
                st.markdown(f"""
                ### Logit Transformation for Proportion
                
                **Why use logit transformation for proportions?**
                
                Proportions are bounded between 0 and 1, which can lead to non-normal sampling distributions, especially for values near the boundaries. The logit transformation maps proportions to the entire real line, making the distribution more normal.
                
                **Results Comparison:**
                
                1. **Wald CI**: [{wald_lower:.4f}, {wald_upper:.4f}]
                   - Width: {wald_upper - wald_lower:.4f}
                   - {'Contains' if wald_lower <= true_prop <= wald_upper else 'Does not contain'} the true proportion
                
                2. **Logit-transformed CI**: [{logit_lower_prop:.4f}, {logit_upper_prop:.4f}]
                   - Width: {logit_upper_prop - logit_lower_prop:.4f}
                   - {'Contains' if logit_lower_prop <= true_prop <= logit_upper_prop else 'Does not contain'} the true proportion
                
                3. **Wilson Score CI**: [{wilson_lower:.4f}, {wilson_upper:.4f}]
                   - Width: {wilson_upper - wilson_lower:.4f}
                   - {'Contains' if wilson_lower <= true_prop <= wilson_upper else 'Does not contain'} the true proportion
                
                **Key Observations:**
                
                - The Wald interval can extend below 0 or above 1 for proportions near the boundaries (though truncated here)
                - The logit-transformed interval respects the 0-1 bounds naturally
                - The Wilson Score interval also respects the 0-1 bounds and generally has good coverage properties
                
                **When to use logit transformation:**
                
                - When the proportion is near 0 or 1
                - When sample size is small relative to the expected number of successes/failures
                - When wanting to ensure the interval stays within [0,1]
                """)
                
                # Add example
                st.markdown("""
                **Example Application:**
                
                In clinical trials with rare events, the logit transformation helps prevent confidence intervals from extending beyond realistic bounds. For instance, if a new treatment shows 0 adverse events in 100 patients, a naive CI would extend below 0, while a logit-transformed CI would provide a more reasonable upper bound for the true rate.
                """)
            
            elif transform_type == "Fisher's Z Transformation for Correlation":
                # Fisher's Z transformation for correlation
                
                # Generate correlated data
                np.random.seed(None)
                
                # Generate bivariate normal data with specified correlation
                rho = true_corr
                n = sample_size
                
                # Cholesky decomposition
                cov_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(cov_matrix)
                
                # Generate uncorrelated data
                uncorrelated = np.random.normal(0, 1, size=(2, n))
                
                # Apply correlation structure
                correlated = np.dot(L, uncorrelated)
                
                # Extract variables
                x = correlated[0]
                y = correlated[1]
                
                # Calculate sample correlation
                sample_corr = np.corrcoef(x, y)[0, 1]
                
                # Method 1: Direct CI using Fisher's z-transformation
                z = np.arctanh(sample_corr)  # Fisher's z-transformation
                z_se = 1 / np.sqrt(n - 3)
                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                
                z_lower = z - z_crit * z_se
                z_upper = z + z_crit * z_se
                
                # Back-transform to correlation scale
                fisher_lower = np.tanh(z_lower)
                fisher_upper = np.tanh(z_upper)
                
                # Method 2: Bootstrap CI
                n_bootstrap = 1000
                bootstrap_corrs = []
                
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    indices = np.random.choice(n, n, replace=True)
                    x_bootstrap = x[indices]
                    y_bootstrap = y[indices]
                    
                    # Calculate correlation
                    bootstrap_corr = np.corrcoef(x_bootstrap, y_bootstrap)[0, 1]
                    bootstrap_corrs.append(bootstrap_corr)
                
                # Percentile bootstrap CI
                bootstrap_lower = np.percentile(bootstrap_corrs, 100 * (1 - conf_level) / 2)
                bootstrap_upper = np.percentile(bootstrap_corrs, 100 * (1 - (1 - conf_level) / 2))
                
                # Display results
                st.subheader("Simulation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Sample Correlation",
                        value=f"{sample_corr:.4f}"
                    )
                    
                    st.metric(
                        label="Fisher's Z",
                        value=f"{z:.4f}"
                    )
                
                with col2:
                    st.metric(
                        label="True Correlation",
                        value=f"{true_corr:.4f}"
                    )
                    
                    st.metric(
                        label="Sample Size",
                        value=f"{sample_size}"
                    )
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    'Method': ["Fisher's Z Transformation", 'Bootstrap Percentile'],
                    'Lower Bound': [fisher_lower, bootstrap_lower],
                    'Upper Bound': [fisher_upper, bootstrap_upper],
                    'Width': [fisher_upper - fisher_lower, bootstrap_upper - bootstrap_lower],
                    'Contains True Value': [fisher_lower <= true_corr <= fisher_upper,
                                          bootstrap_lower <= true_corr <= bootstrap_upper]
                })
                
                st.subheader(f"{conf_level*100:.0f}% Confidence Intervals")
                st.dataframe(comparison_df.style.format({
                    'Lower Bound': '{:.4f}',
                    'Upper Bound': '{:.4f}',
                    'Width': '{:.4f}',
                }))
                
                # Visualize the scatter plot
                scatter_fig = go.Figure()
                
                scatter_fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(size=8, opacity=0.7),
                    name='Sample Data'
                ))
                
                scatter_fig.update_layout(
                    title=f'Scatter Plot (r = {sample_corr:.4f})',
                    xaxis_title='X',
                    yaxis_title='Y',
                    height=500
                )
                
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Visualize the bootstrap distribution
                boot_fig = go.Figure()
                
                # Add histogram of bootstrap correlations
                boot_fig.add_trace(go.Histogram(
                    x=bootstrap_corrs,
                    nbinsx=30,
                    opacity=0.7,
                    name="Bootstrap Distribution"
                ))
                
                # Add vertical lines
                boot_fig.add_vline(x=sample_corr, line=dict(color='blue', width=2),
                             annotation=dict(text=f"Sample: {sample_corr:.4f}", showarrow=False))
                
                boot_fig.add_vline(x=true_corr, line=dict(color='green', width=2, dash='dot'),
                             annotation=dict(text=f"True: {true_corr:.4f}", showarrow=False))
                
                # Add CI bounds
                boot_fig.add_vline(x=fisher_lower, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"Fisher Lower: {fisher_lower:.4f}", showarrow=False))
                
                boot_fig.add_vline(x=fisher_upper, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"Fisher Upper: {fisher_upper:.4f}", showarrow=False))
                
                boot_fig.add_vline(x=bootstrap_lower, line=dict(color='purple', width=2, dash='dash'),
                             annotation=dict(text=f"Boot Lower: {bootstrap_lower:.4f}", showarrow=False))
                
                boot_fig.add_vline(x=bootstrap_upper, line=dict(color='purple', width=2, dash='dash'),
                             annotation=dict(text=f"Boot Upper: {bootstrap_upper:.4f}", showarrow=False))
                
                boot_fig.update_layout(
                    title='Bootstrap Distribution of Correlation Coefficient',
                    xaxis_title='Correlation',
                    yaxis_title='Frequency',
                    height=400
                )
                
                st.plotly_chart(boot_fig, use_container_width=True)
                
                # Add Fisher's Z transformation visualization
                z_fig = go.Figure()
                
                # Create x-axis for correlation values
                corr_vals = np.linspace(-0.99, 0.99, 1000)
                z_vals = np.arctanh(corr_vals)
                
                # Add transformation curve
                z_fig.add_trace(go.Scatter(
                    x=corr_vals,
                    y=z_vals,
                    mode='lines',
                    name="Fisher's Z Transform",
                    line=dict(color='blue', width=2)
                ))
                
                # Add points for sample correlation
                z_fig.add_trace(go.Scatter(
                    x=[sample_corr],
                    y=[z],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Sample Correlation'
                ))
                
                # Add horizontal lines for Z interval
                z_fig.add_shape(
                    type="line",
                    x0=-1,
                    x1=1,
                    y0=z_lower,
                    y1=z_lower,
                    line=dict(color='red', width=2, dash='dash')
                )
                
                z_fig.add_shape(
                    type="line",
                    x0=-1,
                    x1=1,
                    y0=z_upper,
                    y1=z_upper,
                    line=dict(color='red', width=2, dash='dash')
                )
                
                # Add text annotations
                z_fig.add_annotation(
                    x=0.8,
                    y=z_lower,
                    text=f"z_lower = {z_lower:.4f}",
                    showarrow=False,
                    yshift=-10
                )
                
                z_fig.add_annotation(
                    x=0.8,
                    y=z_upper,
                    text=f"z_upper = {z_upper:.4f}",
                    showarrow=False,
                    yshift=10
                )
                
                z_fig.update_layout(
                    title="Fisher's Z Transformation",
                    xaxis_title='Correlation (r)',
                    yaxis_title="Fisher's Z",
                    height=400
                )
                
                st.plotly_chart(z_fig, use_container_width=True)
                
                # Add interpretation
                st.subheader("Interpretation")
                
                st.markdown(f"""
                ### Fisher's Z Transformation for Correlation
                
                **Why use Fisher's Z transformation for correlation?**
                
                Correlation coefficients are bounded between -1 and 1, and their sampling distribution becomes skewed as the true correlation approaches these boundaries. Fisher's Z transformation maps correlations to the entire real line, resulting in an approximately normal sampling distribution with nearly constant variance.
                
                **Transformation Details:**
                
                - Fisher's Z transform: z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
                - Standard error of z: SE(z) = 1/sqrt(n-3)
                - For your sample correlation (r = {sample_corr:.4f}): z = {z:.4f}
                
                **Results Comparison:**
                
                1. **Fisher's Z CI**: [{fisher_lower:.4f}, {fisher_upper:.4f}]
                   - Width: {fisher_upper - fisher_lower:.4f}
                   - {'Contains' if fisher_lower <= true_corr <= fisher_upper else 'Does not contain'} the true correlation
                
                2. **Bootstrap CI**: [{bootstrap_lower:.4f}, {bootstrap_upper:.4f}]
                   - Width: {bootstrap_upper - bootstrap_lower:.4f}
                   - {'Contains' if bootstrap_lower <= true_corr <= bootstrap_upper else 'Does not contain'} the true correlation
                
                **Key Observations:**
                
                - Fisher's Z interval is {'symmetric' if np.isclose(sample_corr - fisher_lower, fisher_upper - sample_corr, rtol=0.05) else 'asymmetric'} around the sample correlation on the original scale
                - The bootstrap interval is {'symmetric' if np.isclose(sample_corr - bootstrap_lower, bootstrap_upper - sample_corr, rtol=0.05) else 'asymmetric'} around the sample correlation
                - Fisher's Z transformation performs better as |r| approaches 1
                - The approximation works well for sample sizes > 10
                
                **When to use Fisher's Z transformation:**
                
                - When the correlation is strong (|r| > 0.5)
                - When constructing confidence intervals for correlation coefficients
                - When testing hypotheses about correlation coefficients
                - When performing meta-analysis of correlation studies
                """)
                
                # Add example
                st.markdown("""
                **Example Application:**
                
                In meta-analysis of psychological studies, correlation coefficients are often the effect size of interest. Fisher's Z transformation is routinely applied to correlations before combining results across studies, as it normalizes the sampling distribution and stabilizes the variance.
                """)
    
    elif sim_type == "Non-normality Impact":
        st.subheader("Impact of Non-normality on Confidence Intervals")
        
        st.markdown("""
        This simulation demonstrates how departures from normality affect the coverage and width of confidence intervals for various parameters.
        
        Many standard confidence intervals rely on normality assumptions. Understanding the robustness of these methods to violations of normality is important for real-world applications.
        """)
        
        # Select parameter and distribution
        col1, col2 = st.columns(2)
        
        with col1:
            parameter = st.radio(
                "Parameter of interest",
                ["Mean", "Median", "Variance"]
            )
            
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
            sample_size = st.slider("Sample size", 10, 200, 30)
        
        with col2:
            distribution = st.radio(
                "Distribution type",
                ["Normal", "Skewed (Log-normal)", "Heavy-tailed (t with df=3)", 
                 "Bimodal", "Contaminated Normal"]
            )
            
            n_sims = st.slider("Number of simulations", 500, 5000, 1000, 500)
        
        if st.button("Run Simulation", key="run_nonnormal_sim"):
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Set true parameter values based on distribution
            if distribution == "Normal":
                mean = 0
                std = 1
                true_mean = mean
                true_median = mean
                true_var = std**2
                
                # Function to generate samples
                def generate_sample(n):
                    return np.random.normal(mean, std, n)
                
            elif distribution == "Skewed (Log-normal)":
                mu = 0
                sigma = 0.5
                true_mean = np.exp(mu + sigma**2/2)
                true_median = np.exp(mu)
                true_var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
                
                # Function to generate samples
                def generate_sample(n):
                    return np.random.lognormal(mu, sigma, n)
                
            elif distribution == "Heavy-tailed (t with df=3)":
                df = 3
                true_mean = 0 if df > 1 else np.nan
                true_median = 0
                true_var = df / (df - 2) if df > 2 else np.nan
                
                # Function to generate samples
                def generate_sample(n):
                    return stats.t.rvs(df, size=n)
                
            elif distribution == "Bimodal":
                # Mixture of two normals
                mean1, std1 = -2, 1
                mean2, std2 = 2, 1
                mix_prob = 0.5
                true_mean = mix_prob * mean1 + (1 - mix_prob) * mean2
                # True median depends on the mixture weights
                if mix_prob == 0.5:
                    true_median = (mean1 + mean2) / 2
                elif mix_prob < 0.5:
                    true_median = mean2
                else:
                    true_median = mean1
                true_var = (mix_prob * (std1**2 + mean1**2) + 
                           (1 - mix_prob) * (std2**2 + mean2**2) - 
                           true_mean**2)
                
                # Function to generate samples
                def generate_sample(n):
                    mask = np.random.random(n) < mix_prob
                    return mask * np.random.normal(mean1, std1, n) + (1 - mask) * np.random.normal(mean2, std2, n)
                
            elif distribution == "Contaminated Normal":
                # Normal with occasional outliers
                main_mean, main_std = 0, 1
                contam_mean, contam_std = 0, 5
                contam_prob = 0.1
                true_mean = (1 - contam_prob) * main_mean + contam_prob * contam_mean
                true_median = main_mean  # Median is robust to contamination
                true_var = ((1 - contam_prob) * (main_std**2 + main_mean**2) + 
                           contam_prob * (contam_std**2 + contam_mean**2) - 
                           true_mean**2)
                
                # Function to generate samples
                def generate_sample(n):
                    mask = np.random.random(n) < contam_prob
                    return (1 - mask) * np.random.normal(main_mean, main_std, n) + mask * np.random.normal(contam_mean, contam_std, n)
            
            # Initialize tracking variables
            standard_contains = 0
            robust_contains = 0
            bootstrap_contains = 0
            
            standard_widths = []
            robust_widths = []
            bootstrap_widths = []
            
            # Run simulations
            for i in range(n_sims):
                # Update progress
                if i % (n_sims // 20) == 0:
                    progress = (i + 1) / n_sims
                    progress_bar.progress(progress)
                    status_text.text(f"Running simulation {i+1}/{n_sims}")
                
                # Generate sample
                sample = generate_sample(sample_size)
                
                if parameter == "Mean":
                    # Standard method (t-interval)
                    sample_mean = np.mean(sample)
                    sample_std = np.std(sample, ddof=1)
                    t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                    
                    standard_margin = t_crit * sample_std / np.sqrt(sample_size)
                    standard_lower = sample_mean - standard_margin
                    standard_upper = sample_mean + standard_margin
                    
                    # Check if interval contains true parameter
                    standard_contains += (standard_lower <= true_mean <= standard_upper)
                    standard_widths.append(standard_upper - standard_lower)
                    
                    # Robust method (trimmed mean)
                    trim_prop = 0.2  # 20% trimming
                    trimmed_mean = stats.trim_mean(sample, trim_prop)
                    
                    # Use bootstrap for trimmed mean CI
                    trim_bootstrap_means = []
                    for _ in range(200):  # 200 bootstrap samples
                        boot_sample = np.random.choice(sample, sample_size, replace=True)
                        trim_bootstrap_means.append(stats.trim_mean(boot_sample, trim_prop))
                    
                    robust_lower = np.percentile(trim_bootstrap_means, 100 * (1 - conf_level) / 2)
                    robust_upper = np.percentile(trim_bootstrap_means, 100 * (1 - (1 - conf_level) / 2))
                    
                    # Check if interval contains true parameter
                    # Note: True trimmed mean may differ from true mean
                    # Using true mean as a proxy here
                    robust_contains += (robust_lower <= true_mean <= robust_upper)
                    robust_widths.append(robust_upper - robust_lower)
                    
                    # Bootstrap method (regular bootstrap for mean)
                    bootstrap_means = []
                    for _ in range(200):  # 200 bootstrap samples
                        boot_sample = np.random.choice(sample, sample_size, replace=True)
                        bootstrap_means.append(np.mean(boot_sample))
                    
                    bootstrap_lower = np.percentile(bootstrap_means, 100 * (1 - conf_level) / 2)
                    bootstrap_upper = np.percentile(bootstrap_means, 100 * (1 - (1 - conf_level) / 2))
                    
                    # Check if interval contains true parameter
                    bootstrap_contains += (bootstrap_lower <= true_mean <= bootstrap_upper)
                    bootstrap_widths.append(bootstrap_upper - bootstrap_lower)
                
                elif parameter == "Median":
                    # Standard method (normal approximation)
                    sample_median = np.median(sample)
                    
                    # Standard error of median using asymptotic formula
                    # SE(median) ≈ 1 / (2 * f(F^(-1)(0.5)) * sqrt(n))
                    # where f is the PDF and F^(-1)(0.5) is the median
                    # For normal, f(median) = 1/sqrt(2*pi*sigma^2)
                    # Approximating sigma with MAD
                    mad = stats.median_abs_deviation(sample)
                    sigma_est = mad / 0.6745  # Convert MAD to sigma for normal
                    
                    med_se = 1.2533 * sigma_est / np.sqrt(sample_size)  # 1.2533 = sqrt(pi/2)
                    z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                    
                    standard_margin = z_crit * med_se
                    standard_lower = sample_median - standard_margin
                    standard_upper = sample_median + standard_margin
                    
                    # Check if interval contains true parameter
                    standard_contains += (standard_lower <= true_median <= standard_upper)
                    standard_widths.append(standard_upper - standard_lower)
                    
                    # Robust method (sign test)
                    k = stats.binom.ppf((1 - conf_level)/2, sample_size, 0.5)
                    j = sample_size - k
                    
                    # Sort the sample
                    sorted_sample = np.sort(sample)
                    
                    if k < 1:
                        robust_lower = sorted_sample[0]
                    else:
                        robust_lower = sorted_sample[int(k) - 1]
                    
                    if j >= sample_size:
                        robust_upper = sorted_sample[-1]
                    else:
                        robust_upper = sorted_sample[int(j)]
                    
                    # Check if interval contains true parameter
                    robust_contains += (robust_lower <= true_median <= robust_upper)
                    robust_widths.append(robust_upper - robust_lower)
                    
                    # Bootstrap method
                    bootstrap_medians = []
                    for _ in range(200):  # 200 bootstrap samples
                        boot_sample = np.random.choice(sample, sample_size, replace=True)
                        bootstrap_medians.append(np.median(boot_sample))
                    
                    bootstrap_lower = np.percentile(bootstrap_medians, 100 * (1 - conf_level) / 2)
                    bootstrap_upper = np.percentile(bootstrap_medians, 100 * (1 - (1 - conf_level) / 2))
                    
                    # Check if interval contains true parameter
                    bootstrap_contains += (bootstrap_lower <= true_median <= bootstrap_upper)
                    bootstrap_widths.append(bootstrap_upper - bootstrap_lower)
                
                elif parameter == "Variance":
                    # Standard method (chi-square)
                    sample_var = np.var(sample, ddof=1)
                    
                    chi2_lower = stats.chi2.ppf((1 - conf_level)/2, sample_size - 1)
                    chi2_upper = stats.chi2.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                    
                    standard_lower = (sample_size - 1) * sample_var / chi2_upper
                    standard_upper = (sample_size - 1) * sample_var / chi2_lower
                    
                    # Check if interval contains true parameter
                    standard_contains += (standard_lower <= true_var <= standard_upper)
                    standard_widths.append(standard_upper - standard_lower)
                    
                    # Robust method (trimmed variance with bootstrap)
                    # For variance, we'll use a robust estimator like biweight midvariance
                    def biweight_midvariance(x):
                        # Tuning constant
                        c = 9.0
                        
                        # Median and MAD
                        M = np.median(x)
                        MAD = stats.median_abs_deviation(x)
                        
                        # Avoid division by zero
                        if MAD == 0:
                            return np.var(x, ddof=1)
                        
                        # Standardized distance
                        u = (x - M) / (c * MAD)
                        
                        # Weights
                        w = (1 - u**2)**2
                        w[np.abs(u) >= 1] = 0
                        
                        # Biweight midvariance
                        n = len(x)
                        numerator = np.sum(w * (x - M)**2)
                        denominator = np.sum(w)**2 - np.sum(w**2)
                        
                        return n * numerator / denominator
                    
                    robust_var = biweight_midvariance(sample)
                    
                    # Bootstrap for robust variance
                    robust_bootstrap_vars = []
                    for _ in range(200):  # 200 bootstrap samples
                        boot_sample = np.random.choice(sample, sample_size, replace=True)
                        robust_bootstrap_vars.append(biweight_midvariance(boot_sample))
                    
                    robust_lower = np.percentile(robust_bootstrap_vars, 100 * (1 - conf_level) / 2)
                    robust_upper = np.percentile(robust_bootstrap_vars, 100 * (1 - (1 - conf_level) / 2))
                    
                    # Check if interval contains true parameter
                    robust_contains += (robust_lower <= true_var <= robust_upper)
                    robust_widths.append(robust_upper - robust_lower)
                    
                    # Bootstrap method (regular bootstrap for variance)
                    bootstrap_vars = []
                    for _ in range(200):  # 200 bootstrap samples
                        boot_sample = np.random.choice(sample, sample_size, replace=True)
                        bootstrap_vars.append(np.var(boot_sample, ddof=1))
                    
                    bootstrap_lower = np.percentile(bootstrap_vars, 100 * (1 - conf_level) / 2)
                    bootstrap_upper = np.percentile(bootstrap_vars, 100 * (1 - (1 - conf_level) / 2))
                    
                    # Check if interval contains true parameter
                    bootstrap_contains += (bootstrap_lower <= true_var <= bootstrap_upper)
                    bootstrap_widths.append(bootstrap_upper - bootstrap_lower)
            
            # Clear progress indicators
            status_text.empty()
            progress_bar.empty()
            
            # Calculate actual coverage and average widths
            standard_coverage = standard_contains / n_sims * 100
            robust_coverage = robust_contains / n_sims * 100
            bootstrap_coverage = bootstrap_contains / n_sims * 100
            
            standard_avg_width = np.mean(standard_widths)
            robust_avg_width = np.mean(robust_widths)
            bootstrap_avg_width = np.mean(bootstrap_widths)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Method': ['Standard Parametric', 'Robust', 'Bootstrap'],
                'Coverage (%)': [standard_coverage, robust_coverage, bootstrap_coverage],
                'Average Width': [standard_avg_width, robust_avg_width, bootstrap_avg_width],
                'Count': [standard_contains, robust_contains, bootstrap_contains]
            })
            
            # Display results
            st.subheader("Simulation Results")
            st.dataframe(results_df.style.format({
                'Coverage (%)': '{:.1f}',
                'Average Width': '{:.4f}'
            }))
            
            # Create coverage visualization
            fig = px.bar(
                results_df, 
                x='Method', 
                y='Coverage (%)',
                text_auto='.1f',
                color='Method',
                title=f"Actual Coverage of {conf_level*100:.0f}% Confidence Intervals<br>"
                      f"({n_sims} simulations, {distribution} distribution, n={sample_size})"
            )
            
            # Add horizontal line for nominal coverage
            fig.add_hline(y=conf_level*100, line=dict(color='black', width=2, dash='dash'),
                         annotation=dict(text=f"Nominal {conf_level*100:.0f}%", 
                                       showarrow=False,
                                       yref="y",
                                       xref="paper",
                                       x=1.05))
            
            fig.update_layout(
                yaxis=dict(
                    title='Actual Coverage (%)',
                    range=[min(min(standard_coverage, robust_coverage, bootstrap_coverage),
                               conf_level*100)*0.95, 
                           max(max(standard_coverage, robust_coverage, bootstrap_coverage),
                               conf_level*100)*1.05]
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create width comparison
            width_fig = px.bar(
                results_df, 
                x='Method', 
                y='Average Width',
                text_auto='.4f',
                color='Method',
                title=f"Average Interval Width<br>"
                      f"({n_sims} simulations, {distribution} distribution, n={sample_size})"
            )
            
            width_fig.update_layout(
                yaxis=dict(title='Average Width'),
                height=400
            )
            
            st.plotly_chart(width_fig, use_container_width=True)
            
            # Generate a sample for visualization
            vis_sample = generate_sample(1000)  # Large sample for good visualization
            
            # Create distribution visualization
            dist_fig = go.Figure()
            
            # Add histogram
            dist_fig.add_trace(go.Histogram(
                x=vis_sample,
                nbinsx=30,
                opacity=0.7,
                name=distribution
            ))
            
            # Add vertical lines for true parameters
            if parameter == "Mean":
                dist_fig.add_vline(x=true_mean, line=dict(color='red', width=2, dash='dash'),
                                 annotation=dict(text=f"True Mean: {true_mean:.4f}", showarrow=False))
            elif parameter == "Median":
                dist_fig.add_vline(x=true_median, line=dict(color='green', width=2, dash='dash'),
                                 annotation=dict(text=f"True Median: {true_median:.4f}", showarrow=False))
            
            dist_fig.update_layout(
                title=f'{distribution} Distribution',
                xaxis_title='Value',
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Add QQ-plot to assess normality
            qq_fig = go.Figure()
            
            # Generate theoretical quantiles from normal distribution
            theo_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
            
            # Calculate sample quantiles
            sample_quantiles = np.quantile(vis_sample, np.linspace(0.01, 0.99, 100))
            
            # Add scatter plot
            qq_fig.add_trace(go.Scatter(
                x=theo_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='QQ Plot'
            ))
            
            # Add reference line
            min_q = min(min(theo_quantiles), min(sample_quantiles))
            max_q = max(max(theo_quantiles), max(sample_quantiles))
            
            qq_fig.add_trace(go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ))
            
            qq_fig.update_layout(
                title='Normal Q-Q Plot',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                height=400
            )
            
            st.plotly_chart(qq_fig, use_container_width=True)
            
            # Add interpretation
            st.subheader("Interpretation")
            
            st.markdown(f"""
            ### Impact of Non-normality on {parameter} Confidence Intervals
            
            **Distribution Characteristics:**
            
            The {distribution} distribution used in this simulation {'follows' if distribution == 'Normal' else 'deviates from'} normal distribution assumptions.
            """)
            
            if distribution == "Skewed (Log-normal)":
                st.markdown("""
                This distribution has a strong right skew, meaning it has a long tail on the right side. This violates the symmetry assumption of many standard confidence interval methods.
                """)
            elif distribution == "Heavy-tailed (t with df=3)":
                st.markdown("""
                This distribution has heavier tails than the normal distribution, meaning extreme values occur more frequently. This can lead to increased variability in sample statistics.
                """)
            elif distribution == "Bimodal":
                st.markdown("""
                This distribution has two peaks (modes), violating the unimodality assumption of the normal distribution. This can affect the behavior of sample means and especially medians.
                """)
            elif distribution == "Contaminated Normal":
                st.markdown("""
                This distribution represents a normal distribution with occasional outliers. It models a situation where most data follows normal assumptions, but some observations come from a different process.
                """)
            
            st.markdown(f"""
            **Results Comparison:**
            
            1. **Standard Parametric Method**:
               - Coverage: {standard_coverage:.1f}% (Target: {conf_level*100:.0f}%)
               - Average Width: {standard_avg_width:.4f}
               - {'Adequate' if abs(standard_coverage - conf_level*100) < 5 else 'Poor'} coverage under {distribution} distribution
            
            2. **Robust Method**:
               - Coverage: {robust_coverage:.1f}% (Target: {conf_level*100:.0f}%)
               - Average Width: {robust_avg_width:.4f}
               - {'Adequate' if abs(robust_coverage - conf_level*100) < 5 else 'Poor'} coverage under {distribution} distribution
            
            3. **Bootstrap Method**:
               - Coverage: {bootstrap_coverage:.1f}% (Target: {conf_level*100:.0f}%)
               - Average Width: {bootstrap_avg_width:.4f}
               - {'Adequate' if abs(bootstrap_coverage - conf_level*100) < 5 else 'Poor'} coverage under {distribution} distribution
            """)
            
            # Add parameter-specific interpretations
            if parameter == "Mean":
                st.markdown(f"""
                **Insights for Mean Confidence Intervals:**
                
                - The standard t-interval {'is robust' if abs(standard_coverage - conf_level*100) < 5 else 'lacks robustness'} to the {distribution} distribution
                - {'Trimmed means provide better coverage than standard means for this non-normal distribution' if robust_coverage > standard_coverage + 2 else 'Trimming does not significantly improve coverage for this distribution'}
                - Bootstrap methods {'offer an advantage' if bootstrap_coverage > standard_coverage + 2 else 'do not provide significant improvements'} over parametric methods in this case
                
                According to the Central Limit Theorem, the sampling distribution of the mean approaches normality as sample size increases, regardless of the original distribution. With n={sample_size}, this effect is {'strong' if sample_size >= 30 else 'moderate' if sample_size >= 15 else 'weak'}.
                """)
            
            elif parameter == "Median":
                st.markdown(f"""
                **Insights for Median Confidence Intervals:**
                
                - The asymptotic normal approximation for the median {'works well' if abs(standard_coverage - conf_level*100) < 5 else 'is problematic'} for the {distribution} distribution
                - The sign test method {'provides better coverage' if robust_coverage > standard_coverage + 2 else 'does not significantly improve coverage'} compared to the normal approximation
                - Bootstrap methods {'offer clear advantages' if bootstrap_coverage > standard_coverage + 2 else 'provide similar performance'} to parametric methods
                
                The median is generally more robust to outliers and non-normality than the mean, which {'is evident' if abs(robust_coverage - conf_level*100) < abs(standard_coverage - conf_level*100) else 'is not strongly demonstrated'} in these results.
                """)
            
            elif parameter == "Variance":
                st.markdown(f"""
                **Insights for Variance Confidence Intervals:**
                
                - The chi-square interval {'performs adequately' if abs(standard_coverage - conf_level*100) < 5 else 'performs poorly'} for the {distribution} distribution
                - The robust variance estimator {'provides better coverage' if robust_coverage > standard_coverage + 2 else 'does not significantly improve coverage'}
                - Bootstrap methods {'show clear advantages' if bootstrap_coverage > standard_coverage + 2 else 'offer similar performance'} to the chi-square method
                
                Variance estimates are particularly sensitive to non-normality, especially with heavier tails or outliers. The chi-square interval relies heavily on normality assumptions and {'is showing resilience' if abs(standard_coverage - conf_level*100) < 5 else 'is breaking down'} under this distribution.
                """)
            
            # Add practical recommendations
            st.subheader("Practical Recommendations")
            
            st.markdown(f"""
            Based on the simulation results for {parameter} with {distribution} distribution:
            
            1. **Best method for this scenario**: 
               {
                'Standard Parametric' 
                if standard_coverage >= robust_coverage and standard_coverage >= bootstrap_coverage 
                else 'Robust' 
                if robust_coverage >= standard_coverage and robust_coverage >= bootstrap_coverage
                else 'Bootstrap'
               }
            
            2. **When to use each method**:
               - **Standard parametric**: When data is approximately normal or sample size is large
               - **Robust methods**: When outliers or heavy tails are present
               - **Bootstrap**: When distribution shape is unknown or complex
            
            3. **Trade-offs to consider**:
               - Coverage accuracy vs. interval width
               - Computational complexity vs. theoretical simplicity
               - Robustness vs. efficiency under ideal conditions
            
            4. **Diagnostic checks recommended**:
               - Q-Q plots to assess normality
               - Tests for skewness and kurtosis
               - Outlier detection methods
            """)
            
            # Add references or further reading
            st.markdown("""
            **Further Reading:**
            
            1. Wilcox, R. R. (2012). Introduction to Robust Estimation and Hypothesis Testing (3rd ed.). Academic Press.
            2. Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. Chapman & Hall/CRC.
            3. Hampel, F. R., Ronchetti, E. M., Rousseeuw, P. J., & Stahel, W. A. (2011). Robust Statistics: The Approach Based on Influence Functions. Wiley.
            """)

# Advanced Methods Module
elif nav == "Advanced Methods":
    st.header("Advanced Confidence Interval Methods")
    
    method_type = st.selectbox(
        "Select method",
        ["Bayesian Credible Intervals", "Multiple Testing Adjustment", 
         "Profile Likelihood Intervals", "Simultaneous Confidence Bands"]
    )
    
    if method_type == "Bayesian Credible Intervals":
        st.subheader("Bayesian Credible Intervals")
        
        st.markdown("""
        Bayesian credible intervals are the Bayesian analog to frequentist confidence intervals, but with a more direct probability interpretation.
        
        A 95% credible interval means that, given the observed data and prior beliefs, there is a 95% probability that the true parameter lies within the interval.
        """)
        
        # Bayesian example options
        col1, col2 = st.columns(2)
        
        with col1:
            parameter = st.radio(
                "Parameter of interest",
                ["Mean (Normal)", "Proportion (Binomial)", "Rate (Poisson)"]
            )
            credible_level = st.slider("Credible level", 0.80, 0.99, 0.95, 0.01)
        
        with col2:
            prior_type = st.radio(
                "Prior type",
                ["Informative", "Weakly informative", "Non-informative"]
            )
            sample_size = st.slider("Sample size", 5, 100, 20)
        
        if parameter == "Mean (Normal)":
            # Additional options for normal mean
            col1, col2, col3 = st.columns(3)
            
            with col1:
                true_mean = st.slider("True population mean", -10.0, 10.0, 0.0, 0.5)
                true_sd = st.slider("True population std dev", 0.1, 10.0, 1.0, 0.1)
            
            with col2:
                if prior_type == "Informative":
                    prior_mean = st.slider("Prior mean", -10.0, 10.0, 0.0, 0.5)
                    prior_sd = st.slider("Prior std dev", 0.1, 10.0, 2.0, 0.1)
                elif prior_type == "Weakly informative":
                    prior_mean = st.slider("Prior mean", -10.0, 10.0, 0.0, 0.5)
                    prior_sd = st.slider("Prior std dev", 0.1, 10.0, 5.0, 0.1)
            
            with col3:
                known_variance = st.checkbox("Known variance", value=True)
                if not known_variance:
                    prior_alpha = st.slider("Prior alpha (variance)", 0.1, 10.0, 1.0, 0.1)
                    prior_beta = st.slider("Prior beta (variance)", 0.1, 10.0, 1.0, 0.1)
        
        elif parameter == "Proportion (Binomial)":
            # Additional options for binomial proportion
            col1, col2 = st.columns(2)
            
            with col1:
                true_prop = st.slider("True proportion", 0.01, 0.99, 0.3, 0.01)
            
            with col2:
                if prior_type == "Informative":
                    prior_alpha = st.slider("Prior alpha", 0.5, 20.0, 2.0, 0.5)
                    prior_beta = st.slider("Prior beta", 0.5, 20.0, 5.0, 0.5)
                elif prior_type == "Weakly informative":
                    prior_alpha = st.slider("Prior alpha", 0.5, 5.0, 1.0, 0.5)
                    prior_beta = st.slider("Prior beta", 0.5, 5.0, 1.0, 0.5)
                else:  # Non-informative
                    prior_alpha = 0.5  # Jeffreys prior
                    prior_beta = 0.5   # Jeffreys prior
                    st.markdown("Using Jeffreys prior: Beta(0.5, 0.5)")
        
        elif parameter == "Rate (Poisson)":
            # Additional options for poisson rate
            col1, col2 = st.columns(2)
            
            with col1:
                true_rate = st.slider("True rate", 0.1, 20.0, 5.0, 0.1)
            
            with col2:
                if prior_type == "Informative":
                    prior_alpha = st.slider("Prior alpha", 0.5, 20.0, 5.0, 0.5)
                    prior_beta = st.slider("Prior beta", 0.1, 5.0, 1.0, 0.1)
                elif prior_type == "Weakly informative":
                    prior_alpha = st.slider("Prior alpha", 0.5, 5.0, 1.0, 0.5)
                    prior_beta = st.slider("Prior beta", 0.1, 5.0, 0.1, 0.1)
                else:  # Non-informative
                    prior_alpha = 0.001  # Approximately improper prior
                    prior_beta = 0.001
                    st.markdown("Using approximately improper prior: Gamma(0.001, 0.001)")
        
        if st.button("Generate Bayesian Credible Interval", key="gen_bayesian"):
            # Generate data
            np.random.seed(None)
            
            if parameter == "Mean (Normal)":
                # Generate normal data
                data = np.random.normal(true_mean, true_sd, sample_size)
                sample_mean = np.mean(data)
                sample_var = np.var(data, ddof=1)
                
                # Frequentist CI
                t_crit = stats.t.ppf(1 - (1 - credible_level)/2, sample_size - 1)
                freq_margin = t_crit * np.sqrt(sample_var / sample_size)
                freq_lower = sample_mean - freq_margin
                freq_upper = sample_mean + freq_margin
                
                # Bayesian inference
                if known_variance:
                    # Known variance case (conjugate normal prior)
                    if prior_type == "Non-informative":
                        # Improper uniform prior (equivalent to just using likelihood)
                        post_mean = sample_mean
                        post_var = true_sd**2 / sample_size
                    else:
                        # Normal prior
                        prior_var = prior_sd**2
                        post_var = 1 / (1/prior_var + sample_size/true_sd**2)
                        post_mean = post_var * (prior_mean/prior_var + sample_mean*sample_size/true_sd**2)
                    
                    # Credible interval
                    post_sd = np.sqrt(post_var)
                    z_crit = stats.norm.ppf(1 - (1 - credible_level)/2)
                    bayes_margin = z_crit * post_sd
                    bayes_lower = post_mean - bayes_margin
                    bayes_upper = post_mean + bayes_margin
                    
                    # Method description
                    if prior_type == "Non-informative":
                        method = "Bayesian with improper uniform prior (known variance)"
                    else:
                        method = f"Bayesian with {prior_type} normal prior (known variance)"
                
                else:
                    # Unknown variance case (conjugate normal-inverse-gamma prior)
                    if prior_type == "Non-informative":
                        # Jeffreys prior for normal with unknown mean and variance
                        post_mean = sample_mean
                        post_df = sample_size - 1
                        post_scale = np.sqrt(sample_var / sample_size)
                    else:
                        # Normal-inverse-gamma prior
                        # This is a simplification; full conjugate analysis is more complex
                        # Here we just use a t-distribution with adjusted parameters
                        post_df = sample_size - 1 + 2 * prior_alpha
                        post_mean = sample_mean  # Simplification
                        post_scale = np.sqrt(sample_var / sample_size)  # Simplification
                    
                    # Credible interval (using t-distribution)
                    t_crit_bayes = stats.t.ppf(1 - (1 - credible_level)/2, post_df)
                    bayes_margin = t_crit_bayes * post_scale
                    bayes_lower = post_mean - bayes_margin
                    bayes_upper = post_mean + bayes_margin
                    
                    # Method description
                    if prior_type == "Non-informative":
                        method = "Bayesian with Jeffreys prior (unknown variance)"
                    else:
                        method = f"Bayesian with {prior_type} normal-inverse-gamma prior (unknown variance)"
            
            elif parameter == "Proportion (Binomial)":
                # Generate binomial data
                data = np.random.binomial(1, true_prop, sample_size)
                successes = np.sum(data)
                sample_prop = successes / sample_size
                
                # Frequentist CI (Wilson score interval)
                z_crit = stats.norm.ppf(1 - (1 - credible_level)/2)
                denominator = 1 + z_crit**2/sample_size
                center = (sample_prop + z_crit**2/(2*sample_size)) / denominator
                margin = z_crit * np.sqrt((sample_prop*(1-sample_prop) + z_crit**2/(4*sample_size)) / sample_size) / denominator
                freq_lower = max(0, center - margin)
                freq_upper = min(1, center + margin)
                
                # Bayesian inference (conjugate beta prior)
                post_alpha = prior_alpha + successes
                post_beta = prior_beta + sample_size - successes
                
                # Credible interval
                bayes_lower = stats.beta.ppf((1 - credible_level)/2, post_alpha, post_beta)
                bayes_upper = stats.beta.ppf(1 - (1 - credible_level)/2, post_alpha, post_beta)
                
                # Posterior mean and mode
                post_mean = post_alpha / (post_alpha + post_beta)
                post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else (post_alpha == 1 and post_beta == 1)
                
                # Method description
                if prior_type == "Non-informative" and np.isclose(prior_alpha, 0.5) and np.isclose(prior_beta, 0.5):
                    method = "Bayesian with Jeffreys prior (Beta(0.5, 0.5))"
                elif prior_type == "Non-informative" and np.isclose(prior_alpha, 1) and np.isclose(prior_beta, 1):
                    method = "Bayesian with uniform prior (Beta(1, 1))"
                else:
                    method = f"Bayesian with {prior_type} Beta({prior_alpha}, {prior_beta}) prior"
            
            elif parameter == "Rate (Poisson)":
                # Generate Poisson data
                data = np.random.poisson(true_rate, sample_size)
                total_events = np.sum(data)
                sample_rate = total_events / sample_size
                
                # Frequentist CI
                freq_lower = stats.chi2.ppf((1 - credible_level)/2, 2*total_events) / 2 / sample_size
                freq_upper = stats.chi2.ppf(1 - (1 - credible_level)/2, 2*(total_events + 1)) / 2 / sample_size
                
                # Bayesian inference (conjugate gamma prior)
                post_alpha = prior_alpha + total_events
                post_beta = prior_beta + sample_size
                
                # Credible interval
                bayes_lower = stats.gamma.ppf((1 - credible_level)/2, post_alpha, scale=1/post_beta)
                bayes_upper = stats.gamma.ppf(1 - (1 - credible_level)/2, post_alpha, scale=1/post_beta)
                
                # Posterior mean and mode
                post_mean = post_alpha / post_beta
                post_mode = (post_alpha - 1) / post_beta if post_alpha > 1 else 0
                
                # Method description
                if prior_type == "Non-informative" and np.isclose(prior_alpha, 0.001) and np.isclose(prior_beta, 0.001):
                    method = "Bayesian with approximately improper prior (Gamma(0.001, 0.001))"
                else:
                    method = f"Bayesian with {prior_type} Gamma({prior_alpha}, {prior_beta}) prior"
            
            # Display results
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Observed Data:**")
                if parameter == "Mean (Normal)":
                    st.markdown(f"Sample Mean: {sample_mean:.4f}")
                    st.markdown(f"Sample Std Dev: {np.sqrt(sample_var):.4f}")
                elif parameter == "Proportion (Binomial)":
                    st.markdown(f"Successes: {successes} out of {sample_size}")
                    st.markdown(f"Sample Proportion: {sample_prop:.4f}")
                elif parameter == "Rate (Poisson)":
                    st.markdown(f"Total Events: {total_events}")
                    st.markdown(f"Sample Rate: {sample_rate:.4f}")
            
            with col2:
                st.markdown(f"**True Parameter:**")
                if parameter == "Mean (Normal)":
                    st.markdown(f"True Mean: {true_mean}")
                    st.markdown(f"True Std Dev: {true_sd}")
                elif parameter == "Proportion (Binomial)":
                    st.markdown(f"True Proportion: {true_prop}")
                elif parameter == "Rate (Poisson)":
                    st.markdown(f"True Rate: {true_rate}")
            
            # Create comparison table
            comparison_df = pd.DataFrame({
                'Method': ['Frequentist', 'Bayesian'],
                'Lower Bound': [freq_lower, bayes_lower],
                'Upper Bound': [freq_upper, bayes_upper],
                'Width': [freq_upper - freq_lower, bayes_upper - bayes_lower],
                'Contains True Value': [
                    freq_lower <= (true_mean if parameter == "Mean (Normal)" else 
                                  true_prop if parameter == "Proportion (Binomial)" else
                                  true_rate) <= freq_upper,
                    bayes_lower <= (true_mean if parameter == "Mean (Normal)" else 
                                   true_prop if parameter == "Proportion (Binomial)" else
                                   true_rate) <= bayes_upper
                ]
            })
            
            st.subheader(f"{credible_level*100:.0f}% Intervals")
            st.dataframe(comparison_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Width': '{:.4f}',
            }))
            
            # Create visualization
            if parameter == "Mean (Normal)":
                # Create ranges for plotting
                x_range = np.linspace(
                    min(freq_lower, bayes_lower) - 1,
                    max(freq_upper, bayes_upper) + 1,
                    1000
                )
                
                # Create figure
                fig = go.Figure()
                
                # Add posterior distribution
                if known_variance:
                    posterior_y = stats.norm.pdf(x_range, post_mean, post_sd)
                    fig.add_trace(go.Scatter(
                        x=x_range, y=posterior_y,
                        mode='lines', name='Posterior Distribution',
                        line=dict(color='blue', width=2)
                    ))
                else:
                    posterior_y = stats.t.pdf(
                        (x_range - post_mean) / post_scale, 
                        post_df
                    ) / post_scale
                    fig.add_trace(go.Scatter(
                        x=x_range, y=posterior_y,
                        mode='lines', name='Posterior Distribution',
                        line=dict(color='blue', width=2)
                    ))
                
                # Add prior distribution if informative
                if prior_type != "Non-informative":
                    prior_y = stats.norm.pdf(x_range, prior_mean, prior_sd)
                    max_prior = np.max(prior_y)
                    max_posterior = np.max(posterior_y)
                    scaling_factor = max_posterior / max_prior * 0.5
                    
                    fig.add_trace(go.Scatter(
                        x=x_range, y=prior_y * scaling_factor,
                        mode='lines', name='Prior Distribution (scaled)',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                
                # Add likelihood
                likelihood_y = stats.norm.pdf(x_range, sample_mean, np.sqrt(sample_var / sample_size))
                max_likelihood = np.max(likelihood_y)
                scaling_factor_likelihood = max_posterior / max_likelihood * 0.7
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=likelihood_y * scaling_factor_likelihood,
                    mode='lines', name='Likelihood (scaled)',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                # Add credible interval region
                credible_x = x_range[(x_range >= bayes_lower) & (x_range <= bayes_upper)]
                if known_variance:
                    credible_y = stats.norm.pdf(credible_x, post_mean, post_sd)
                else:
                    credible_y = stats.t.pdf(
                        (credible_x - post_mean) / post_scale, 
                        post_df
                    ) / post_scale
                
                fig.add_trace(go.Scatter(
                    x=credible_x, y=credible_y,
                    fill='tozeroy', mode='none',
                    name=f'{credible_level*100:.0f}% Credible Interval',
                    fillcolor='rgba(0, 0, 255, 0.2)'
                ))
                
                # Add vertical lines
                fig.add_vline(x=true_mean, line=dict(color='black', width=2, dash='dash'),
                             annotation=dict(text=f"True Mean: {true_mean}", showarrow=False))
                
                fig.add_vline(x=sample_mean, line=dict(color='red', width=2),
                             annotation=dict(text=f"Sample Mean: {sample_mean:.4f}", showarrow=False))
                
                if prior_type != "Non-informative":
                    fig.add_vline(x=prior_mean, line=dict(color='green', width=2, dash='dash'),
                                 annotation=dict(text=f"Prior Mean: {prior_mean}", showarrow=False))
                
                # Add posterior mean/mode
                fig.add_vline(x=post_mean, line=dict(color='blue', width=2),
                             annotation=dict(text=f"Posterior Mean: {post_mean:.4f}", showarrow=False))
                
                # Update layout
                fig.update_layout(
                    title=f'Bayesian Analysis for Normal Mean<br>{method}',
                    xaxis_title='Mean',
                    yaxis_title='Density',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interval comparison
                int_fig = go.Figure()
                
                # Add intervals as segments
                methods = ['Frequentist (t)', 'Bayesian']
                y_positions = [1, 2]
                lower_bounds = [freq_lower, bayes_lower]
                upper_bounds = [freq_upper, bayes_upper]
                colors = ['red', 'blue']
                
                for i, method_name in enumerate(methods):
                    int_fig.add_trace(go.Scatter(
                        x=[lower_bounds[i], upper_bounds[i]],
                        y=[y_positions[i], y_positions[i]],
                        mode='lines',
                        name=method_name,
                        line=dict(color=colors[i], width=4)
                    ))
                    
                    # Add endpoints as markers
                    int_fig.add_trace(go.Scatter(
                        x=[lower_bounds[i], upper_bounds[i]],
                        y=[y_positions[i], y_positions[i]],
                        mode='markers',
                        showlegend=False,
                        marker=dict(color=colors[i], size=8)
                    ))
                    
                    # Add labels for bounds
                    int_fig.add_annotation(
                        x=lower_bounds[i] - 0.1,
                        y=y_positions[i],
                        text=f"{lower_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="right"
                    )
                    
                    int_fig.add_annotation(
                        x=upper_bounds[i] + 0.1,
                        y=y_positions[i],
                        text=f"{upper_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="left"
                    )
                
                # Add vertical line for true value
                int_fig.add_vline(x=true_mean, line=dict(color='black', width=2, dash='dash'),
                                 annotation=dict(text=f"True Mean: {true_mean}", showarrow=False))
                
                # Update layout
                int_fig.update_layout(
                    title=f'Comparison of {credible_level*100:.0f}% Intervals',
                    xaxis_title='Mean',
                    yaxis=dict(
                        tickmode='array',
                        tickvals=y_positions,
                        ticktext=methods,
                        showgrid=False
                    ),
                    height=300
                )
                
                st.plotly_chart(int_fig, use_container_width=True)
            
            elif parameter == "Proportion (Binomial)":
                # Create ranges for plotting
                x_range = np.linspace(0, 1, 1000)
                
                # Create figure
                fig = go.Figure()
                
                # Add posterior distribution
                posterior_y = stats.beta.pdf(x_range, post_alpha, post_beta)
                fig.add_trace(go.Scatter(
                    x=x_range, y=posterior_y,
                    mode='lines', name='Posterior Distribution',
                    line=dict(color='blue', width=2)
                ))
                
                # Add prior distribution
                prior_y = stats.beta.pdf(x_range, prior_alpha, prior_beta)
                max_prior = np.max(prior_y)
                max_posterior = np.max(posterior_y)
                scaling_factor = max_posterior / max_prior * 0.5 if max_prior > 0 else 1
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=prior_y * scaling_factor,
                    mode='lines', name='Prior Distribution (scaled)',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                # Add likelihood (binomial probability mass scaled to be comparable)
                likelihood = np.zeros_like(x_range)
                for i, p in enumerate(x_range):
                    likelihood[i] = stats.binom.pmf(successes, sample_size, p)
                max_likelihood = np.max(likelihood)
                scaling_factor_likelihood = max_posterior / max_likelihood if max_likelihood > 0 else 1
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=likelihood * scaling_factor_likelihood,
                    mode='lines', name='Likelihood (scaled)',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                # Add credible interval region
                credible_x = x_range[(x_range >= bayes_lower) & (x_range <= bayes_upper)]
                credible_y = stats.beta.pdf(credible_x, post_alpha, post_beta)
                
                fig.add_trace(go.Scatter(
                    x=credible_x, y=credible_y,
                    fill='tozeroy', mode='none',
                    name=f'{credible_level*100:.0f}% Credible Interval',
                    fillcolor='rgba(0, 0, 255, 0.2)'
                ))
                
                # Add vertical lines
                fig.add_vline(x=true_prop, line=dict(color='black', width=2, dash='dash'),
                             annotation=dict(text=f"True Prop: {true_prop}", showarrow=False))
                
                fig.add_vline(x=sample_prop, line=dict(color='red', width=2),
                             annotation=dict(text=f"Sample Prop: {sample_prop:.4f}", showarrow=False))
                
                fig.add_vline(x=post_mean, line=dict(color='blue', width=2),
                             annotation=dict(text=f"Posterior Mean: {post_mean:.4f}", showarrow=False))
                
                if post_alpha > 1 and post_beta > 1:
                    fig.add_vline(x=post_mode, line=dict(color='purple', width=2, dash='dot'),
                                 annotation=dict(text=f"Posterior Mode: {post_mode:.4f}", showarrow=False))
                
                # Update layout
                fig.update_layout(
                    title=f'Bayesian Analysis for Binomial Proportion<br>{method}',
                    xaxis_title='Proportion',
                    yaxis_title='Density',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interval comparison
                int_fig = go.Figure()
                
                # Add intervals as segments
                methods = ['Frequentist (Wilson)', 'Bayesian']
                y_positions = [1, 2]
                lower_bounds = [freq_lower, bayes_lower]
                upper_bounds = [freq_upper, bayes_upper]
                colors = ['red', 'blue']
                
                for i, method_name in enumerate(methods):
                    int_fig.add_trace(go.Scatter(
                        x=[lower_bounds[i], upper_bounds[i]],
                        y=[y_positions[i], y_positions[i]],
                        mode='lines',
                        name=method_name,
                        line=dict(color=colors[i], width=4)
                    ))
                    
                    # Add endpoints as markers
                    int_fig.add_trace(go.Scatter(
                        x=[lower_bounds[i], upper_bounds[i]],
                        y=[y_positions[i], y_positions[i]],
                        mode='markers',
                        showlegend=False,
                        marker=dict(color=colors[i], size=8)
                    ))
                    
                    # Add labels for bounds
                    int_fig.add_annotation(
                        x=lower_bounds[i] - 0.02,
                        y=y_positions[i],
                        text=f"{lower_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="right"
                    )
                    
                    int_fig.add_annotation(
                        x=upper_bounds[i] + 0.02,
                        y=y_positions[i],
                        text=f"{upper_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="left"
                    )
                
                # Add vertical line for true value
                int_fig.add_vline(x=true_prop, line=dict(color='black', width=2, dash='dash'),
                                 annotation=dict(text=f"True Proportion: {true_prop}", showarrow=False))
                
                # Update layout
                int_fig.update_layout(
                    title=f'Comparison of {credible_level*100:.0f}% Intervals',
                    xaxis_title='Proportion',
                    yaxis=dict(
                        tickmode='array',
                        tickvals=y_positions,
                        ticktext=methods,
                        showgrid=False
                    ),
                    height=300,
                    xaxis=dict(range=[-0.1, 1.1])
                )
                
                st.plotly_chart(int_fig, use_container_width=True)
            
            elif parameter == "Rate (Poisson)":
                # Create ranges for plotting
                x_max = max(bayes_upper, freq_upper) * 1.5
                x_range = np.linspace(0, x_max, 1000)
                
                # Create figure
                fig = go.Figure()
                
                # Add posterior distribution
                posterior_y = stats.gamma.pdf(x_range, post_alpha, scale=1/post_beta)
                fig.add_trace(go.Scatter(
                    x=x_range, y=posterior_y,
                    mode='lines', name='Posterior Distribution',
                    line=dict(color='blue', width=2)
                ))
                
                # Add prior distribution if not improper
                if not (prior_type == "Non-informative" and np.isclose(prior_alpha, 0.001) and np.isclose(prior_beta, 0.001)):
                    prior_y = stats.gamma.pdf(x_range, prior_alpha, scale=1/prior_beta)
                    max_prior = np.max(prior_y)
                    max_posterior = np.max(posterior_y)
                    scaling_factor = max_posterior / max_prior * 0.5 if max_prior > 0 else 1
                    
                    fig.add_trace(go.Scatter(
                        x=x_range, y=prior_y * scaling_factor,
                        mode='lines', name='Prior Distribution (scaled)',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                
                # Add likelihood (poisson probability mass scaled to be comparable)
                likelihood = np.zeros_like(x_range)
                for i, rate in enumerate(x_range):
                    # Scale by sample size since we're modeling the rate
                    effective_lambda = rate * sample_size
                    if effective_lambda < 700:  # Avoid overflow
                        likelihood[i] = stats.poisson.pmf(total_events, effective_lambda)
                max_likelihood = np.max(likelihood)
                scaling_factor_likelihood = max_posterior / max_likelihood if max_likelihood > 0 else 1
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=likelihood * scaling_factor_likelihood,
                    mode='lines', name='Likelihood (scaled)',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                # Add credible interval region
                credible_x = x_range[(x_range >= bayes_lower) & (x_range <= bayes_upper)]
                credible_y = stats.gamma.pdf(credible_x, post_alpha, scale=1/post_beta)
                
                fig.add_trace(go.Scatter(
                    x=credible_x, y=credible_y,
                    fill='tozeroy', mode='none',
                    name=f'{credible_level*100:.0f}% Credible Interval',
                    fillcolor='rgba(0, 0, 255, 0.2)'
                ))
                
                # Add vertical lines
                fig.add_vline(x=true_rate, line=dict(color='black', width=2, dash='dash'),
                             annotation=dict(text=f"True Rate: {true_rate}", showarrow=False))
                
                fig.add_vline(x=sample_rate, line=dict(color='red', width=2),
                             annotation=dict(text=f"Sample Rate: {sample_rate:.4f}", showarrow=False))
                
                fig.add_vline(x=post_mean, line=dict(color='blue', width=2),
                             annotation=dict(text=f"Posterior Mean: {post_mean:.4f}", showarrow=False))
                
                if post_alpha > 1:
                    fig.add_vline(x=post_mode, line=dict(color='purple', width=2, dash='dot'),
                                 annotation=dict(text=f"Posterior Mode: {post_mode:.4f}", showarrow=False))
                
                # Update layout
                fig.update_layout(
                    title=f'Bayesian Analysis for Poisson Rate<br>{method}',
                    xaxis_title='Rate',
                    yaxis_title='Density',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interval comparison
                int_fig = go.Figure()
                
                # Add intervals as segments
                methods = ['Frequentist', 'Bayesian']
                y_positions = [1, 2]
                lower_bounds = [freq_lower, bayes_lower]
                upper_bounds = [freq_upper, bayes_upper]
                colors = ['red', 'blue']
                
                for i, method_name in enumerate(methods):
                    int_fig.add_trace(go.Scatter(
                        x=[lower_bounds[i], upper_bounds[i]],
                        y=[y_positions[i], y_positions[i]],
                        mode='lines',
                        name=method_name,
                        line=dict(color=colors[i], width=4)
                    ))
                    
                    # Add endpoints as markers
                    int_fig.add_trace(go.Scatter(
                        x=[lower_bounds[i], upper_bounds[i]],
                        y=[y_positions[i], y_positions[i]],
                        mode='markers',
                        showlegend=False,
                        marker=dict(color=colors[i], size=8)
                    ))
                    
                    # Add labels for bounds
                    int_fig.add_annotation(
                        x=lower_bounds[i] - 0.2,
                        y=y_positions[i],
                        text=f"{lower_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="right"
                    )
                    
                    int_fig.add_annotation(
                        x=upper_bounds[i] + 0.2,
                        y=y_positions[i],
                        text=f"{upper_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="left"
                    )
                
                # Add vertical line for true value
                int_fig.add_vline(x=true_rate, line=dict(color='black', width=2, dash='dash'),
                                 annotation=dict(text=f"True Rate: {true_rate}", showarrow=False))
                
                # Update layout
                int_fig.update_layout(
                    title=f'Comparison of {credible_level*100:.0f}% Intervals',
                    xaxis_title='Rate',
                    yaxis=dict(
                        tickmode='array',
                        tickvals=y_positions,
                        ticktext=methods,
                        showgrid=False
                    ),
                    height=300
                )
                
                st.plotly_chart(int_fig, use_container_width=True)
            
            # Add interpretation
            st.subheader("Interpretation")
            
            st.markdown(f"""
            ### Bayesian vs. Frequentist Intervals
            
            **Key differences in interpretation:**
            
            - **Frequentist {credible_level*100:.0f}% confidence interval**: 
              If we repeated the experiment many times, {credible_level*100:.0f}% of such intervals would contain the true parameter.
            
            - **Bayesian {credible_level*100:.0f}% credible interval**: 
              Given the observed data and our prior beliefs, there is a {credible_level*100:.0f}% probability that the true parameter lies within this interval.
            
            **Results for this example:**
            
            For the {parameter} with a {prior_type} prior:
            
            - Frequentist interval: [{freq_lower:.4f}, {freq_upper:.4f}]
              - Width: {freq_upper - freq_lower:.4f}
              - {'Contains' if freq_lower <= (true_mean if parameter == "Mean (Normal)" else true_prop if parameter == "Proportion (Binomial)" else true_rate) <= freq_upper else 'Does not contain'} the true parameter
            
            - Bayesian interval: [{bayes_lower:.4f}, {bayes_upper:.4f}]
              - Width: {bayes_upper - bayes_lower:.4f}
              - {'Contains' if bayes_lower <= (true_mean if parameter == "Mean (Normal)" else true_prop if parameter == "Proportion (Binomial)" else true_rate) <= bayes_upper else 'Does not contain'} the true parameter
            """)
            
            # Parameter-specific interpretation
            if parameter == "Mean (Normal)":
                st.markdown(f"""
                **Insights for Normal Mean:**
                
                The Bayesian approach allows us to incorporate prior information about the mean, which {'significantly' if prior_type == 'Informative' else 'slightly'} affects the posterior.
                
                - Prior influence: {'Strong' if prior_type == 'Informative' else 'Moderate' if prior_type == 'Weakly informative' else 'Minimal'}
                - Posterior mean: {post_mean:.4f} (which is {'closer to the prior' if abs(post_mean - prior_mean) < abs(post_mean - sample_mean) and prior_type != 'Non-informative' else 'closer to the sample mean'})
                
                As sample size increases, the influence of the prior diminishes, and the Bayesian interval approaches the frequentist interval.
                """)
                
                if prior_type != "Non-informative":
                    if sample_size < 10:
                        st.markdown("""
                        With this small sample size, the prior has a substantial influence on the posterior.
                        """)
                    elif sample_size < 30:
                        st.markdown("""
                        With this moderate sample size, the prior still noticeably influences the posterior, but the data plays a major role.
                        """)
                    else:
                        st.markdown("""
                        With this large sample size, the data dominates the posterior, and the prior's influence is less pronounced.
                        """)
            
            elif parameter == "Proportion (Binomial)":
                st.markdown(f"""
                **Insights for Binomial Proportion:**
                
                The Beta prior is conjugate to the Binomial likelihood, resulting in a Beta posterior with parameters:
                - α = prior_alpha + successes = {prior_alpha} + {successes} = {post_alpha}
                - β = prior_beta + failures = {prior_beta} + {sample_size - successes} = {post_beta}
                
                The posterior mean is α/(α+β) = {post_alpha}/{post_alpha + post_beta} = {post_mean:.4f}
                
                {'The Wilson score interval (frequentist) and Beta posterior interval (Bayesian) are both designed to handle proportions near 0 or 1 better than the standard Wald interval.' if sample_prop < 0.1 or sample_prop > 0.9 else ''}
                """)
                
                if prior_type == "Non-informative" and np.isclose(prior_alpha, 0.5) and np.isclose(prior_beta, 0.5):
                    st.markdown("""
                    The Jeffreys prior (Beta(0.5, 0.5)) is a common non-informative prior for proportions, offering good frequency properties. 
                    It's uniform in the variance-stabilized space and places more weight near 0 and 1.
                    """)
                elif prior_type == "Non-informative" and np.isclose(prior_alpha, 1) and np.isclose(prior_beta, 1):
                    st.markdown("""
                    The uniform prior (Beta(1, 1)) represents equal probability for all proportion values between 0 and 1.
                    With this prior, the posterior mode equals the sample proportion.
                    """)
            
            elif parameter == "Rate (Poisson)":
                st.markdown(f"""
                **Insights for Poisson Rate:**
                
                The Gamma prior is conjugate to the Poisson likelihood, resulting in a Gamma posterior with parameters:
                - α = prior_alpha + total_events = {prior_alpha} + {total_events} = {post_alpha}
                - β = prior_beta + sample_size = {prior_beta} + {sample_size} = {post_beta}
                
                The posterior mean is α/β = {post_alpha}/{post_beta} = {post_mean:.4f}
                
                The credible interval automatically respects the constraint that rates must be non-negative.
                """)
                
                if prior_type == "Non-informative" and np.isclose(prior_alpha, 0.001) and np.isclose(prior_beta, 0.001):
                    st.markdown("""
                    The approximately improper Gamma(0.001, 0.001) prior is nearly flat on (0, ∞) and has minimal influence on the posterior.
                    """)
            
            # General Bayesian vs Frequentist comparison
            st.markdown("""
            ### When to use Bayesian Credible Intervals
            
            **Advantages of Bayesian intervals:**
            
            1. **Intuitive interpretation**: Direct probability statements about parameters
            2. **Incorporation of prior knowledge**: Valuable when reliable prior information exists
            3. **Small sample performance**: Often better than frequentist methods with small samples
            4. **Respects parameter constraints**: Naturally handles bounded parameters (e.g., variances ≥ 0)
            5. **Uncertainty about uncertainty**: Can account for uncertainty in nuisance parameters
            
            **Considerations:**
            
            1. **Prior sensitivity**: Results can be influenced by prior choice
            2. **Computational complexity**: May require MCMC or other simulation methods for complex problems
            3. **Subjectivity concerns**: Choice of prior can be controversial in some contexts
            
            **Practical recommendation:**
            
            Consider using Bayesian credible intervals when:
            - You have meaningful prior information
            - You need direct probability statements about parameters
            - You're working with small samples
            - The parameter has natural constraints
            - You want to average over uncertainty in nuisance parameters
            """)
    
    elif method_type == "Multiple Testing Adjustment":
        st.subheader("Confidence Intervals with Multiple Testing Adjustment")
        
        st.markdown("""
        When constructing multiple confidence intervals simultaneously, the probability that at least one interval fails to contain its true parameter increases with the number of intervals. This is analogous to the multiple testing problem in hypothesis testing.
        
        Multiple testing adjustments for confidence intervals ensure that the family-wise coverage probability meets the desired level.
        """)
        
        # Options for multiple testing simulation
        col1, col2 = st.columns(2)
        
        with col1:
            n_parameters = st.slider("Number of parameters", 2, 20, 10)
            conf_level = st.slider("Desired family-wise confidence level", 0.80, 0.99, 0.95, 0.01)
            sample_size = st.slider("Sample size per parameter", 5, 100, 30)
        
        with col2:
            adjustment_method = st.radio(
                "Adjustment method",
                ["Bonferroni", "Šidák", "Simultaneous Confidence Regions"]
            )
            simulation_type = st.radio(
                "Simulation type",
                ["Independent Means", "Regression Coefficients"]
            )
        
        # Additional options based on simulation type
        if simulation_type == "Independent Means":
            effect_size = st.slider("Effect size", 0.0, 2.0, 0.5, 0.1)
        elif simulation_type == "Regression Coefficients":
            x_correlation = st.slider("Predictor correlation", -0.9, 0.9, 0.0, 0.1)
        
        if st.button("Run Multiple Confidence Intervals Simulation", key="run_multiple_ci"):
            # Generate data
            np.random.seed(None)
            
            if simulation_type == "Independent Means":
                # Generate data for multiple independent means
                true_means = np.zeros(n_parameters)
                true_means[0] = effect_size  # Only first parameter has an effect
                
                # Generate samples
                samples = []
                for i in range(n_parameters):
                    samples.append(np.random.normal(true_means[i], 1.0, sample_size))
                
                # Compute sample statistics
                sample_means = [np.mean(sample) for sample in samples]
                sample_sds = [np.std(sample, ddof=1) for sample in samples]
                
                # Standard (unadjusted) confidence intervals
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                margins = [t_crit * sd / np.sqrt(sample_size) for sd in sample_sds]
                
                unadjusted_lower = [mean - margin for mean, margin in zip(sample_means, margins)]
                unadjusted_upper = [mean + margin for mean, margin in zip(sample_means, margins)]
                
                # Adjusted confidence intervals
                if adjustment_method == "Bonferroni":
                    # Bonferroni adjustment
                    t_crit_adj = stats.t.ppf(1 - (1 - conf_level)/(2 * n_parameters), sample_size - 1)
                    margins_adj = [t_crit_adj * sd / np.sqrt(sample_size) for sd in sample_sds]
                    
                    adjusted_lower = [mean - margin for mean, margin in zip(sample_means, margins_adj)]
                    adjusted_upper = [mean + margin for mean, margin in zip(sample_means, margins_adj)]
                    
                    method_name = "Bonferroni"
                    
                elif adjustment_method == "Šidák":
                    # Šidák adjustment
                    alpha_adj = 1 - (1 - (1 - conf_level))**(1/n_parameters)
                    t_crit_adj = stats.t.ppf(1 - alpha_adj/2, sample_size - 1)
                    margins_adj = [t_crit_adj * sd / np.sqrt(sample_size) for sd in sample_sds]
                    
                    adjusted_lower = [mean - margin for mean, margin in zip(sample_means, margins_adj)]
                    adjusted_upper = [mean + margin for mean, margin in zip(sample_means, margins_adj)]
                    
                    method_name = "Šidák"
                    
                elif adjustment_method == "Simultaneous Confidence Regions":
                    # Multivariate t-distribution approach (Hotelling's T²)
                    # This is an approximation for independent means
                    f_crit = stats.f.ppf(conf_level, n_parameters, sample_size - n_parameters)
                    t_crit_adj = np.sqrt(n_parameters * f_crit)
                    margins_adj = [t_crit_adj * sd / np.sqrt(sample_size) for sd in sample_sds]
                    
                    adjusted_lower = [mean - margin for mean, margin in zip(sample_means, margins_adj)]
                    adjusted_upper = [mean + margin for mean, margin in zip(sample_means, margins_adj)]
                    
                    method_name = "Hotelling's T² (approx.)"
                
                # Create results
                results = []
                for i in range(n_parameters):
                    contains_unadj = unadjusted_lower[i] <= true_means[i] <= unadjusted_upper[i]
                    contains_adj = adjusted_lower[i] <= true_means[i] <= adjusted_upper[i]
                    
                    results.append({
                        'Parameter': f'μ{i+1}',
                        'True Value': true_means[i],
                        'Estimate': sample_means[i],
                        'Unadjusted Lower': unadjusted_lower[i],
                        'Unadjusted Upper': unadjusted_upper[i],
                        'Adjusted Lower': adjusted_lower[i],
                        'Adjusted Upper': adjusted_upper[i],
                        'Unadjusted Contains': contains_unadj,
                        'Adjusted Contains': contains_adj
                    })
                
                results_df = pd.DataFrame(results)
            
            elif simulation_type == "Regression Coefficients":
                # Generate correlated predictors for multiple regression
                n_obs = sample_size
                
                # Generate correlation matrix with specified correlation
                cor_matrix = np.ones((n_parameters, n_parameters)) * x_correlation
                np.fill_diagonal(cor_matrix, 1.0)
                
                # Convert correlation to covariance matrix (unit variance)
                cov_matrix = cor_matrix.copy()
                
                # Generate multivariate normal predictors
                X = np.random.multivariate_normal(np.zeros(n_parameters), cov_matrix, n_obs)
                
                # True coefficients (only first has effect)
                true_betas = np.zeros(n_parameters)
                true_betas[0] = effect_size
                
                # Generate response
                y = X @ true_betas + np.random.normal(0, 1, n_obs)
                
                # Fit linear regression
                X_with_intercept = np.column_stack([np.ones(n_obs), X])
                beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
                
                # Compute residuals
                residuals = y - X_with_intercept @ beta_hat
                residual_var = np.sum(residuals**2) / (n_obs - n_parameters - 1)
                
                # Compute standard errors
                cov_beta = residual_var * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                se_beta = np.sqrt(np.diag(cov_beta))
                
                # Extract coefficients (excluding intercept)
                sample_betas = beta_hat[1:]
                se_betas = se_beta[1:]
                
                # Standard (unadjusted) confidence intervals
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, n_obs - n_parameters - 1)
                margins = t_crit * se_betas
                
                unadjusted_lower = sample_betas - margins
                unadjusted_upper = sample_betas + margins
                
                # Adjusted confidence intervals
                if adjustment_method == "Bonferroni":
                    # Bonferroni adjustment
                    t_crit_adj = stats.t.ppf(1 - (1 - conf_level)/(2 * n_parameters), n_obs - n_parameters - 1)
                    margins_adj = t_crit_adj * se_betas
                    
                    adjusted_lower = sample_betas - margins_adj
                    adjusted_upper = sample_betas + margins_adj
                    
                    method_name = "Bonferroni"
                    
                elif adjustment_method == "Šidák":
                    # Šidák adjustment
                    alpha_adj = 1 - (1 - (1 - conf_level))**(1/n_parameters)
                    t_crit_adj = stats.t.ppf(1 - alpha_adj/2, n_obs - n_parameters - 1)
                    margins_adj = t_crit_adj * se_betas
                    
                    adjusted_lower = sample_betas - margins_adj
                    adjusted_upper = sample_betas + margins_adj
                    
                    method_name = "Šidák"
                    
                elif adjustment_method == "Simultaneous Confidence Regions":
                    # Working-Hotelling band
                    f_crit = stats.f.ppf(conf_level, n_parameters, n_obs - n_parameters - 1)
                    mult_factor = np.sqrt(n_parameters * f_crit)
                    margins_adj = mult_factor * se_betas
                    
                    adjusted_lower = sample_betas - margins_adj
                    adjusted_upper = sample_betas + margins_adj
                    
                    method_name = "Working-Hotelling"
                
                # Create results
                results = []
                for i in range(n_parameters):
                    contains_unadj = unadjusted_lower[i] <= true_betas[i] <= unadjusted_upper[i]
                    contains_adj = adjusted_lower[i] <= true_betas[i] <= adjusted_upper[i]
                    
                    results.append({
                        'Parameter': f'β{i+1}',
                        'True Value': true_betas[i],
                        'Estimate': sample_betas[i],
                        'Unadjusted Lower': unadjusted_lower[i],
                        'Unadjusted Upper': unadjusted_upper[i],
                        'Adjusted Lower': adjusted_lower[i],
                        'Adjusted Upper': adjusted_upper[i],
                        'Unadjusted Contains': contains_unadj,
                        'Adjusted Contains': contains_adj
                    })
                
                results_df = pd.DataFrame(results)
            
            # Display results
            st.subheader("Results")
            
            # Format the results dataframe
            formatted_results = results_df.style.format({
                'True Value': '{:.4f}',
                'Estimate': '{:.4f}',
                'Unadjusted Lower': '{:.4f}',
                'Unadjusted Upper': '{:.4f}',
                'Adjusted Lower': '{:.4f}',
                'Adjusted Upper': '{:.4f}'
            }).apply(lambda row: [
                'background-color: rgba(144, 238, 144, 0.5)' if cell else 'background-color: rgba(255, 182, 193, 0.5)'
                for cell in [row['Unadjusted Contains'], row['Adjusted Contains']]
            ], axis=1, subset=['Unadjusted Contains', 'Adjusted Contains'])
            
            st.dataframe(formatted_results)
            
            # Create visualization
            fig = go.Figure()
            
            # Add confidence intervals
            parameters = results_df['Parameter'].tolist()
            estimates = results_df['Estimate'].tolist()
            true_values = results_df['True Value'].tolist()
            
            # Add unadjusted intervals
            for i, param in enumerate(parameters):
                fig.add_trace(go.Scatter(
                    x=[results_df.loc[i, 'Unadjusted Lower'], results_df.loc[i, 'Unadjusted Upper']],
                    y=[i, i],
                    mode='lines',
                    name='Unadjusted CI' if i == 0 else None,
                    showlegend=(i == 0),
                    line=dict(color='blue', width=2)
                ))
            
            # Add adjusted intervals
            for i, param in enumerate(parameters):
                fig.add_trace(go.Scatter(
                    x=[results_df.loc[i, 'Adjusted Lower'], results_df.loc[i, 'Adjusted Upper']],
                    y=[i, i],
                    mode='lines',
                    name=f'{method_name} Adjusted CI' if i == 0 else None,
                    showlegend=(i == 0),
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            # Add point estimates
            fig.add_trace(go.Scatter(
                x=estimates,
                y=list(range(len(parameters))),
                mode='markers',
                name='Estimates',
                marker=dict(color='black', size=8)
            ))
            
            # Add true values
            fig.add_trace(go.Scatter(
                x=true_values,
                y=list(range(len(parameters))),
                mode='markers',
                name='True Values',
                marker=dict(color='green', size=8, symbol='diamond')
            ))
            
            # Add vertical line at zero
            fig.add_vline(x=0, line=dict(color='gray', width=1, dash='dot'))
            
            # Update layout
            fig.update_layout(
                title=f'Comparison of {conf_level*100:.0f}% Confidence Intervals<br>'
                      f'(Standard vs. {method_name} Adjusted)',
                xaxis_title='Parameter Value',
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(parameters))),
                    ticktext=parameters
                ),
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate familywise coverage
            all_unadjusted_contain = results_df['Unadjusted Contains'].all()
            all_adjusted_contain = results_df['Adjusted Contains'].all()
            
            any_unadjusted_misses = not all_unadjusted_contain
            any_adjusted_misses = not all_adjusted_contain
            
            # Compute average width
            unadjusted_widths = results_df['Unadjusted Upper'] - results_df['Unadjusted Lower']
            adjusted_widths = results_df['Adjusted Upper'] - results_df['Adjusted Lower']
            
            avg_unadjusted_width = unadjusted_widths.mean()
            avg_adjusted_width = adjusted_widths.mean()
            width_increase = (avg_adjusted_width / avg_unadjusted_width - 1) * 100
            
            # Display summary metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="All Unadjusted CIs Contain True Values", 
                    value="Yes" if all_unadjusted_contain else "No"
                )
                
                st.metric(
                    label="Average Unadjusted Width", 
                    value=f"{avg_unadjusted_width:.4f}"
                )
            
            with col2:
                st.metric(
                    label=f"All {method_name} Adjusted CIs Contain True Values", 
                    value="Yes" if all_adjusted_contain else "No"
                )
                
                st.metric(
                    label=f"Average {method_name} Adjusted Width", 
                    value=f"{avg_adjusted_width:.4f}",
                    delta=f"{width_increase:.1f}%"
                )
            
            # Add interpretation
            st.subheader("Interpretation")
            
            alpha_individual = 1 - conf_level
            
            if adjustment_method == "Bonferroni":
                alpha_adjusted = alpha_individual / n_parameters
                adjusted_conf = 1 - alpha_adjusted
                
                st.markdown(f"""
                ### Bonferroni Adjustment
                
                The Bonferroni adjustment controls the family-wise error rate (FWER) by dividing the significance level (α) by the number of comparisons.
                
                **Mathematical details:**
                - Individual confidence level: {conf_level*100:.1f}% (α = {alpha_individual:.4f})
                - Adjusted confidence level per interval: {adjusted_conf*100:.4f}% (α/m = {alpha_adjusted:.4f})
                - Number of parameters (m): {n_parameters}
                
                **Width impact:**
                - Unadjusted CIs are {avg_unadjusted_width:.4f} wide on average
                - Bonferroni-adjusted CIs are {avg_adjusted_width:.4f} wide on average
                - This represents a {width_increase:.1f}% increase in width
                
                **Conservativeness:**
                The Bonferroni adjustment is generally conservative (more so as the number of parameters increases). It guarantees that the probability of at least one false confidence statement is at most α = {alpha_individual:.4f}.
                """)
                
            elif adjustment_method == "Šidák":
                alpha_adjusted = 1 - (1 - alpha_individual)**(1/n_parameters)
                adjusted_conf = 1 - alpha_adjusted
                
                st.markdown(f"""
                ### Šidák Adjustment
                
                The Šidák adjustment controls the family-wise error rate (FWER) and is slightly less conservative than Bonferroni when the tests are independent.
                
                **Mathematical details:**
                - Individual confidence level: {conf_level*100:.1f}% (α = {alpha_individual:.4f})
                - Adjusted confidence level per interval: {adjusted_conf*100:.4f}% (1-(1-α)^(1/m) = {alpha_adjusted:.4f})
                - Number of parameters (m): {n_parameters}
                
                **Width impact:**
                - Unadjusted CIs are {avg_unadjusted_width:.4f} wide on average
                - Šidák-adjusted CIs are {avg_adjusted_width:.4f} wide on average
                - This represents a {width_increase:.1f}% increase in width
                
                **Independence assumption:**
                The Šidák correction assumes independence between tests. In the {'regression context' if simulation_type == 'Regression Coefficients' else 'multiple means context'}, this assumption is {'violated due to predictor correlations' if simulation_type == 'Regression Coefficients' and abs(x_correlation) > 0.1 else 'generally satisfied'}.
                """)
                
            elif adjustment_method == "Simultaneous Confidence Regions":
                if simulation_type == "Independent Means":
                    st.markdown(f"""
                    ### Hotelling's T² Approximation
                    
                    This approach constructs confidence intervals based on the multivariate t-distribution, which accounts for the joint distribution of the parameter estimates.
                    
                    **Mathematical details:**
                    - Individual confidence level: {conf_level*100:.1f}%
                    - For independent means, this uses an approximation based on the F-distribution
                    - Critical value multiplier: {t_crit_adj:.4f} (compared to {t_crit:.4f} for unadjusted intervals)
                    
                    **Width impact:**
                    - Unadjusted CIs are {avg_unadjusted_width:.4f} wide on average
                    - Hotelling-adjusted CIs are {avg_adjusted_width:.4f} wide on average
                    - This represents a {width_increase:.1f}% increase in width
                    
                    **Benefits:**
                    This approach can be less conservative than Bonferroni or Šidák when parameter estimates are correlated.
                    """)
                else:  # Regression Coefficients
                    st.markdown(f"""
                    ### Working-Hotelling Confidence Bands
                    
                    This approach constructs simultaneous confidence bands for regression coefficients, taking into account their correlation structure.
                    
                    **Mathematical details:**
                    - Uses the F-distribution to determine the critical value
                    - Multiplier: sqrt({n_parameters} * F_{{{n_parameters},{n_obs - n_parameters - 1},1-α}}) = {mult_factor:.4f}
                    - This accounts for the correlation among regression coefficient estimates
                    
                    **Width impact:**
                    - Unadjusted CIs are {avg_unadjusted_width:.4f} wide on average
                    - Working-Hotelling CIs are {avg_adjusted_width:.4f} wide on average
                    - This represents a {width_increase:.1f}% increase in width
                    
                    **Benefits:**
                    For correlated predictors (correlation = {x_correlation}), this approach properly accounts for the joint distribution of the coefficients.
                    """)
            
            # Add general guidance
            st.markdown("""
            ### When to Adjust for Multiple Comparisons
            
            **Need for adjustment:**
            
            When conducting multiple significance tests or constructing multiple confidence intervals, the probability of at least one false conclusion increases with the number of tests. This is known as the "multiple testing problem."
            
            **Key considerations:**
            
            1. **Family-wise Error Rate (FWER)**: Probability of making at least one Type I error across all tests
            2. **False Discovery Rate (FDR)**: Expected proportion of false discoveries among all discoveries
            3. **Trade-off**: Controlling for multiple comparisons reduces false positives but increases false negatives
            
            **When adjustments are important:**
            
            - When the cost of a false positive is high
            - When you need strong control of error rates
            - When making conclusive statements about multiple parameters simultaneously
            - In confirmatory (rather than exploratory) analyses
            
            **Methods to consider:**
            
            - **Bonferroni**: Simple and widely used, but conservative
            - **Šidák**: Slightly less conservative than Bonferroni when tests are independent
            - **Holm-Bonferroni**: Sequential approach that is more powerful than standard Bonferroni
            - **Simultaneous confidence regions**: Accounts for correlation between parameters
            
            For large numbers of parameters, consider methods that control the False Discovery Rate (FDR) instead, such as the Benjamini-Hochberg procedure.
            """)
    
    elif method_type == "Profile Likelihood Intervals":
        st.subheader("Profile Likelihood Confidence Intervals")
        
        st.markdown("""
        Profile likelihood confidence intervals are useful for parameters in complex models where standard Wald-type intervals may be inappropriate. They are based on the likelihood ratio test and often have better coverage properties, especially for non-linear models or small samples.
        """)
        
        # Options for profile likelihood simulation
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.radio(
                "Model type",
                ["Logistic Regression", "Exponential Decay", "Weibull Survival"]
            )
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
        
        with col2:
            sample_size = st.slider("Sample size", 10, 200, 50)
            param_value = st.slider("True parameter value", 0.1, 5.0, 1.0, 0.1)
        
        # Additional options based on model type
        if model_type == "Logistic Regression":
            intercept = st.slider("Intercept (β₀)", -5.0, 5.0, 0.0, 0.5)
            add_quadratic = st.checkbox("Add quadratic term", value=False)
        
        elif model_type == "Exponential Decay":
            noise_level = st.slider("Noise level", 0.01, 1.0, 0.1, 0.01)
        
        elif model_type == "Weibull Survival":
            censoring_rate = st.slider("Censoring rate", 0.0, 0.8, 0.3, 0.05)
            shape_param = st.slider("Shape parameter", 0.5, 5.0, 1.5, 0.1)
        
        if st.button("Generate Profile Likelihood Intervals", key="gen_profile"):
            # Define custom functions for profile likelihood calculation
            def negative_log_likelihood(params, model_func, x, y):
                """Generic negative log-likelihood function"""
                y_pred = model_func(x, params)
                return -np.sum(stats.norm.logpdf(y, loc=y_pred, scale=1.0))
            
            def profile_likelihood_interval(x, y, model_func, full_params, param_idx, param_name, conf_level):
                """Calculate profile likelihood interval for a specific parameter"""
                # Full model fit
                result = minimize(negative_log_likelihood, full_params, args=(model_func, x, y))
                mle_params = result.x
                min_nll = result.fun
                
                # Critical value for likelihood ratio test
                crit_value = stats.chi2.ppf(conf_level, 1) / 2
                
                # Function to optimize for profile likelihood
                def profile_obj(param_value):
                    # Create a copy of the MLE parameters and update the parameter of interest
                    params = mle_params.copy()
                    params[param_idx] = param_value
                    
                    # Define a function to optimize the remaining parameters
                    def conditional_nll(other_params):
                        full_params = np.concatenate([
                            other_params[:param_idx], 
                            [param_value], 
                            other_params[param_idx:]
                        ])
                        return negative_log_likelihood(full_params, model_func, x, y)
                    
                    # Optimize the remaining parameters
                    other_params = np.concatenate([params[:param_idx], params[param_idx+1:]])
                    result = minimize(conditional_nll, other_params)
                    
                    # Return the difference between the profile likelihood and the critical value
                    return result.fun - min_nll - crit_value
                
                # Find the lower bound
                lower_result = minimize_scalar(
                    lambda v: np.abs(profile_obj(v)), 
                    bounds=(mle_params[param_idx] * 0.1, mle_params[param_idx]),
                    method='bounded'
                )
                lower_bound = lower_result.x
                
                # Find the upper bound
                upper_result = minimize_scalar(
                    lambda v: np.abs(profile_obj(v)), 
                    bounds=(mle_params[param_idx], mle_params[param_idx] * 5),
                    method='bounded'
                )
                upper_bound = upper_result.x
                
                # Calculate Wald interval for comparison
                # Note: This is a simplified approximation using numerical Hessian
                epsilon = 1e-6
                hessian = np.zeros((len(mle_params), len(mle_params)))
                
                for i in range(len(mle_params)):
                    for j in range(len(mle_params)):
                        params_pp = mle_params.copy()
                        params_pm = mle_params.copy()
                        params_mp = mle_params.copy()
                        params_mm = mle_params.copy()
                        
                        params_pp[i] += epsilon
                        params_pp[j] += epsilon
                        params_pm[i] += epsilon
                        params_pm[j] -= epsilon
                        params_mp[i] -= epsilon
                        params_mp[j] += epsilon
                        params_mm[i] -= epsilon
                        params_mm[j] -= epsilon
                        
                        f_pp = negative_log_likelihood(params_pp, model_func, x, y)
                        f_pm = negative_log_likelihood(params_pm, model_func, x, y)
                        f_mp = negative_log_likelihood(params_mp, model_func, x, y)
                        f_mm = negative_log_likelihood(params_mm, model_func, x, y)
                        
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                
                # Try to invert Hessian; if not invertible, use a regularized version
                try:
                    cov_matrix = np.linalg.inv(hessian)
                except np.linalg.LinAlgError:
                    # Regularize Hessian if not invertible
                    hessian_reg = hessian + np.eye(len(mle_params)) * 1e-6
                    cov_matrix = np.linalg.inv(hessian_reg)
                
                se = np.sqrt(cov_matrix[param_idx, param_idx])
                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                
                wald_lower = mle_params[param_idx] - z_crit * se
                wald_upper = mle_params[param_idx] + z_crit * se
                
                return {
                    'parameter': param_name,
                    'mle': mle_params[param_idx],
                    'profile_lower': lower_bound,
                    'profile_upper': upper_bound,
                    'wald_lower': wald_lower,
                    'wald_upper': wald_upper,
                    'full_mle': mle_params
                }
            
            # Generate data and compute intervals based on model type
            np.random.seed(None)
            
            if model_type == "Logistic Regression":
                # Generate logistic regression data
                x = np.random.normal(0, 1, sample_size)
                
                if add_quadratic:
                    # Logistic regression with quadratic term
                    true_params = np.array([intercept, param_value, -0.5])  # [intercept, beta1, beta2]
                    logit = true_params[0] + true_params[1] * x + true_params[2] * x**2
                else:
                    # Simple logistic regression
                    true_params = np.array([intercept, param_value])  # [intercept, beta]
                    logit = true_params[0] + true_params[1] * x
                
                # Generate binary outcomes
                prob = 1 / (1 + np.exp(-logit))
                y = np.random.binomial(1, prob)
                
                # Define model function
                def logistic_model(x, params):
                    if len(params) == 2:
                        # Simple logistic regression
                        return 1 / (1 + np.exp(-(params[0] + params[1] * x)))
                    else:
                        # Logistic regression with quadratic term
                        return 1 / (1 + np.exp(-(params[0] + params[1] * x + params[2] * x**2)))
                
                # Initial parameter guess
                init_params = np.zeros_like(true_params)
                
                # Fit standard logistic regression using statsmodels for comparison
                X_design = np.column_stack([np.ones(sample_size), x])
                if add_quadratic:
                    X_design = np.column_stack([X_design, x**2])
                
                import statsmodels.api as sm
                logit_model = sm.Logit(y, X_design)
                result = logit_model.fit(disp=0)
                
                # Extract coefficients and confidence intervals
                coefs = result.params
                conf_intervals = result.conf_int(alpha=1-conf_level)
                
                # Create results for comparison
                standard_results = []
                for i, name in enumerate(['Intercept', 'Slope', 'Quadratic'] if add_quadratic else ['Intercept', 'Slope']):
                    standard_results.append({
                        'parameter': name,
                        'true_value': true_params[i],
                        'estimate': coefs[i],
                        'lower': conf_intervals[i, 0],
                        'upper': conf_intervals[i, 1]
                    })
                
                standard_df = pd.DataFrame(standard_results)
                
                # Calculate profile likelihood interval for the slope parameter
                profile_results = profile_likelihood_interval(
                    x, y, logistic_model, init_params, 1, 'Slope', conf_level
                )
                
                param_name = "Slope (β₁)"
                true_value = param_value
                model_description = f"Logistic Regression{' with quadratic term' if add_quadratic else ''}"
                
            elif model_type == "Exponential Decay":
                # Generate exponential decay data
                x = np.linspace(0, 10, sample_size)
                true_params = np.array([3.0, param_value])  # [amplitude, decay_rate]
                y_true = true_params[0] * np.exp(-true_params[1] * x)
                y = y_true + np.random.normal(0, noise_level, sample_size)
                
                # Define model function
                def exp_decay_model(x, params):
                    return params[0] * np.exp(-params[1] * x)
                
                # Initial parameter guess
                init_params = np.array([1.0, 0.5])
                
                # Calculate profile likelihood interval for the decay rate parameter
                profile_results = profile_likelihood_interval(
                    x, y, exp_decay_model, init_params, 1, 'Decay Rate', conf_level
                )
                
                # Fit using standard method (non-linear least squares) for comparison
                from scipy.optimize import curve_fit
                
                def curve_fit_func(x, amplitude, decay_rate):
                    return amplitude * np.exp(-decay_rate * x)
                
                standard_params, pcov = curve_fit(curve_fit_func, x, y, p0=[1.0, 0.5])
                standard_errors = np.sqrt(np.diag(pcov))
                
                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                standard_lower = standard_params[1] - z_crit * standard_errors[1]
                standard_upper = standard_params[1] + z_crit * standard_errors[1]
                
                # Create standard results for comparison
                standard_results = [
                    {
                        'parameter': 'Amplitude',
                        'true_value': true_params[0],
                        'estimate': standard_params[0],
                        'lower': standard_params[0] - z_crit * standard_errors[0],
                        'upper': standard_params[0] + z_crit * standard_errors[0]
                    },
                    {
                        'parameter': 'Decay Rate',
                        'true_value': true_params[1],
                        'estimate': standard_params[1],
                        'lower': standard_lower,
                        'upper': standard_upper
                    }
                ]
                
                standard_df = pd.DataFrame(standard_results)
                
                param_name = "Decay Rate (λ)"
                true_value = param_value
                model_description = f"Exponential Decay Model (noise level = {noise_level})"
                
            elif model_type == "Weibull Survival":
                # Generate Weibull survival data
                scale = param_value  # Scale parameter
                shape = shape_param  # Shape parameter
                
                # Generate survival times
                u = np.random.uniform(0, 1, sample_size)
                survival_times = scale * (-np.log(u))**(1/shape)
                
                # Generate censoring times
                censoring_times = np.random.exponential(
                    scale=scale * (-np.log(1-censoring_rate))**(1/shape), 
                    size=sample_size
                )
                
                # Observed times and censoring indicators
                observed_times = np.minimum(survival_times, censoring_times)
                censored = (censoring_times <= survival_times).astype(int)
                
                # For Weibull model, we'll directly use the negative log-likelihood
                def weibull_nll(params, times, censored):
                    """Negative log-likelihood for Weibull survival data"""
                    shape, scale = params
                    if shape <= 0 or scale <= 0:
                        return np.inf
                    
                    # Log-likelihood components
                    ll_event = np.log(shape) + shape * np.log(times) - shape * np.log(scale) - (times/scale)**shape
                    ll_censored = -(times/scale)**shape
                    
                    # Total log-likelihood
                    ll = np.sum((1-censored) * ll_event + censored * ll_censored)
                    return -ll
                
                # Define a model function compatible with the profile likelihood function
                def weibull_model(x, params):
                    # This is just a placeholder for the profile likelihood function
                    # We'll override the negative_log_likelihood function for Weibull
                    return np.zeros_like(x)
                
                # Override the negative log-likelihood function for Weibull model
                def negative_log_likelihood_weibull(params, model_func, times, censored):
                    return weibull_nll(params, times, censored)
                
                # Initial parameter guess
                init_params = np.array([1.0, 1.0])  # [shape, scale]
                
                # Fit using maximum likelihood for comparison
                from scipy.optimize import minimize
                
                result = minimize(weibull_nll, init_params, args=(observed_times, censored))
                standard_params = result.x
                
                # Calculate Hessian numerically
                epsilon = 1e-6
                hessian = np.zeros((2, 2))
                
                for i in range(2):
                    for j in range(2):
                        params_pp = standard_params.copy()
                        params_pm = standard_params.copy()
                        params_mp = standard_params.copy()
                        params_mm = standard_params.copy()
                        
                        params_pp[i] += epsilon
                        params_pp[j] += epsilon
                        params_pm[i] += epsilon
                        params_pm[j] -= epsilon
                        params_mp[i] -= epsilon
                        params_mp[j] += epsilon
                        params_mm[i] -= epsilon
                        params_mm[j] -= epsilon
                        
                        f_pp = weibull_nll(params_pp, observed_times, censored)
                        f_pm = weibull_nll(params_pm, observed_times, censored)
                        f_mp = weibull_nll(params_mp, observed_times, censored)
                        f_mm = weibull_nll(params_mm, observed_times, censored)
                        
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                
                # Calculate standard errors
                try:
                    cov_matrix = np.linalg.inv(hessian)
                    standard_errors = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    # Regularize Hessian if not invertible
                    hessian_reg = hessian + np.eye(2) * 1e-6
                    cov_matrix = np.linalg.inv(hessian_reg)
                    standard_errors = np.sqrt(np.diag(cov_matrix))
                
                z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
                standard_lower = standard_params[1] - z_crit * standard_errors[1]
                standard_upper = standard_params[1] + z_crit * standard_errors[1]
                
                # Create standard results for comparison
                standard_results = [
                    {
                        'parameter': 'Shape',
                        'true_value': shape,
                        'estimate': standard_params[0],
                        'lower': standard_params[0] - z_crit * standard_errors[0],
                        'upper': standard_params[0] + z_crit * standard_errors[0]
                    },
                    {
                        'parameter': 'Scale',
                        'true_value': scale,
                        'estimate': standard_params[1],
                        'lower': standard_lower,
                        'upper': standard_upper
                    }
                ]
                
                standard_df = pd.DataFrame(standard_results)
                
                # Calculate profile likelihood interval for the scale parameter
                # Override the negative_log_likelihood function temporarily
                original_nll = negative_log_likelihood
                negative_log_likelihood = negative_log_likelihood_weibull
                
                profile_results = profile_likelihood_interval(
                    observed_times, censored, weibull_model, init_params, 1, 'Scale', conf_level
                )
                
                # Restore the original function
                negative_log_likelihood = original_nll
                
                param_name = "Scale (λ)"
                true_value = param_value
                model_description = f"Weibull Survival Model (shape = {shape_param}, censoring rate = {censoring_rate*100:.0f}%)"
            
            # Display results
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Model**: {model_description}")
                st.markdown(f"**Parameter of Interest**: {param_name}")
                st.markdown(f"**True Value**: {true_value}")
                st.markdown(f"**Sample Size**: {sample_size}")
            
            with col2:
                st.markdown(f"**MLE Estimate**: {profile_results['mle']:.4f}")
                st.markdown(f"**Profile Likelihood {conf_level*100:.0f}% CI**: [{profile_results['profile_lower']:.4f}, {profile_results['profile_upper']:.4f}]")
                st.markdown(f"**Wald-type {conf_level*100:.0f}% CI**: [{profile_results['wald_lower']:.4f}, {profile_results['wald_upper']:.4f}]")
            
            # Create comparison table
            comparison_df = pd.DataFrame({
                'Method': ['Wald-type', 'Profile Likelihood'],
                'Lower Bound': [profile_results['wald_lower'], profile_results['profile_lower']],
                'Upper Bound': [profile_results['wald_upper'], profile_results['profile_upper']],
                'Width': [
                    profile_results['wald_upper'] - profile_results['wald_lower'],
                    profile_results['profile_upper'] - profile_results['profile_lower']
                ],
                'Contains True Value': [
                    profile_results['wald_lower'] <= true_value <= profile_results['wald_upper'],
                    profile_results['profile_lower'] <= true_value <= profile_results['profile_upper']
                ],
                'Symmetric': [
                    np.isclose(profile_results['mle'] - profile_results['wald_lower'], 
                              profile_results['wald_upper'] - profile_results['mle'], rtol=0.05),
                    np.isclose(profile_results['mle'] - profile_results['profile_lower'], 
                              profile_results['profile_upper'] - profile_results['mle'], rtol=0.05)
                ]
            })
            
            st.subheader(f"{conf_level*100:.0f}% Confidence Intervals for {param_name}")
            st.dataframe(comparison_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Width': '{:.4f}'
            }))
            
            # Create visualization
            fig = go.Figure()
            
            # Add intervals as segments
            methods = ['Wald-type', 'Profile Likelihood']
            y_positions = [1, 2]
            lower_bounds = [profile_results['wald_lower'], profile_results['profile_lower']]
            upper_bounds = [profile_results['wald_upper'], profile_results['profile_upper']]
            colors = ['blue', 'red']
            
            for i, method_name in enumerate(methods):
                fig.add_trace(go.Scatter(
                    x=[lower_bounds[i], upper_bounds[i]],
                    y=[y_positions[i], y_positions[i]],
                    mode='lines',
                    name=method_name,
                    line=dict(color=colors[i], width=4)
                ))
                
                # Add endpoints as markers
                fig.add_trace(go.Scatter(
                    x=[lower_bounds[i], upper_bounds[i]],
                    y=[y_positions[i], y_positions[i]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(color=colors[i], size=8)
                ))
                
                # Add labels for bounds
                fig.add_annotation(
                    x=lower_bounds[i] - 0.1,
                    y=y_positions[i],
                    text=f"{lower_bounds[i]:.4f}",
                    showarrow=False,
                    xanchor="right"
                )
                
                fig.add_annotation(
                    x=upper_bounds[i] + 0.1,
                    y=y_positions[i],
                    text=f"{upper_bounds[i]:.4f}",
                    showarrow=False,
                    xanchor="left"
                )
            
            # Add vertical lines for true value and MLE
            fig.add_vline(x=true_value, line=dict(color='black', width=2, dash='dash'),
                         annotation=dict(text=f"True Value: {true_value}", showarrow=False))
            
            fig.add_vline(x=profile_results['mle'], line=dict(color='green', width=2),
                         annotation=dict(text=f"MLE: {profile_results['mle']:.4f}", showarrow=False))
            
            # Update layout
# Update layout
            fig.update_layout(
                title=f'Comparison of {conf_level*100:.0f}% Confidence Intervals for {param_name}',
                xaxis_title='Parameter Value',
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_positions,
                    ticktext=methods,
                    showgrid=False
                ),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate visualization of model fit
            if model_type == "Exponential Decay":
                # Create data visualization with fitted model
                fit_fig = go.Figure()
                
                # Add observed data points
                fit_fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name='Observed Data',
                    marker=dict(color='blue', size=8)
                ))
                
                # Add true model
                fit_fig.add_trace(go.Scatter(
                    x=x, y=y_true,
                    mode='lines',
                    name='True Model',
                    line=dict(color='green', width=2)
                ))
                
                # Add fitted model using MLE
                y_fit = profile_results['full_mle'][0] * np.exp(-profile_results['full_mle'][1] * x)
                fit_fig.add_trace(go.Scatter(
                    x=x, y=y_fit,
                    mode='lines',
                    name='Fitted Model (MLE)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Update layout
                fit_fig.update_layout(
                    title='Exponential Decay Model Fit',
                    xaxis_title='x',
                    yaxis_title='y',
                    height=400
                )
                
                st.plotly_chart(fit_fig, use_container_width=True)
                
            elif model_type == "Logistic Regression":
                # Create data visualization with fitted model
                fit_fig = go.Figure()
                
                # Sort x for better visualization
                sort_idx = np.argsort(x)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                
                # Add observed data points with jittered y for better visualization
                y_jittered = y_sorted * 0.95 + 0.025
                fit_fig.add_trace(go.Scatter(
                    x=x_sorted, y=y_jittered,
                    mode='markers',
                    name='Observed Data',
                    marker=dict(color='blue', size=8, opacity=0.7)
                ))
                
                # Add true probability curve
                x_curve = np.linspace(min(x), max(x), 100)
                if add_quadratic:
                    logit_curve = true_params[0] + true_params[1] * x_curve + true_params[2] * x_curve**2
                else:
                    logit_curve = true_params[0] + true_params[1] * x_curve
                prob_curve = 1 / (1 + np.exp(-logit_curve))
                
                fit_fig.add_trace(go.Scatter(
                    x=x_curve, y=prob_curve,
                    mode='lines',
                    name='True Probability',
                    line=dict(color='green', width=2)
                ))
                
                # Add fitted probability curve using MLE
                if add_quadratic:
                    logit_fit = profile_results['full_mle'][0] + profile_results['full_mle'][1] * x_curve + profile_results['full_mle'][2] * x_curve**2
                else:
                    logit_fit = profile_results['full_mle'][0] + profile_results['full_mle'][1] * x_curve
                prob_fit = 1 / (1 + np.exp(-logit_fit))
                
                fit_fig.add_trace(go.Scatter(
                    x=x_curve, y=prob_fit,
                    mode='lines',
                    name='Fitted Probability (MLE)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Update layout
                fit_fig.update_layout(
                    title='Logistic Regression Model Fit',
                    xaxis_title='x',
                    yaxis_title='Probability',
                    height=400
                )
                
                st.plotly_chart(fit_fig, use_container_width=True)
                
            elif model_type == "Weibull Survival":
                # Create survival curve visualization
                surv_fig = go.Figure()
                
                # Generate Kaplan-Meier estimates
                from statsmodels.duration.survfunc import SurvfuncRight
                kmf = SurvfuncRight(observed_times, 1-censored)
                
                # Add Kaplan-Meier curve
                surv_fig.add_trace(go.Scatter(
                    x=kmf.surv_times,
                    y=kmf.surv_prob,
                    mode='lines+markers',
                    name='Kaplan-Meier Estimate',
                    line=dict(color='blue', width=2)
                ))
                
                # Add true survival curve
                t_curve = np.linspace(0, max(observed_times) * 1.2, 100)
                true_surv = np.exp(-(t_curve/param_value)**shape_param)
                
                surv_fig.add_trace(go.Scatter(
                    x=t_curve, y=true_surv,
                    mode='lines',
                    name='True Survival Function',
                    line=dict(color='green', width=2)
                ))
                
                # Add fitted survival curve using MLE
                fitted_surv = np.exp(-(t_curve/profile_results['full_mle'][1])**profile_results['full_mle'][0])
                
                surv_fig.add_trace(go.Scatter(
                    x=t_curve, y=fitted_surv,
                    mode='lines',
                    name='Fitted Survival Function (MLE)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Update layout
                surv_fig.update_layout(
                    title='Weibull Survival Model Fit',
                    xaxis_title='Time',
                    yaxis_title='Survival Probability',
                    height=400
                )
                
                st.plotly_chart(surv_fig, use_container_width=True)
            
            # Add profile likelihood visualization
            st.subheader("Profile Likelihood Visualization")
            
            # Generate profile likelihood curve
            param_range = np.linspace(
                profile_results['profile_lower'] * 0.8,
                profile_results['profile_upper'] * 1.2,
                100
            )
            
            profile_values = []
            
            if model_type == "Weibull Survival":
                # Use Weibull-specific negative log-likelihood
                for param in param_range:
                    # Create a copy of the MLE parameters and update the parameter of interest
                    params = profile_results['full_mle'].copy()
                    params[1] = param
                    
                    # Compute negative log-likelihood
                    nll = weibull_nll(params, observed_times, censored)
                    profile_values.append(nll)
            else:
                # Use the general negative log-likelihood
                for param in param_range:
                    # Create a copy of the MLE parameters and update the parameter of interest
                    params = profile_results['full_mle'].copy()
                    params[1] = param
                    
                    # Compute negative log-likelihood
                    if model_type == "Logistic Regression":
                        nll = negative_log_likelihood(params, logistic_model, x, y)
                    else:  # Exponential Decay
                        nll = negative_log_likelihood(params, exp_decay_model, x, y)
                    
                    profile_values.append(nll)
            
            # Convert to profile likelihood
            min_nll = min(profile_values)
            profile_likelihood = np.exp(-(np.array(profile_values) - min_nll))
            
            # Create profile likelihood plot
            prof_fig = go.Figure()
            
            # Add profile likelihood curve
            prof_fig.add_trace(go.Scatter(
                x=param_range, y=profile_likelihood,
                mode='lines',
                name='Profile Likelihood',
                line=dict(color='blue', width=2)
            ))
            
            # Add threshold for confidence interval
            threshold = np.exp(-stats.chi2.ppf(conf_level, 1) / 2)
            prof_fig.add_hline(y=threshold, line=dict(color='red', width=2, dash='dash'),
                             annotation=dict(text=f"Threshold: {threshold:.4f}", showarrow=False))
            
            # Add vertical lines for interval bounds
            prof_fig.add_vline(x=profile_results['profile_lower'], line=dict(color='red', width=2),
                             annotation=dict(text=f"Lower: {profile_results['profile_lower']:.4f}", showarrow=False))
            
            prof_fig.add_vline(x=profile_results['profile_upper'], line=dict(color='red', width=2),
                             annotation=dict(text=f"Upper: {profile_results['profile_upper']:.4f}", showarrow=False))
            
            # Add vertical line for MLE
            prof_fig.add_vline(x=profile_results['mle'], line=dict(color='green', width=2),
                             annotation=dict(text=f"MLE: {profile_results['mle']:.4f}", showarrow=False))
            
            # Add vertical line for true value
            prof_fig.add_vline(x=true_value, line=dict(color='black', width=2, dash='dash'),
                             annotation=dict(text=f"True: {true_value}", showarrow=False))
            
            # Update layout
            prof_fig.update_layout(
                title='Profile Likelihood Function',
                xaxis_title=param_name,
                yaxis_title='Profile Likelihood',
                height=400
            )
            
            st.plotly_chart(prof_fig, use_container_width=True)
            
            # Add interpretation
            st.subheader("Interpretation")
            
            # Asymmetry measure for profile likelihood interval
            profile_asymmetry = (profile_results['profile_upper'] - profile_results['mle']) / (profile_results['mle'] - profile_results['profile_lower']) - 1
            
            # Standard vs. profile width ratio
            width_ratio = (profile_results['profile_upper'] - profile_results['profile_lower']) / (profile_results['wald_upper'] - profile_results['wald_lower'])
            
            st.markdown(f"""
            ### Profile Likelihood vs. Wald-type Confidence Intervals
            
            **Key results for {param_name}:**
            
            - Maximum Likelihood Estimate (MLE): {profile_results['mle']:.4f}
            - True parameter value: {true_value}
            - Profile likelihood {conf_level*100:.0f}% CI: [{profile_results['profile_lower']:.4f}, {profile_results['profile_upper']:.4f}]
            - Standard Wald-type {conf_level*100:.0f}% CI: [{profile_results['wald_lower']:.4f}, {profile_results['wald_upper']:.4f}]
            
            **Comparison of methods:**
            
            1. **Interval width**: 
               - Profile likelihood interval: {profile_results['profile_upper'] - profile_results['profile_lower']:.4f}
               - Wald-type interval: {profile_results['wald_upper'] - profile_results['wald_lower']:.4f}
               - Ratio (Profile/Wald): {width_ratio:.2f}
            
            2. **Symmetry around MLE**:
               - Profile likelihood interval: {"Symmetric" if abs(profile_asymmetry) < 0.1 else "Asymmetric"}
               - Wald-type interval: {"Symmetric" if np.isclose(profile_results['mle'] - profile_results['wald_lower'], profile_results['wald_upper'] - profile_results['mle'], rtol=0.05) else "Asymmetric"}
            
            3. **Contain true parameter**:
               - Profile likelihood interval: {"Yes" if profile_results['profile_lower'] <= true_value <= profile_results['profile_upper'] else "No"}
               - Wald-type interval: {"Yes" if profile_results['wald_lower'] <= true_value <= profile_results['wald_upper'] else "No"}
            """)
            
            # Model-specific interpretation
            if model_type == "Logistic Regression":
                st.markdown(f"""
                **Insights for Logistic Regression**:
                
                Logistic regression parameters can have skewed sampling distributions, especially with small samples or when the parameter values are large. The profile likelihood interval accounts for this skewness.
                
                {"The addition of a quadratic term makes the model more complex, increasing the benefit of using profile likelihood methods." if add_quadratic else ""}
                
                With this sample size (n = {sample_size}), the profile likelihood interval is {"substantially" if width_ratio < 0.9 or width_ratio > 1.1 else "only slightly"} different from the Wald interval.
                """)
                
            elif model_type == "Exponential Decay":
                st.markdown(f"""
                **Insights for Exponential Decay Model**:
                
                The decay rate parameter in exponential models often has a skewed sampling distribution, particularly when the true value is small or the noise level is high.
                
                With noise level = {noise_level} and sample size n = {sample_size}, the profile likelihood interval {"captures this skewness well" if abs(profile_asymmetry) > 0.1 else "is relatively symmetric, similar to the Wald interval"}.
                
                For nonlinear models like exponential decay, profile likelihood intervals generally provide better coverage properties than Wald intervals, especially for small samples.
                """)
                
            elif model_type == "Weibull Survival":
                st.markdown(f"""
                **Insights for Weibull Survival Model**:
                
                Survival models with censored data often result in skewed sampling distributions for the parameters. The profile likelihood method accounts for this skewness, which is especially important with higher censoring rates.
                
                With censoring rate = {censoring_rate*100:.0f}% and sample size n = {sample_size}, the profile likelihood interval {"shows notable asymmetry" if abs(profile_asymmetry) > 0.1 else "is relatively symmetric in this particular sample"}.
                
                The scale parameter in Weibull models represents the characteristic life, and its sampling distribution can be particularly skewed for small samples or high censoring.
                """)
            
            # General guidance on profile likelihood
            st.markdown("""
            ### When to Use Profile Likelihood Intervals
            
            **Benefits of profile likelihood intervals:**
            
            1. **Better coverage properties**, especially for:
               - Small sample sizes
               - Parameters with restricted ranges (e.g., variances, rates)
               - Nonlinear models
               - Models with nuisance parameters
            
            2. **Asymmetric intervals** when appropriate:
               - Automatically adjusts for skewness in the sampling distribution
               - Respects parameter boundaries (e.g., non-negative parameters)
            
            3. **Invariance to parameterization**:
               - Results are the same regardless of how the model is parameterized
            
            **Considerations:**
            
            1. **Computational intensity**:
               - Requires optimization for each point in the profile
               - More complex to implement than Wald intervals
            
            2. **Interpretation**:
               - Based on likelihood ratio tests
               - Represents the set of parameter values that cannot be rejected by a likelihood ratio test
            
            **Practical recommendation:**
            
            Consider using profile likelihood intervals when:
            - Working with complex nonlinear models
            - Sample size is small
            - Parameters have restricted ranges
            - Standard assumptions for Wald intervals may be violated
            - Coverage properties are critically important
            """)
    
    elif method_type == "Simultaneous Confidence Bands":
        st.subheader("Simultaneous Confidence Bands")
        
        st.markdown("""
        Simultaneous confidence bands extend the concept of confidence intervals to functions, providing bounds that contain the entire true function with a specified probability. These are useful in regression, smoothing, and time series analysis.
        """)
        
        # Options for simultaneous confidence bands
        col1, col2 = st.columns(2)
        
        with col1:
            function_type = st.radio(
                "Function type",
                ["Linear Regression", "Polynomial Regression", "Nonparametric Regression"]
            )
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
        
        with col2:
            sample_size = st.slider("Sample size", 10, 200, 50)
            noise_level = st.slider("Noise level", 0.1, 2.0, 1.0, 0.1)
        
        # Additional options based on function type
        if function_type == "Linear Regression":
            true_intercept = st.slider("True intercept", -5.0, 5.0, 1.0, 0.5)
            true_slope = st.slider("True slope", -2.0, 2.0, 0.5, 0.1)
        
        elif function_type == "Polynomial Regression":
            poly_degree = st.slider("Polynomial degree", 2, 5, 3)
            extrapolate = st.checkbox("Show extrapolation region", value=True)
        
        elif function_type == "Nonparametric Regression":
            bandwidth = st.slider("Smoothing bandwidth", 0.05, 1.0, 0.2, 0.05)
            band_method = st.radio(
                "Confidence band method", 
                ["Working-Hotelling", "Bootstrap", "Bonferroni"]
            )
        
        if st.button("Generate Simultaneous Confidence Bands", key="gen_bands"):
            # Generate data based on function type
            np.random.seed(None)
            
            if function_type == "Linear Regression":
                # Generate data for linear regression
                x = np.random.uniform(-3, 3, sample_size)
                true_y = true_intercept + true_slope * x
                y = true_y + np.random.normal(0, noise_level, sample_size)
                
                # Fit linear regression
                X_design = np.column_stack([np.ones(sample_size), x])
                beta_hat = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
                fitted_intercept, fitted_slope = beta_hat
                
                # Calculate residuals and MSE
                residuals = y - (fitted_intercept + fitted_slope * x)
                mse = np.mean(residuals**2)
                
                # Generate x values for prediction
                x_pred = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
                X_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
                
                # Calculate fitted values and pointwise standard errors
                y_pred = X_pred @ beta_hat
                se_fit = np.sqrt(mse * np.diag(X_pred @ np.linalg.inv(X_design.T @ X_design) @ X_pred.T))
                
                # Pointwise confidence intervals
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 2)
                pointwise_margin = t_crit * se_fit
                pointwise_lower = y_pred - pointwise_margin
                pointwise_upper = y_pred + pointwise_margin
                
                # Working-Hotelling simultaneous confidence bands
                f_crit = stats.f.ppf(conf_level, 2, sample_size - 2)
                w = np.sqrt(2 * f_crit)
                simultaneous_margin = w * se_fit
                simultaneous_lower = y_pred - simultaneous_margin
                simultaneous_upper = y_pred + simultaneous_margin
                
                # Calculate true function values for comparison
                true_y_pred = true_intercept + true_slope * x_pred
                
                # Create labels
                function_name = "Linear Regression"
                pointwise_name = f"Pointwise {conf_level*100:.0f}% CI"
                simultaneous_name = f"Working-Hotelling {conf_level*100:.0f}% Band"
                
            elif function_type == "Polynomial Regression":
                # Generate data for polynomial regression with a true polynomial function
                x = np.random.uniform(-1, 1, sample_size)
                
                # Generate true polynomial coefficients
                true_coefs = np.random.normal(0, 1, poly_degree + 1)
                true_coefs = true_coefs / np.max(np.abs(true_coefs)) * 2  # Scale coefficients
                
                # Calculate true function values
                X_poly = np.column_stack([x**i for i in range(poly_degree + 1)])
                true_y = X_poly @ true_coefs
                y = true_y + np.random.normal(0, noise_level, sample_size)
                
                # Fit polynomial regression
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                
                poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
                X_poly = poly_features.fit_transform(x.reshape(-1, 1))
                
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Generate x values for prediction
                if extrapolate:
                    x_pred = np.linspace(-1.5, 1.5, 100)
                else:
                    x_pred = np.linspace(min(x), max(x), 100)
                
                X_pred_poly = poly_features.transform(x_pred.reshape(-1, 1))
                
                # Calculate fitted values
                y_pred = model.predict(X_pred_poly)
                
                # Calculate true function values for comparison
                true_y_pred = np.zeros_like(x_pred)
                for i in range(poly_degree + 1):
                    true_y_pred += true_coefs[i] * x_pred**i
                
                # Calculate residuals and MSE
                fitted_y = model.predict(X_poly)
                residuals = y - fitted_y
                mse = np.mean(residuals**2)
                
                # Calculate pointwise confidence intervals
                # Var(ŷ) = σ² * x_pred' * (X'X)⁻¹ * x_pred
                X_design = X_poly
                cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)
                se_fit = np.sqrt(np.sum(X_pred_poly @ cov_matrix * X_pred_poly, axis=1))
                
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - poly_degree - 1)
                pointwise_margin = t_crit * se_fit
                pointwise_lower = y_pred - pointwise_margin
                pointwise_upper = y_pred + pointwise_margin
                
                # Scheffe simultaneous confidence bands
                f_crit = stats.f.ppf(conf_level, poly_degree + 1, sample_size - poly_degree - 1)
                w = np.sqrt((poly_degree + 1) * f_crit)
                simultaneous_margin = w * se_fit
                simultaneous_lower = y_pred - simultaneous_margin
                simultaneous_upper = y_pred + simultaneous_margin
                
                # Create labels
                function_name = f"Polynomial Regression (degree = {poly_degree})"
                pointwise_name = f"Pointwise {conf_level*100:.0f}% CI"
                simultaneous_name = f"Scheffe {conf_level*100:.0f}% Band"
                
            elif function_type == "Nonparametric Regression":
                # Generate data for nonparametric regression
                x = np.random.uniform(0, 1, sample_size)
                
                # True function: sinusoidal with some complexity
                true_y = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
                y = true_y + np.random.normal(0, noise_level, sample_size)
                
                # Sort data for easier smoothing
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]
                
                # Generate x values for prediction
                x_pred = np.linspace(0, 1, 100)
                
                # Kernel smoothing
                from sklearn.metrics.pairwise import rbf_kernel
                
                def kernel_smoother(x_train, y_train, x_test, bandwidth):
                    # Compute RBF kernel weights
                    gamma = 1 / (2 * bandwidth**2)
                    weights = rbf_kernel(x_test.reshape(-1, 1), x_train.reshape(-1, 1), gamma=gamma)
                    
                    # Normalize weights
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    
                    # Compute weighted average
                    y_pred = weights @ y_train
                    
                    return y_pred, weights
                
                # Compute smoothed predictions
                y_pred, kernel_weights = kernel_smoother(x, y, x_pred, bandwidth)
                
                # Compute true function values
                true_y_pred = np.sin(2 * np.pi * x_pred) + 0.5 * np.sin(4 * np.pi * x_pred)
                
                # Calculate pointwise standard errors
                # For kernel smoothing: Var(ŷ) ≈ σ² * sum(w_i²)
                residuals = y - kernel_smoother(x, y, x, bandwidth)[0]
                mse = np.mean(residuals**2)
                se_fit = np.sqrt(mse * np.sum(kernel_weights**2, axis=1))
                
                # Pointwise confidence intervals
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 2)  # Approx df
                pointwise_margin = t_crit * se_fit
                pointwise_lower = y_pred - pointwise_margin
                pointwise_upper = y_pred + pointwise_margin
                
                # Simultaneous bands based on selected method
                if band_method == "Working-Hotelling":
                    # Working-Hotelling band (approximation)
                    # Calculate effective degrees of freedom
                    effective_df = np.trace(kernel_weights @ kernel_weights.T)
                    f_crit = stats.f.ppf(conf_level, effective_df, sample_size - effective_df)
                    w = np.sqrt(effective_df * f_crit)
                    simultaneous_margin = w * se_fit
                    
                    method_name = "Working-Hotelling"
                    
                elif band_method == "Bootstrap":
                    # Bootstrap-based bands
                    n_bootstrap = 1000
                    max_deviations = []
                    
                    for _ in range(n_bootstrap):
                        # Generate bootstrap sample
                        bootstrap_idx = np.random.choice(sample_size, sample_size, replace=True)
                        x_boot = x[bootstrap_idx]
                        y_boot = y[bootstrap_idx]
                        
                        # Compute bootstrap predictions
                        y_boot_pred, _ = kernel_smoother(x_boot, y_boot, x_pred, bandwidth)
                        
                        # Compute deviations from original predictions
                        deviations = np.abs(y_boot_pred - y_pred) / se_fit
                        max_deviations.append(np.max(deviations))
                    
                    # Compute critical multiplier as the conf_level quantile of max deviations
                    w = np.quantile(max_deviations, conf_level)
                    simultaneous_margin = w * se_fit
                    
                    method_name = "Bootstrap"
                    
                else:  # Bonferroni
                    # Bonferroni adjustment for multiple comparisons
                    alpha_adj = (1 - conf_level) / len(x_pred)
                    t_crit_adj = stats.t.ppf(1 - alpha_adj/2, sample_size - 2)  # Approx df
                    simultaneous_margin = t_crit_adj * se_fit
                    
                    method_name = "Bonferroni"
                
                simultaneous_lower = y_pred - simultaneous_margin
                simultaneous_upper = y_pred + simultaneous_margin
                
                # Create labels
                function_name = f"Nonparametric Regression (bandwidth = {bandwidth})"
                pointwise_name = f"Pointwise {conf_level*100:.0f}% CI"
                simultaneous_name = f"{method_name} {conf_level*100:.0f}% Band"
            
            # Create visualization
            fig = go.Figure()
            
            # Add data points
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Observed Data',
                marker=dict(color='black', size=8, opacity=0.6)
            ))
            
            # Add true function
            fig.add_trace(go.Scatter(
                x=x_pred, y=true_y_pred,
                mode='lines',
                name='True Function',
                line=dict(color='green', width=2)
            ))
            
            # Add fitted function
            fig.add_trace(go.Scatter(
                x=x_pred, y=y_pred,
                mode='lines',
                name='Fitted Function',
                line=dict(color='blue', width=2)
            ))
            
            # Add pointwise confidence intervals
            fig.add_trace(go.Scatter(
                x=x_pred, y=pointwise_upper,
                mode='lines',
                name=pointwise_name,
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=x_pred, y=pointwise_lower,
                mode='lines',
                name='',
                showlegend=False,
                line=dict(color='red', width=1, dash='dash')
            ))
            
            # Add simultaneous confidence bands
            fig.add_trace(go.Scatter(
                x=x_pred, y=simultaneous_upper,
                mode='lines',
                name=simultaneous_name,
                line=dict(color='purple', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_pred, y=simultaneous_lower,
                mode='lines',
                name='',
                showlegend=False,
                line=dict(color='purple', width=1)
            ))
            
            # Fill the region between pointwise intervals
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_pred, x_pred[::-1]]),
                y=np.concatenate([pointwise_upper, pointwise_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                name=pointwise_name + ' Region',
                showlegend=False
            ))
            
            # Fill the region between simultaneous bands
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_pred, x_pred[::-1]]),
                y=np.concatenate([simultaneous_upper, simultaneous_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 0, 128, 0.1)',
                line=dict(color='rgba(128, 0, 128, 0)'),
                name=simultaneous_name + ' Region',
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{function_name} with Confidence Bands',
                xaxis_title='x',
                yaxis_title='y',
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate coverage statistics
            true_in_pointwise = np.all((true_y_pred >= pointwise_lower) & (true_y_pred <= pointwise_upper))
            true_in_simultaneous = np.all((true_y_pred >= simultaneous_lower) & (true_y_pred <= simultaneous_upper))
            
            pointwise_width = np.mean(pointwise_upper - pointwise_lower)
            simultaneous_width = np.mean(simultaneous_upper - simultaneous_lower)
            width_ratio = simultaneous_width / pointwise_width
            
            # Create summary table
            summary_df = pd.DataFrame({
                'Band Type': ['Pointwise', 'Simultaneous'],
                'Average Width': [pointwise_width, simultaneous_width],
                'Width Ratio': [1.0, width_ratio],
                'Contains True Function': [true_in_pointwise, true_in_simultaneous]
            })
            
            st.subheader("Confidence Band Summary")
            st.dataframe(summary_df.style.format({
                'Average Width': '{:.4f}',
                'Width Ratio': '{:.2f}'
            }))
            
            # Add interpretation
            st.subheader("Interpretation")
            
            st.markdown(f"""
            ### Simultaneous vs. Pointwise Confidence Bands
            
            **Key results**:
            
            - Pointwise {conf_level*100:.0f}% confidence bands: 
              - Average width: {pointwise_width:.4f}
              - Contains entire true function: {"Yes" if true_in_pointwise else "No"}
            
            - Simultaneous {conf_level*100:.0f}% confidence bands:
              - Average width: {simultaneous_width:.4f}
              - Contains entire true function: {"Yes" if true_in_simultaneous else "No"}
              - Width ratio (Simultaneous/Pointwise): {width_ratio:.2f}
            """)
            
            # Function-specific interpretation
            if function_type == "Linear Regression":
                st.markdown(f"""
                **Insights for Linear Regression**:
                
                The Working-Hotelling bands for linear regression account for the joint distribution of the intercept and slope parameters. They ensure that the entire regression line is contained within the bands with {conf_level*100:.0f}% confidence.
                
                For linear regression, these bands are widest at the extremes of the x-range and narrowest near the mean of x, creating a characteristic "bow-tie" shape.
                
                The simultaneous bands are approximately {(width_ratio-1)*100:.1f}% wider than the pointwise intervals, reflecting the additional uncertainty when making inferences about the entire regression line rather than individual points.
                """)
                
            elif function_type == "Polynomial Regression":
                st.markdown(f"""
                **Insights for Polynomial Regression**:
                
                With polynomial regression of degree {poly_degree}, the Scheffe bands account for the joint distribution of all {poly_degree+1} regression parameters. The width adjustment factor is ${(width_ratio-1)*100:.1f}\%$ larger than for pointwise intervals.
                
                {"The bands widen substantially in the extrapolation region (outside the range of observed data), reflecting the increased uncertainty when predicting beyond the data range." if extrapolate else "The bands maintain a relatively consistent width across the observed data range, with slight widening at the boundaries."}
                
                Higher polynomial degrees result in wider confidence bands due to:
                1. More parameters to estimate (increased degrees of freedom)
                2. Greater potential for overfitting
                3. Higher correlation between parameter estimates
                """)
                
            elif function_type == "Nonparametric Regression":
                st.markdown(f"""
                **Insights for Nonparametric Regression**:
                
                The {band_method} simultaneous bands for nonparametric regression ensure that the entire smooth function is contained within the bands with {conf_level*100:.0f}% confidence.
                
                With kernel smoothing (bandwidth = {bandwidth}):
                - Smaller bandwidths lead to more flexible curves but wider confidence bands
                - Larger bandwidths lead to smoother curves but potentially increased bias
                
                The simultaneous bands are {(width_ratio-1)*100:.1f}% wider than the pointwise intervals, which is {"larger than" if width_ratio > 1.5 else "comparable to" if width_ratio > 1.2 else "smaller than"} what we typically see in parametric models. This reflects the {"greater" if width_ratio > 1.5 else "typical" if width_ratio > 1.2 else "lower"} dimensionality of the nonparametric estimation.
                """)
                
                # Method-specific interpretation
                if band_method == "Working-Hotelling":
                    st.markdown("""
                    The Working-Hotelling approach approximates the effective degrees of freedom in the nonparametric model to construct simultaneous bands. This approximation works well for smoothing methods but may be conservative.
                    """)
                elif band_method == "Bootstrap":
                    st.markdown("""
                    The bootstrap-based bands directly estimate the distribution of the maximum deviation between the true and estimated functions, providing accurate coverage without parametric assumptions.
                    """)
                else:  # Bonferroni
                    st.markdown("""
                    The Bonferroni approach adjusts for multiple comparisons across the entire function domain. It is generally conservative but easy to implement for any regression method.
                    """)
            
            # General guidance
            st.markdown("""
            ### When to Use Simultaneous Confidence Bands
            
            **Key differences**:
            
            - **Pointwise intervals** provide confidence bounds for the true function at each individual point
            - **Simultaneous bands** provide confidence bounds that contain the entire true function with the specified probability
            
            **Applications of simultaneous bands**:
            
            1. **Regression diagnostics**: Testing for linearity, identifying regions of poor fit
            2. **Function comparison**: Testing whether two functions differ significantly
            3. **Uncertainty visualization**: Showing the overall uncertainty in function estimation
            4. **Model validation**: Checking if a theoretical model is consistent with data
            
            **Considerations**:
            
            1. **Width trade-off**: Simultaneous bands are necessarily wider than pointwise intervals
            2. **Interpretability**: Bands represent uncertainty about the entire function
            3. **Method selection**: Different methods (Working-Hotelling, Scheffé, bootstrap) have different properties
            4. **Dimensionality**: Higher-dimensional models require wider bands
            
            **Practical recommendation**:
            
            Use simultaneous confidence bands when:
            - Making inferences about the entire function rather than individual points
            - Testing functional hypotheses
            - Presenting comprehensive uncertainty assessments
            - Controlling the family-wise error rate across the function domain
            """)

# Mathematical Proofs Module
elif nav == "Mathematical Proofs":
    st.header("Mathematical Proofs and Derivations")
    
    proof_type = st.selectbox(
        "Select topic",
        ["Pivotal Quantity Method", "Normal Mean Derivation", "Student's t Derivation",
         "Binomial Proportion Intervals", "Confidence vs. Credible Intervals"]
    )
    
    if proof_type == "Pivotal Quantity Method":
        st.subheader("The Pivotal Quantity Method")
        
        st.markdown(r"""
        ### Definition and Concept
        
        A **pivotal quantity** is a function of both the data and the parameter whose distribution does not depend on any unknown parameter.
        
        Let $X = (X_1, X_2, \ldots, X_n)$ be a random sample from a distribution with parameter $\theta$. A function $Q(X, \theta)$ is a pivotal quantity if its distribution is the same for all values of $\theta$.
        
        ### Key Properties
        
        1. The distribution of $Q(X, \theta)$ is completely known (it doesn't depend on unknown parameters)
        2. $Q(X, \theta)$ involves both the data $X$ and the parameter $\theta$
        3. We can use $Q(X, \theta)$ to construct confidence intervals by "pivoting"
        
        ### General Method for Constructing Confidence Intervals
        
        1. Find a pivotal quantity $Q(X, \theta)$
        2. Determine values $a$ and $b$ such that $P(a \leq Q(X, \theta) \leq b) = 1-\alpha$
        3. Solve the inequalities for $\theta$ to get $L(X) \leq \theta \leq U(X)$
        4. The interval $[L(X), U(X)]$ is a $(1-\alpha)$ confidence interval for $\theta$
        
        ### Example: Normal Mean with Known Variance
        
        For $X_1, X_2, \ldots, X_n \sim N(\mu, \sigma^2)$ with known $\sigma^2$:
        
        1. The pivotal quantity is:
        
        $$Q(X, \mu) = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$
        
        2. For a standard normal distribution, we know:
        
        $$P(-z_{\alpha/2} \leq Q(X, \mu) \leq z_{\alpha/2}) = 1-\alpha$$
        
        3. Substituting and solving for $\mu$:
        
        $$P\left(-z_{\alpha/2} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}\right) = 1-\alpha$$
        
        $$P\left(\bar{X} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) = 1-\alpha$$
        
        4. Therefore, a $(1-\alpha)$ confidence interval for $\mu$ is:
        
        $$\left[\bar{X} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}}, \bar{X} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right]$$
        
        ### Example: Normal Variance
        
        For $X_1, X_2, \ldots, X_n \sim N(\mu, \sigma^2)$ (we don't need to know $\mu$):
        
        1. The pivotal quantity is:
        
        $$Q(X, \sigma^2) = \frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$$
        
        where $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$
        
        2. For a chi-square distribution with $n-1$ degrees of freedom:
        
        $$P\left(\chi^2_{\alpha/2, n-1} \leq Q(X, \sigma^2) \leq \chi^2_{1-\alpha/2, n-1}\right) = 1-\alpha$$
        
        3. Substituting and solving for $\sigma^2$:
        
        $$P\left(\chi^2_{\alpha/2, n-1} \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, n-1}\right) = 1-\alpha$$
        
        $$P\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}} \leq \sigma^2 \leq \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right) = 1-\alpha$$
        
        4. Therefore, a $(1-\alpha)$ confidence interval for $\sigma^2$ is:
        
        $$\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}, \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right]$$
        
        ### Advantages of Pivotal Quantity Method
        
        1. **General applicability**: Can be used for various parameters and distributions
        2. **Exact intervals**: When exact pivotal quantities exist, provides exact confidence intervals
        3. **Transformation invariance**: For monotone transformations of parameters
        
        ### Limitations
        
        1. Finding a pivotal quantity may not always be possible
        2. For complex models, pivotal quantities might not have simple closed-form distributions
        3. Sometimes leads to implicit rather than explicit formulas for the bounds
        """)
        
    elif proof_type == "Normal Mean Derivation":
        st.subheader("Derivation of Confidence Interval for Normal Mean")
        
        st.markdown(r"""
        ### Case 1: Known Variance
        
        Let $X_1, X_2, \ldots, X_n$ be a random sample from a normal distribution with unknown mean $\mu$ and known variance $\sigma^2$.
        
        #### Step 1: Find a pivotal quantity
        
        The sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ has distribution:
        
        $$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$
        
        Therefore, we can standardize it to create a pivotal quantity:
        
        $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$
        
        This follows a standard normal distribution regardless of the value of $\mu$.
        
        #### Step 2: Use the known distribution to create bounds
        
        For a standard normal random variable $Z$ and significance level $\alpha$:
        
        $$P(-z_{\alpha/2} \leq Z \leq z_{\alpha/2}) = 1-\alpha$$
        
        where $z_{\alpha/2}$ is the $(1-\alpha/2)$ quantile of the standard normal distribution.
        
        #### Step 3: Substitute the pivotal quantity and solve for the parameter
        
        $$P\left(-z_{\alpha/2} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}\right) = 1-\alpha$$
        
        Multiply all parts by $\sigma/\sqrt{n}$:
        
        $$P\left(-z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \leq \bar{X} - \mu \leq z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right) = 1-\alpha$$
        
        Multiply all parts by $-1$ and rearrange:
        
        $$P\left(\bar{X} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right) = 1-\alpha$$
        
        #### Step 4: Write the confidence interval
        
        A $(1-\alpha)$ confidence interval for $\mu$ is:
        
        $$\left[\bar{X} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}, \bar{X} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right]$$
        
        or more compactly:
        
        $$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$
        
        ### Case 2: Unknown Variance
        
        When the variance $\sigma^2$ is unknown, we need to estimate it from the data and use a different pivotal quantity.
        
        #### Step 1: Find a pivotal quantity
        
        The sample variance $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is an unbiased estimator of $\sigma^2$.
        
        For normally distributed data, we can use the t-statistic as a pivotal quantity:
        
        $$T = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$$
        
        This follows a t-distribution with $n-1$ degrees of freedom, regardless of the values of $\mu$ and $\sigma^2$.
        
        #### Step 2: Use the known distribution to create bounds
        
        For a t-distributed random variable $T$ with $n-1$ degrees of freedom:
        
        $$P(-t_{\alpha/2, n-1} \leq T \leq t_{\alpha/2, n-1}) = 1-\alpha$$
        
        where $t_{\alpha/2, n-1}$ is the $(1-\alpha/2)$ quantile of the t-distribution with $n-1$ degrees of freedom.
        
        #### Step 3: Substitute the pivotal quantity and solve for the parameter
        
        $$P\left(-t_{\alpha/2, n-1} \leq \frac{\bar{X} - \mu}{S/\sqrt{n}} \leq t_{\alpha/2, n-1}\right) = 1-\alpha$$
        
        Multiply all parts by $S/\sqrt{n}$:
        
        $$P\left(-t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}} \leq \bar{X} - \mu \leq t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}\right) = 1-\alpha$$
        
        Multiply all parts by $-1$ and rearrange:
        
        $$P\left(\bar{X} - t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}} \leq \mu \leq \bar{X} + t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}\right) = 1-\alpha$$
        
        #### Step 4: Write the confidence interval
        
        A $(1-\alpha)$ confidence interval for $\mu$ is:
        
        $$\left[\bar{X} - t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}, \bar{X} + t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}\right]$$
        
        or more compactly:
        
        $$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}$$
        
        ### Key Differences
        
        1. When $\sigma^2$ is known, we use the standard normal distribution ($Z$)
        2. When $\sigma^2$ is unknown, we use the t-distribution ($T$)
        3. As $n$ increases, $t_{n-1}$ approaches the standard normal distribution, so the two intervals become similar for large sample sizes
        """)
        
    elif proof_type == "Student's t Derivation":
        st.subheader("Derivation of Student's t Distribution")
        
        st.markdown(r"""
        ### Derivation of Student's t Distribution
        
        The t-distribution arises naturally when estimating the mean of a normally distributed population when the sample size is small and the population standard deviation is unknown.
        
        #### Key Components
        
        For a random sample $X_1, X_2, \ldots, X_n$ from a normal distribution $N(\mu, \sigma^2)$:
        
        1. The sample mean $\bar{X} \sim N(\mu, \sigma^2/n)$
        2. The sample variance $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is an unbiased estimator of $\sigma^2$
        3. $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ (chi-square distribution with $n-1$ degrees of freedom)
        4. $\bar{X}$ and $S^2$ are independent random variables
        
        #### Step 1: Start with the standardized sample mean
        
        $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$
        
        #### Step 2: Since $\sigma$ is unknown, replace it with $S$
        
        $$T = \frac{\bar{X} - \mu}{S/\sqrt{n}}$$
        
        We can rewrite this as:
        
        $$T = \frac{Z}{\sqrt{\frac{(n-1)S^2/\sigma^2}{n-1}}} = \frac{Z}{\sqrt{\frac{V/(n-1)}{n-1}}} = \frac{Z}{\sqrt{V/(n-1)}}$$
        
        where $V = (n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$
        
        #### Step 3: Use properties of normal and chi-square distributions
        
        The t-statistic can be expressed as:
        
        $$T = \frac{Z}{\sqrt{V/(n-1)}}$$
        
        where:
        - $Z \sim N(0, 1)$ 
        - $V \sim \chi^2_{n-1}$
        - $Z$ and $V$ are independent
        
        #### Step 4: Derive the probability density function
        
        The probability density function (PDF) of the t-distribution with $\nu = n-1$ degrees of freedom is:
        
        $$f(t) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$
        
        where $\Gamma$ is the gamma function.
        
        The full derivation involves a change of variables and integration over the joint distribution of $Z$ and $V$, which is beyond the scope of this presentation.
        
        #### Step 5: Properties of the t-distribution
        
        1. **Symmetry**: The t-distribution is symmetric around 0
        2. **Heavier tails**: It has heavier tails than the normal distribution
        3. **Convergence**: As $\nu \to \infty$, the t-distribution approaches the standard normal distribution
        4. **Mean and variance**: 
           - For $\nu > 1$: E(T) = 0
           - For $\nu > 2$: Var(T) = $\frac{\nu}{\nu-2}$
        
        ### Application to Confidence Intervals
        
        The t-distribution is used in the construction of confidence intervals for the mean when the population standard deviation is unknown:
        
        $$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}$$
        
        where $t_{\alpha/2, n-1}$ is the critical value from the t-distribution with $n-1$ degrees of freedom.
        
        ### Key Insights
        
        1. The t-distribution accounts for the additional uncertainty from estimating $\sigma$ with $S$
        2. For small samples, t-critical values are larger than corresponding normal critical values
        3. As the sample size increases, the difference becomes negligible
        4. The beauty of the t-distribution is that it provides exact confidence intervals for normal data with unknown variance, regardless of sample size
        """)
        
    elif proof_type == "Binomial Proportion Intervals":
        st.subheader("Derivation of Confidence Intervals for Binomial Proportion")
        
        st.markdown(r"""
        ### Confidence Intervals for a Binomial Proportion
        
        For a binomial random variable $X \sim \text{Bin}(n, p)$ where $p$ is the unknown proportion:
        
        ### 1. Wald (Standard) Interval
        
        The Wald interval is based on the normal approximation to the binomial distribution.
        
        #### Step 1: Define the sample proportion
        
        The sample proportion is $\hat{p} = X/n$, with:
        
        $$E(\hat{p}) = p$$
        $$Var(\hat{p}) = \frac{p(1-p)}{n}$$
        
        #### Step 2: Apply the Central Limit Theorem
        
        For large $n$, by the Central Limit Theorem:
        
        $$\frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}} \stackrel{approx}{\sim} N(0, 1)$$
        
        #### Step 3: Construct the confidence interval
        
        $$P\left(-z_{\alpha/2} \leq \frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}} \leq z_{\alpha/2}\right) \approx 1-\alpha$$
        
        Since $p$ appears in the denominator, we replace it with $\hat{p}$ to get a usable formula:
        
        $$P\left(\hat{p} - z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \leq p \leq \hat{p} + z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\right) \approx 1-\alpha$$
        
        #### Step 4: Write the confidence interval
        
        A $(1-\alpha)$ Wald confidence interval for $p$ is:
        
        $$\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$
        
        **Limitations:**
        - Poor coverage for small $n$
        - Poor coverage when $p$ is close to 0 or 1
        - Can produce intervals outside [0,1]
        
        ### 2. Wilson Score Interval
        
        The Wilson Score interval has better coverage properties, especially for small $n$ or extreme $p$.
        
        #### Step 1: Start with the score test
        
        The score test statistic for testing $H_0: p = p_0$ is:
        
        $$Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$
        
        #### Step 2: Invert the test to get confidence limits
        
        We find values of $p_0$ where $|Z| \leq z_{\alpha/2}$, which means:
        
        $$\left|\frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}\right| \leq z_{\alpha/2}$$
        
        Squaring both sides:
        
        $$\frac{(\hat{p} - p_0)^2}{\frac{p_0(1-p_0)}{n}} \leq z_{\alpha/2}^2$$
        
        #### Step 3: Solve the quadratic equation
        
        After algebraic manipulation, this becomes a quadratic in $p_0$:
        
        $$n(\hat{p} - p_0)^2 \leq z_{\alpha/2}^2 p_0(1-p_0)$$
        
        $$n\hat{p}^2 - 2n\hat{p}p_0 + np_0^2 \leq z_{\alpha/2}^2 p_0 - z_{\alpha/2}^2 p_0^2$$
        
        $$np_0^2 + z_{\alpha/2}^2 p_0^2 - 2n\hat{p}p_0 - z_{\alpha/2}^2 p_0 + n\hat{p}^2 \leq 0$$
        
        $$(n + z_{\alpha/2}^2)p_0^2 - (2n\hat{p} + z_{\alpha/2}^2)p_0 + n\hat{p}^2 \leq 0$$
        
        #### Step 4: Find the roots of the quadratic
        
        The solutions are:
        
        $$p_0 = \frac{2n\hat{p} + z_{\alpha/2}^2 \pm z_{\alpha/2}\sqrt{z_{\alpha/2}^2 + 4n\hat{p}(1-\hat{p})}}{2(n + z_{\alpha/2}^2)}$$
        
        Simplifying:
        
        $$p_0 = \frac{\hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}}}{1 + \frac{z_{\alpha/2}^2}{n}}$$
        
        #### Step 5: Write the confidence interval
        
        A $(1-\alpha)$ Wilson Score confidence interval for $p$ is:
        
        $$\frac{\hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}}}{1 + \frac{z_{\alpha/2}^2}{n}}$$
        
        **Advantages:**
        - Better coverage properties than Wald
        - Works well for small $n$ or extreme $p$
        - Always gives intervals within [0,1]
        
        ### 3. Clopper-Pearson (Exact) Interval
        
        The Clopper-Pearson interval is based directly on the binomial distribution and is often called the "exact" method.
        
        #### Step 1: Use the relationship between binomial and beta distributions
        
        For observed $X = k$ successes out of $n$ trials:
        
        - Lower bound: Find $p_L$ such that $P(X \geq k) = \alpha/2$ when $p = p_L$
        - Upper bound: Find $p_U$ such that $P(X \leq k) = \alpha/2$ when $p = p_U$
        
        #### Step 2: Use properties of the beta distribution
        
        Due to the relationship between the binomial and beta distributions:
        
        - $p_L$ is the $\alpha/2$ quantile of the Beta$(k, n-k+1)$ distribution
        - $p_U$ is the $1-\alpha/2$ quantile of the Beta$(k+1, n-k)$ distribution
        
        #### Step 3: Write the confidence interval
        
        A $(1-\alpha)$ Clopper-Pearson confidence interval for $p$ is:
        
        - Lower bound: $p_L = B(\alpha/2; k, n-k+1)$
        - Upper bound: $p_U = B(1-\alpha/2; k+1, n-k)$
        
        where $B(q; a, b)$ is the $q$ quantile of the Beta$(a, b)$ distribution.
        
        **Properties:**
        - Guaranteed coverage: Always has coverage probability $\geq 1-\alpha$
        - Often conservative (wider than necessary)
        - Always within [0,1]
        - Based directly on the binomial distribution, not approximations
        
        ### Coverage Comparison
        
        For a $(1-\alpha)$ confidence interval, the actual coverage probability is:
        
        $$C(p) = \sum_{k=0}^n I_{[k \in \{k: p \in [L(k), U(k)]\}]} \binom{n}{k} p^k (1-p)^{n-k}$$
        
        where $I$ is the indicator function, and $L(k)$ and $U(k)$ are the lower and upper bounds when $X = k$.
        
        - Wald: Coverage often below $(1-\alpha)$, especially for small $n$ or extreme $p$
        - Wilson: Coverage approximately $(1-\alpha)$, with slight under-coverage in some cases
        - Clopper-Pearson: Coverage always $\geq (1-\alpha)$, often substantially higher
        
        ### Recommendations
        
        - $n \geq 30$ and $\hat{p}$ not extreme: Use Wald (simplest)
        - Most general use: Wilson Score (good balance of simplicity and accuracy)
        - Conservative approach: Clopper-Pearson (guaranteed coverage)
        """)
        
    elif proof_type == "Confidence vs. Credible Intervals":
        st.subheader("Formal Comparison of Confidence vs. Credible Intervals")
        
        st.markdown(r"""
        ### Confidence vs. Credible Intervals: A Formal Comparison
        
        This derivation illustrates the mathematical differences between frequentist confidence intervals and Bayesian credible intervals.
        
        ### 1. Frequentist Confidence Interval
        
        A confidence interval is constructed from the data such that, if the procedure were repeated many times, the true parameter would be contained in the interval a specified proportion of the time.
        
        #### Definition:
        
        A $(1-\alpha)$ confidence interval for parameter $\theta$ is a pair of statistics $(L(X), U(X))$ such that:
        
        $$P_{\theta}(L(X) \leq \theta \leq U(X)) \geq 1-\alpha \quad \forall \theta \in \Theta$$
        
        #### Example: Normal Mean with Known Variance
        
        For $X_1, X_2, \ldots, X_n \sim N(\mu, \sigma^2)$ with known $\sigma^2$:
        
        $$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$
        
        This interval has the property that:
        
        $$P_{\mu}\left(\bar{X} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right) = 1-\alpha$$
        
        for any value of $\mu$.
        
        ### 2. Bayesian Credible Interval
        
        A credible interval is derived from the posterior distribution and provides a direct probability statement about the parameter given the observed data.
        
        #### Definition:
        
        A $(1-\alpha)$ credible interval for parameter $\theta$ is a set $C \subset \Theta$ such that:
        
        $$P(\theta \in C | X) = \int_{C} \pi(\theta | X) d\theta = 1-\alpha$$
        
        where $\pi(\theta | X)$ is the posterior distribution of $\theta$ given data $X$.
        
        #### Example: Normal Mean with Known Variance
        
        For $X_1, X_2, \ldots, X_n \sim N(\mu, \sigma^2)$ with known $\sigma^2$ and prior $\mu \sim N(\mu_0, \tau^2)$:
        
        The posterior distribution is:
        
        $$\mu | X \sim N\left(\frac{\frac{n}{\sigma^2}\bar{X} + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}, \frac{1}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}\right)$$
        
        Let's denote the posterior mean as $\mu_{post}$ and posterior variance as $\sigma_{post}^2$. Then a $(1-\alpha)$ highest posterior density (HPD) credible interval is:
        
        $$\mu_{post} \pm z_{\alpha/2} \cdot \sigma_{post}$$
        
        ### 3. Mathematical Comparison
        
        #### Different Probability Spaces:
        
        - **Confidence Interval**: $P_{\theta}(L(X) \leq \theta \leq U(X)) \geq 1-\alpha$
           - Probability is over the random data $X$
           - $\theta$ is fixed but unknown
           - The interval limits are random variables
        
        - **Credible Interval**: $P(\theta \in C | X) = 1-\alpha$
           - Probability is over the random parameter $\theta$
           - Data $X$ is fixed (what we observed)
           - The interval limits are fixed given the data
        
        #### Mathematical Equivalence Conditions:
        
        For specific cases, confidence and credible intervals can be numerically identical. This occurs when:
        
        1. The prior distribution is "matching" in a specific way
        2. The parameter is a location parameter
        
        For example, in the normal mean case with known variance, if we use an improper uniform prior ($\tau^2 \to \infty$), then:
        
        $$\mu_{post} = \bar{X}$$
        $$\sigma_{post}^2 = \frac{\sigma^2}{n}$$
        
        And the $(1-\alpha)$ credible interval becomes:
        
        $$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$
        
        which is identical to the confidence interval.
        
        ### 4. Different Construction Methods
        
        #### Types of Credible Intervals:
        
        1. **Equal-tailed interval**: 
           $$P(\theta < L | X) = P(\theta > U | X) = \alpha/2$$
        
        2. **Highest Posterior Density (HPD) interval**: 
           The shortest interval with total probability $1-\alpha$
        
        These two can differ when the posterior is asymmetric.
        
        #### Types of Confidence Intervals:
        
        1. **Pivotal quantities**: Based on statistics with known distributions
        2. **Likelihood ratio intervals**: Based on likelihood ratio tests
        3. **Score intervals**: Based on score tests
        
        ### 5. Interpretation Differences
        
        For a 95% interval:
        
        - **Confidence Interval**: "If we were to repeat the experiment many times, about 95% of the intervals constructed would contain the true parameter value."
        
        - **Credible Interval**: "Given the observed data and our prior beliefs, there is a 95% probability that the true parameter value lies within this interval."
        
        ### Conclusion
        
        The fundamental difference is philosophical:
        
        - **Frequentist approach**: Parameters are fixed, and probability statements are made about the data
        - **Bayesian approach**: Data are fixed, and probability statements are made about the parameters
        
        Both approaches provide interval estimates, but with different interpretations and construction methods.
        """)

# Real-world Applications Module
elif nav == "Real-world Applications":
    st.header("Real-world Applications of Confidence Intervals")
    
    application_type = st.selectbox(
        "Select application domain",
        ["Clinical Trials", "A/B Testing", "Environmental Monitoring", "Manufacturing Quality Control"]
    )
    
    if application_type == "Clinical Trials":
        st.subheader("Confidence Intervals in Clinical Trials")
        
        st.markdown("""
        ### Role of Confidence Intervals in Clinical Trials
        
        Confidence intervals play a critical role in clinical trials and medical research by:
        
        1. **Quantifying uncertainty**: Providing a range of plausible values for treatment effects
        2. **Clinical significance**: Helping assess whether effects are not only statistically significant but also clinically meaningful
        3. **Sample size planning**: Informing study design to achieve desired precision
        4. **Regulatory decisions**: Supporting approval processes based on demonstrated effectiveness
        
        ### Common Applications in Clinical Research
        
        #### Treatment Effect Estimation
        
        In a randomized controlled trial comparing a new treatment to a standard treatment or placebo:
        
        - **Binary outcomes** (e.g., recovery rates): Confidence intervals for risk differences, relative risks, or odds ratios
        - **Continuous outcomes** (e.g., blood pressure reduction): Confidence intervals for mean differences
        - **Time-to-event outcomes** (e.g., survival): Confidence intervals for hazard ratios
        
        #### Bioequivalence Studies
        
        For generic drug approval, confidence intervals are used to establish bioequivalence:
        
        - A 90% confidence interval for the ratio of the geometric means of pharmacokinetic parameters (AUC, Cmax) must fall within 80-125% of the reference drug
        
        #### Non-inferiority and Equivalence Trials
        
        - **Non-inferiority**: The lower bound of the confidence interval for the difference must not cross the pre-specified non-inferiority margin
        - **Equivalence**: The entire confidence interval must lie within the pre-specified equivalence margins
        """)
        
        # Interactive non-inferiority trial example
        st.subheader("Interactive Example: Non-inferiority Trial")
        
        col1, col2 = st.columns(2)
        
        with col1:
            standard_effect = st.slider("Standard treatment effect (%)", 60.0, 90.0, 70.0, 1.0)
            new_effect = st.slider("New treatment effect (%)", 60.0, 90.0, 68.0, 1.0)
            non_inferiority_margin = st.slider("Non-inferiority margin (%)", 5.0, 20.0, 10.0, 1.0)
        
        with col2:
            sample_size = st.slider("Sample size per group", 50, 500, 200, 10)
            conf_level = st.slider("Confidence level (%)", 80.0, 99.0, 95.0, 1.0)
        
        if st.button("Simulate Non-inferiority Trial", key="sim_clinical"):
            # Generate simulated data
            np.random.seed(None)
            
            standard_group = np.random.binomial(1, standard_effect/100, sample_size)
            new_group = np.random.binomial(1, new_effect/100, sample_size)
            
            # Calculate observed effects
            observed_standard = np.mean(standard_group) * 100
            observed_new = np.mean(new_group) * 100
            
            # Calculate difference
            observed_diff = observed_new - observed_standard
            
            # Calculate confidence interval for difference
            pooled_se = np.sqrt(observed_new/100 * (1 - observed_new/100)/sample_size + 
                               observed_standard/100 * (1 - observed_standard/100)/sample_size)
            
            z_crit = stats.norm.ppf(1 - (1 - conf_level/100)/2)
            margin = z_crit * pooled_se * 100
            
            ci_lower = observed_diff - margin
            ci_upper = observed_diff + margin
            
            # Determine non-inferiority status
            non_inferior = ci_lower > -non_inferiority_margin
            
            # Create visualization
            fig = go.Figure()
            
            # Add reference line at 0 (no difference)
            fig.add_vline(x=0, line=dict(color='black', width=1))
            
            # Add non-inferiority margin line
            fig.add_vline(x=-non_inferiority_margin, line=dict(color='red', width=2, dash='dash'),
                         annotation=dict(text=f"Non-inferiority margin: -{non_inferiority_margin}%", showarrow=False))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=[ci_lower, ci_upper],
                y=[1, 1],
                mode='lines',
                line=dict(color='blue', width=4),
                name=f"{conf_level}% CI: [{ci_lower:.1f}, {ci_upper:.1f}]"
            ))
            
            # Add point estimate
            fig.add_trace(go.Scatter(
                x=[observed_diff],
                y=[1],
                mode='markers',
                marker=dict(color='blue', size=12),
                name=f"Observed difference: {observed_diff:.1f}%"
            ))
            
            # Update layout
            fig.update_layout(
                title="Non-inferiority Trial Results",
                xaxis_title="Difference in Treatment Effect (New - Standard) in %",
                yaxis=dict(showticklabels=False),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create results cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Standard Treatment Effect",
                    value=f"{observed_standard:.1f}%",
                    delta=f"{observed_standard - standard_effect:.1f}pp from true"
                )
            
            with col2:
                st.metric(
                    label="New Treatment Effect",
                    value=f"{observed_new:.1f}%",
                    delta=f"{observed_new - new_effect:.1f}pp from true"
                )
            
            with col3:
                st.metric(
                    label="Effect Difference",
                    value=f"{observed_diff:.1f}%",
                    delta=f"{observed_diff - (new_effect - standard_effect):.1f}pp from true"
                )
            
            # Interpretation
            if non_inferior:
                st.success(f"**Conclusion**: The new treatment is non-inferior to the standard treatment at the {conf_level}% confidence level (lower bound of {ci_lower:.1f}% > non-inferiority margin of -{non_inferiority_margin}%).")
            else:
                st.error(f"**Conclusion**: Non-inferiority was not demonstrated at the {conf_level}% confidence level (lower bound of {ci_lower:.1f}% < non-inferiority margin of -{non_inferiority_margin}%).")
            
            st.markdown(f"""
            **Interpretation**:
            
            In this non-inferiority trial:
            
            - The observed effect was {observed_diff:.1f}% ({observed_new:.1f}% for new treatment vs. {observed_standard:.1f}% for standard treatment)
            - The {conf_level}% confidence interval for the difference is [{ci_lower:.1f}%, {ci_upper:.1f}%]
            - The non-inferiority margin was set at -{non_inferiority_margin}%
            
            Since the lower bound of the confidence interval is {'above' if non_inferior else 'below'} the non-inferiority margin, we {'can' if non_inferior else 'cannot'} conclude that the new treatment is non-inferior to the standard treatment.
            
            **Key insights**:
            
            1. Even if the point estimate shows the new treatment is slightly worse, we can still conclude non-inferiority if the confidence interval's lower bound doesn't cross the non-inferiority margin
            2. The width of the confidence interval depends on the sample size and variability in the data
            3. The choice of non-inferiority margin is a clinical decision, not a statistical one
            """)
            
            # Provide additional context
            st.markdown("""
            ### Why Use Confidence Intervals Instead of Just p-values?
            
            In clinical trials, confidence intervals provide several advantages over p-values alone:
            
            1. **Effect size estimation**: CIs provide information about the magnitude of the effect, not just whether it exists
            2. **Clinical relevance**: Help determine if the effect is large enough to be clinically meaningful
            3. **Precision assessment**: The width of the interval indicates the precision of the estimate
            4. **Compatibility with equivalence testing**: Directly applicable to non-inferiority and equivalence trials
            5. **Better scientific communication**: Convey more information about uncertainty and potential effect sizes
            
            ### Regulatory Perspective
            
            Both the FDA and EMA emphasize the importance of confidence intervals in clinical trial reporting:
            
            - FDA's guidance often recommends reporting both p-values and confidence intervals
            - EMA's statistical guidelines state: "Confidence intervals should be presented in order to provide information on the size of the treatment effect and the precision of the estimate."
            """)
    
    elif application_type == "A/B Testing":
        st.subheader("Confidence Intervals in A/B Testing")
        
        st.markdown("""
        ### Role of Confidence Intervals in A/B Testing
        
        A/B testing is a randomized experiment method widely used in product development, marketing, and user experience design to compare two versions of a webpage, app feature, or marketing element.
        
        Confidence intervals in A/B testing help:
        
        1. **Quantify uncertainty**: Provide a range of plausible values for the true difference between versions
        2. **Business decision-making**: Determine if observed differences are practically significant
        3. **Risk assessment**: Evaluate the potential best and worst case scenarios
        4. **Test duration planning**: Determine when enough data has been collected to make a decision
        
        ### Common Applications in A/B Testing
        
        #### Conversion Rate Optimization
        
        - **Binary metrics**: Click-through rates, sign-up rates, purchase completion
        - **Continuous metrics**: Revenue per user, time on page, number of page views
        
        #### Sequential Testing
        
        - Confidence intervals updated as new data arrives
        - Stopping rules based on precision (confidence interval width)
        
        #### Multi-armed Bandit Testing
        
        - Confidence intervals guide exploration/exploitation trade-off
        - Upper confidence bound (UCB) algorithms use confidence intervals to decide which variant to show next
        """)
        
        # Interactive A/B test example
        st.subheader("Interactive Example: E-commerce Conversion Rate A/B Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            control_rate = st.slider("Control conversion rate (%)", 1.0, 20.0, 5.0, 0.1)
            variant_rate = st.slider("Variant conversion rate (%)", 1.0, 20.0, 6.0, 0.1)
        
        with col2:
            daily_visitors = st.slider("Daily visitors per variant", 100, 5000, 1000, 100)
            test_days = st.slider("Test duration (days)", 1, 30, 14, 1)
        
        if st.button("Simulate A/B Test", key="sim_ab_test"):
            # Calculate total sample size
            control_size = variant_size = daily_visitors * test_days
            
            # Generate simulated data
            np.random.seed(None)
            
            control_conversions = np.random.binomial(1, control_rate/100, control_size)
            variant_conversions = np.random.binomial(1, variant_rate/100, variant_size)
            
            # Calculate observed conversion rates
            observed_control = np.mean(control_conversions) * 100
            observed_variant = np.mean(variant_conversions) * 100
            
            # Calculate absolute and relative differences
            abs_diff = observed_variant - observed_control
            rel_diff = (observed_variant / observed_control - 1) * 100
            
            # Calculate confidence intervals for absolute difference
            pooled_se = np.sqrt(observed_variant/100 * (1 - observed_variant/100)/variant_size + 
                               observed_control/100 * (1 - observed_control/100)/control_size)
            
            z_crit = stats.norm.ppf(0.975)  # 95% CI
            margin = z_crit * pooled_se * 100
            
            abs_ci_lower = abs_diff - margin
            abs_ci_upper = abs_diff + margin
            
            # Calculate confidence intervals for relative difference
            # Delta method approximation for variance of log ratio
            se_log_ratio = np.sqrt(
                (1 - observed_control/100) / (observed_control/100 * control_size) +
                (1 - observed_variant/100) / (observed_variant/100 * variant_size)
            )
            
            log_ratio_ci_lower = np.log(observed_variant / observed_control) - z_crit * se_log_ratio
            log_ratio_ci_upper = np.log(observed_variant / observed_control) + z_crit * se_log_ratio
            
            rel_ci_lower = (np.exp(log_ratio_ci_lower) - 1) * 100
            rel_ci_upper = (np.exp(log_ratio_ci_upper) - 1) * 100
            
            # Determine statistical significance
            significant = abs_ci_lower > 0 or abs_ci_upper < 0
            
            # Create visualization for absolute difference
            abs_fig = go.Figure()
            
            # Add reference line at 0 (no difference)
            abs_fig.add_vline(x=0, line=dict(color='black', width=1))
            
            # Add confidence interval
            abs_fig.add_trace(go.Scatter(
                x=[abs_ci_lower, abs_ci_upper],
                y=[1, 1],
                mode='lines',
                line=dict(color='blue', width=4),
                name=f"95% CI: [{abs_ci_lower:.2f}%, {abs_ci_upper:.2f}%]"
            ))
            
            # Add point estimate
            abs_fig.add_trace(go.Scatter(
                x=[abs_diff],
                y=[1],
                mode='markers',
                marker=dict(color='blue', size=12),
                name=f"Observed difference: {abs_diff:.2f}%"
            ))
            
            # Update layout
            abs_fig.update_layout(
                title="Absolute Difference in Conversion Rate (Variant - Control)",
                xaxis_title="Difference in Percentage Points",
                yaxis=dict(showticklabels=False),
                height=300
            )
            
            st.plotly_chart(abs_fig, use_container_width=True)
            
            # Create visualization for relative difference
            rel_fig = go.Figure()
            
            # Add reference line at 0 (no difference)
            rel_fig.add_vline(x=0, line=dict(color='black', width=1))
            
            # Add confidence interval
            rel_fig.add_trace(go.Scatter(
                x=[rel_ci_lower, rel_ci_upper],
                y=[1, 1],
                mode='lines',
                line=dict(color='green', width=4),
                name=f"95% CI: [{rel_ci_lower:.2f}%, {rel_ci_upper:.2f}%]"
            ))
            
            # Add point estimate
            rel_fig.add_trace(go.Scatter(
                x=[rel_diff],
                y=[1],
                mode='markers',
                marker=dict(color='green', size=12),
                name=f"Observed lift: {rel_diff:.2f}%"
            ))
            
            # Update layout
            rel_fig.update_layout(
                title="Relative Difference in Conversion Rate (% Lift)",
                xaxis_title="Percentage Change",
                yaxis=dict(showticklabels=False),
                height=300
            )
            
            st.plotly_chart(rel_fig, use_container_width=True)
            
            # Create results cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Control Conversion Rate",
                    value=f"{observed_control:.2f}%",
                    delta=f"{observed_control - control_rate:.2f}pp from expected"
                )
            
            with col2:
                st.metric(
                    label="Variant Conversion Rate",
                    value=f"{observed_variant:.2f}%",
                    delta=f"{observed_variant - variant_rate:.2f}pp from expected"
                )
            
            with col3:
                st.metric(
                    label="Lift (Relative Difference)",
                    value=f"{rel_diff:.2f}%",
                    delta=f"{'Significant' if significant else 'Not significant'}"
                )
            
            # Additional metrics
            samples_per_day = daily_visitors * 2  # both variants
            conversions_per_day = samples_per_day * ((observed_control + observed_variant) / 200)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Sample Size",
                    value=f"{control_size + variant_size:,}"
                )
            
            with col2:
                st.metric(
                    label="Total Conversions",
                    value=f"{sum(control_conversions) + sum(variant_conversions):,}"
                )
            
            with col3:
                st.metric(
                    label="Statistical Power",
                    value=f"{(1 if significant else 0) * 100:.0f}%"
                )
            
            # Interpretation
            if significant and abs_diff > 0:
                st.success(f"**Conclusion**: The variant shows a statistically significant improvement with 95% confidence. The conversion rate increased by {abs_diff:.2f} percentage points ({rel_diff:.2f}% lift).")
            elif significant and abs_diff < 0:
                st.error(f"**Conclusion**: The variant shows a statistically significant decrease with 95% confidence. The conversion rate decreased by {abs(abs_diff):.2f} percentage points ({abs(rel_diff):.2f}% drop).")
            else:
                st.warning(f"**Conclusion**: The test is inconclusive at the 95% confidence level. We cannot determine whether the variant is better or worse than the control.")
            
            # Add specific business impact calculation
            avg_order_value = 50  # Assuming average order value of $50
            monthly_visitors = daily_visitors * 30
            
            control_revenue = monthly_visitors * (observed_control/100) * avg_order_value
            variant_revenue = monthly_visitors * (observed_variant/100) * avg_order_value
            revenue_diff = variant_revenue - control_revenue
            
            revenue_lower = monthly_visitors * ((observed_control + abs_ci_lower)/100) * avg_order_value - control_revenue
            revenue_upper = monthly_visitors * ((observed_control + abs_ci_upper)/100) * avg_order_value - control_revenue
            
            st.markdown(f"""
            ### Business Impact Analysis
            
            Assuming an average order value of ${avg_order_value} and {monthly_visitors:,} monthly visitors:
            
            - **Projected monthly revenue with control**: ${control_revenue:,.2f}
            - **Projected monthly revenue with variant**: ${variant_revenue:,.2f}
            - **Projected monthly revenue increase**: ${revenue_diff:,.2f}
            - **95% confidence interval for revenue impact**: [${revenue_lower:,.2f}, ${revenue_upper:,.2f}]
            
            **What this means for the business**:
            
            With 95% confidence, implementing the variant would result in a monthly revenue change between ${revenue_lower:,.2f} and ${revenue_upper:,.2f}.
            
            **Recommendation**:
            
            {"Implement the variant, as there is a statistically significant improvement in conversion rate that translates to increased revenue." if significant and abs_diff > 0 else
             "Keep the control, as the variant shows a statistically significant decrease in conversion rate that would result in lost revenue." if significant and abs_diff < 0 else
             "Consider running the test longer to gather more data, as the current results are inconclusive. Alternatively, analyze segments to see if there are specific user groups for whom the variant performs better."}
            """)
            
            # Provide additional context on sample size and test duration
            daily_conv_control = daily_visitors * (observed_control/100)
            daily_conv_variant = daily_visitors * (observed_variant/100)
            mde = 2 * z_crit * np.sqrt((observed_control/100) * (1 - observed_control/100) / daily_visitors)
            days_for_mde = (2 * z_crit / abs_diff * 100)**2 * ((observed_control/100) * (1 - observed_control/100) + (observed_variant/100) * (1 - observed_variant/100)) / daily_visitors
            
            st.markdown(f"""
            ### Test Duration Analysis
            
            With your current traffic levels and conversion rates:
            
            - **Daily control conversions**: ~{daily_conv_control:.1f}
            - **Daily variant conversions**: ~{daily_conv_variant:.1f}
            - **Detectable difference at current sample size (per day)**: ±{mde*100:.2f} percentage points
            - **Days needed for current observed difference to be significant**: {days_for_mde:.1f} days
            
            {"Your test has reached statistical significance within the planned duration." if significant else
             f"Your test would need approximately {days_for_mde:.1f} days to reach statistical significance for the observed difference of {abs(abs_diff):.2f} percentage points."}
            """)
    
    elif application_type == "Environmental Monitoring":
        st.subheader("Confidence Intervals in Environmental Monitoring")
        
        st.markdown("""
        ### Role of Confidence Intervals in Environmental Monitoring
        
        Environmental monitoring involves measuring and assessing environmental parameters over time to identify trends, ensure compliance with regulations, and protect ecosystems and public health.
        
        Confidence intervals in environmental monitoring help:
        
        1. **Quantify uncertainty**: Account for measurement error and natural variability
        2. **Compliance assessment**: Determine if pollutant levels exceed regulatory thresholds
        3. **Trend analysis**: Evaluate whether environmental conditions are improving or degrading
        4. **Spatial interpolation**: Estimate conditions at unmonitored locations
        
        ### Common Applications in Environmental Science
        
        #### Air Quality Monitoring
        
        - Confidence intervals for mean pollutant concentrations (PM2.5, NO2, O3)
        - Comparison with air quality standards
        - Seasonal and annual averages with uncertainty bounds
        
        #### Water Quality Assessment
        
        - Concentration intervals for contaminants in drinking water, lakes, or rivers
        - Estimating compliance with water quality standards
        - Uncertainty in bioaccumulation and toxicity estimates
        
        #### Climate Change Research
        
        - Confidence intervals for temperature and precipitation trends
        - Uncertainty quantification in climate model projections
        - Ranges for greenhouse gas emission estimates
        """)
        
        # Interactive water quality example
        st.subheader("Interactive Example: Water Quality Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            contaminant = st.selectbox(
                "Select contaminant to monitor",
                ["Lead (Pb)", "Nitrate (NO₃)", "E. coli", "Arsenic (As)"]
            )
            
            if contaminant == "Lead (Pb)":
                true_level = st.slider("True contaminant level (ppb)", 0.0, 25.0, 12.0, 0.5)
                regulatory_limit = 15.0  # EPA limit for lead in ppb
                unit = "ppb"
            elif contaminant == "Nitrate (NO₃)":
                true_level = st.slider("True contaminant level (mg/L)", 0.0, 15.0, 8.0, 0.5)
                regulatory_limit = 10.0  # EPA limit for nitrate in mg/L
                unit = "mg/L"
            elif contaminant == "E. coli":
                true_level = st.slider("True contaminant level (CFU/100mL)", 0.0, 300.0, 120.0, 10.0)
                regulatory_limit = 126.0  # EPA limit for E. coli in CFU/100mL
                unit = "CFU/100mL"
            elif contaminant == "Arsenic (As)":
                true_level = st.slider("True contaminant level (ppb)", 0.0, 15.0, 7.0, 0.5)
                regulatory_limit = 10.0  # EPA limit for arsenic in ppb
                unit = "ppb"
        
        with col2:
            n_samples = st.slider("Number of samples collected", 3, 30, 10, 1)
            measurement_error = st.slider("Measurement error (%)", 5, 30, 15, 5)
            conf_level = st.slider("Confidence level (%)", 80, 99, 95, 1)
        
        if st.button("Simulate Water Quality Monitoring", key="sim_water_quality"):
            # Generate simulated measurements with error
            np.random.seed(None)
            
            # Convert percentage error to absolute standard deviation
            error_sd = true_level * (measurement_error / 100)
            
            # Generate measurements with error
            measurements = np.random.normal(true_level, error_sd, n_samples)
            
            # Ensure no negative values for physical measurements
            measurements = np.maximum(measurements, 0)
            
            # Calculate sample statistics
            sample_mean = np.mean(measurements)
            sample_sd = np.std(measurements, ddof=1)
            
            # Calculate confidence interval
            t_crit = stats.t.ppf(1 - (1 - conf_level/100)/2, n_samples - 1)
            margin = t_crit * sample_sd / np.sqrt(n_samples)
            
            ci_lower = sample_mean - margin
            ci_upper = sample_mean + margin
            
            # Determine compliance status
            compliant_lower = ci_upper < regulatory_limit  # Definitely compliant
            compliant_upper = ci_lower > regulatory_limit  # Definitely non-compliant
            inconclusive = not (compliant_lower or compliant_upper)  # CI crosses limit
            
            # Create visualization
            fig = go.Figure()
            
            # Add individual measurements
            fig.add_trace(go.Scatter(
                x=np.arange(1, n_samples + 1),
                y=measurements,
                mode='markers',
                name='Individual Samples',
                marker=dict(size=10, color='blue', opacity=0.6)
            ))
            
            # Add mean line
            fig.add_hline(y=sample_mean, line=dict(color='blue', width=2),
                         annotation=dict(text=f"Sample Mean: {sample_mean:.1f} {unit}", showarrow=False))
            
            # Add true level line
            fig.add_hline(y=true_level, line=dict(color='green', width=2, dash='dash'),
                         annotation=dict(text=f"True Level: {true_level:.1f} {unit}", showarrow=False))
            
            # Add regulatory limit line
            fig.add_hline(y=regulatory_limit, line=dict(color='red', width=2),
                         annotation=dict(text=f"Regulatory Limit: {regulatory_limit:.1f} {unit}", showarrow=False))
            
            # Update layout
            fig.update_layout(
                title=f"{contaminant} Measurements Across {n_samples} Samples",
                xaxis_title="Sample Number",
                yaxis_title=f"Concentration ({unit})",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create confidence interval visualization
            ci_fig = go.Figure()
            
            # Add regulatory limit line
            ci_fig.add_hline(y=regulatory_limit, line=dict(color='red', width=2),
                           annotation=dict(text=f"Regulatory Limit: {regulatory_limit:.1f} {unit}", showarrow=False))
            
            # Add confidence interval
            ci_fig.add_trace(go.Scatter(
                x=[0.5, 1.5],
                y=[sample_mean, sample_mean],
                mode='lines',
                line=dict(color='blue', width=4),
                name=f"Sample Mean: {sample_mean:.1f} {unit}"
            ))
            
            ci_fig.add_trace(go.Scatter(
                x=[1, 1],
                y=[ci_lower, ci_upper],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f"{conf_level}% CI: [{ci_lower:.1f}, {ci_upper:.1f}] {unit}"
            ))
            
            # Add error bars for visual clarity
            ci_fig.add_trace(go.Scatter(
                x=[0.9, 1.1],
                y=[ci_lower, ci_lower],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            
            ci_fig.add_trace(go.Scatter(
                x=[0.9, 1.1],
                y=[ci_upper, ci_upper],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            
            # Update layout
            ci_fig.update_layout(
                title=f"{conf_level}% Confidence Interval for Mean {contaminant} Concentration",
                xaxis=dict(
                    showticklabels=False,
                    range=[0, 2]
                ),
                yaxis_title=f"Concentration ({unit})",
                height=400
            )
            
            st.plotly_chart(ci_fig, use_container_width=True)
            
            # Create results cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"Sample Mean ({unit})",
                    value=f"{sample_mean:.1f}",
                    delta=f"{sample_mean - true_level:.1f} from true level"
                )
            
            with col2:
                st.metric(
                    label="Sample Std Dev",
                    value=f"{sample_sd:.2f}"
                )
            
            with col3:
                if compliant_lower:
                    status = "Compliant"
                    delta_color = "normal"
                elif compliant_upper:
                    status = "Non-compliant"
                    delta_color = "inverse"
                else:
                    status = "Inconclusive"
                    delta_color = "off"
                
                st.metric(
                    label="Compliance Status",
                    value=status,
                    delta=f"{abs(sample_mean - regulatory_limit):.1f} {unit} {'below' if sample_mean < regulatory_limit else 'above'} limit",
                    delta_color=delta_color
                )
            
            # Interpretation
            if compliant_lower:
                st.success(f"**Conclusion**: With {conf_level}% confidence, the mean {contaminant} concentration is below the regulatory limit of {regulatory_limit} {unit}. The site is in compliance.")
            elif compliant_upper:
                st.error(f"**Conclusion**: With {conf_level}% confidence, the mean {contaminant} concentration exceeds the regulatory limit of {regulatory_limit} {unit}. The site is not in compliance.")
            else:
                st.warning(f"**Conclusion**: The compliance status is inconclusive at the {conf_level}% confidence level. The confidence interval [{ci_lower:.1f}, {ci_upper:.1f}] {unit} includes the regulatory limit of {regulatory_limit} {unit}.")
            
            # Add statistical power analysis
            effect_size = abs(true_level - regulatory_limit) / sample_sd
            power = 1 - stats.t.cdf(t_crit - effect_size * np.sqrt(n_samples), n_samples - 1) + stats.t.cdf(-t_crit - effect_size * np.sqrt(n_samples), n_samples - 1)
            
            samples_needed = int(np.ceil((2 * (t_crit)/ effect_size)**2))
            
            st.markdown(f"""
            ### Statistical Analysis and Monitoring Design
            
            **Current sampling plan assessment**:
            
            - **Statistical power**: {power*100:.1f}% probability of detecting a true exceedance
            - **Effect size (standardized)**: {effect_size:.2f}
            - **Minimum detectable difference**: ±{margin:.2f} {unit} at {conf_level}% confidence
            
            **Sampling recommendations**:
            
            - **Samples needed for 80% power**: Approximately {samples_needed} samples
            - **Margin of error reduction**: {"Additional sampling would reduce the margin of error and could help resolve the compliance status." if inconclusive else "Current sampling is sufficient to determine compliance status."}
            
            **Key monitoring insights**:
            
            1. **Measurement variability**: The observed standard deviation ({sample_sd:.2f} {unit}) accounts for both natural variability and measurement error
            2. **Compliance buffer**: {"Consider implementing additional treatment or controls to maintain a safety margin below the regulatory limit." if sample_mean > regulatory_limit * 0.8 else "Current levels provide adequate buffer below regulatory limits."}
            3. **Temporal considerations**: Regular monitoring is recommended to capture seasonal or temporal variations
            """)
            
            # Add information about the selected contaminant
            if contaminant == "Lead (Pb)":
                health_effects = "Developmental delays in children, kidney problems, high blood pressure"
                sources = "Old plumbing, lead service lines, lead solder, natural deposits"
                treatment = "Corrosion control, pH adjustment, lead service line replacement"
            elif contaminant == "Nitrate (NO₃)":
                health_effects = "Blue baby syndrome (methemoglobinemia), oxygen depletion in infants"
                sources = "Agricultural runoff, fertilizers, septic systems, natural deposits"
                treatment = "Ion exchange, reverse osmosis, distillation"
            elif contaminant == "E. coli":
                health_effects = "Gastrointestinal illness, diarrhea, cramps, nausea"
                sources = "Human or animal fecal waste contamination"
                treatment = "Disinfection (chlorination, UV), filtration"
            elif contaminant == "Arsenic (As)":
                health_effects = "Skin damage, circulatory problems, increased cancer risk"
                sources = "Natural deposits, industrial and agricultural pollution"
                treatment = "Oxidation/filtration, adsorption, ion exchange, reverse osmosis"
            
            st.markdown(f"""
            ### {contaminant} Information
            
            **Regulatory limit**: {regulatory_limit} {unit} (EPA Maximum Contaminant Level)
            
            **Health effects**: {health_effects}
            
            **Common sources**: {sources}
            
            **Treatment options**: {treatment}
            """)
    
    elif application_type == "Manufacturing Quality Control":
        st.subheader("Confidence Intervals in Manufacturing Quality Control")
        
        st.markdown("""
        ### Role of Confidence Intervals in Manufacturing Quality Control
        
        Quality control in manufacturing involves monitoring and improving production processes to ensure products meet specifications and quality standards.
        
        Confidence intervals in manufacturing quality control help:
        
        1. **Process capability analysis**: Determine if a process can consistently meet specifications
        2. **Statistical process control**: Identify when a process is out of control
        3. **Acceptance sampling**: Make decisions about accepting or rejecting product lots
        4. **Measurement system analysis**: Assess the precision and accuracy of measurement systems
        
        ### Common Applications in Manufacturing
        
        #### Process Capability Studies
        
        - Confidence intervals for process capability indices (Cp, Cpk)
        - Estimation of process mean and standard deviation
        - Determining if process is capable of meeting specifications
        
        #### Statistical Process Control (SPC)
        
        - Control limits as a form of prediction intervals
        - Confidence intervals for process parameters on control charts
        - Trend analysis and process drift detection
        
        #### Reliability Analysis
        
        - Confidence intervals for mean time between failures (MTBF)
        - Estimation of product lifetime and failure rates
        - Warranty period determination
        """)
        
        # Interactive process capability example
        st.subheader("Interactive Example: Process Capability Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            spec_lower = st.slider("Lower specification limit", 0.0, 10.0, 3.0, 0.1)
            spec_upper = st.slider("Upper specification limit", 10.0, 20.0, 17.0, 0.1)
            process_target = st.slider("Process target", spec_lower, spec_upper, (spec_lower + spec_upper) / 2, 0.1)
        
        with col2:
            process_mean = st.slider("Actual process mean", spec_lower - 2, spec_upper + 2, process_target, 0.1)
            process_sd = st.slider("Process standard deviation", 0.1, 3.0, 1.0, 0.1)
            sample_size = st.slider("Sample size", 10, 200, 50, 5)
        
        if st.button("Analyze Process Capability", key="analyze_capability"):
            # Generate sample data
            np.random.seed(None)
            
            measurements = np.random.normal(process_mean, process_sd, sample_size)
            
            # Calculate sample statistics
            sample_mean = np.mean(measurements)
            sample_sd = np.std(measurements, ddof=1)
            
            # Calculate confidence intervals for mean
            t_crit = stats.t.ppf(0.975, sample_size - 1)
            mean_margin = t_crit * sample_sd / np.sqrt(sample_size)
            
            mean_ci_lower = sample_mean - mean_margin
            mean_ci_upper = sample_mean + mean_margin
            
            # Calculate confidence intervals for standard deviation
            chi2_lower = stats.chi2.ppf(0.025, sample_size - 1)
            chi2_upper = stats.chi2.ppf(0.975, sample_size - 1)
            
            sd_ci_lower = sample_sd * np.sqrt((sample_size - 1) / chi2_upper)
            sd_ci_upper = sample_sd * np.sqrt((sample_size - 1) / chi2_lower)
            
            # Calculate process capability indices
            spec_width = spec_upper - spec_lower
            cp = spec_width / (6 * sample_sd)
            
            # Calculate Cpk
            cpu = (spec_upper - sample_mean) / (3 * sample_sd)
            cpl = (sample_mean - spec_lower) / (3 * sample_sd)
            cpk = min(cpu, cpl)
            
            # Calculate confidence intervals for Cp
            cp_ci_lower = cp * np.sqrt((sample_size - 1) / chi2_upper)
            cp_ci_upper = cp * np.sqrt((sample_size - 1) / chi2_lower)
            
            # Calculate prediction interval for individual future observations
            pred_margin = t_crit * sample_sd * np.sqrt(1 + 1/sample_size)
            pred_lower = sample_mean - pred_margin
            pred_upper = sample_mean + pred_margin
            
            # Calculate estimated process yield
            z_upper = (spec_upper - sample_mean) / sample_sd
            z_lower = (sample_mean - spec_lower) / sample_sd
            
            yield_upper = stats.norm.cdf(z_upper)
            yield_lower = stats.norm.cdf(z_lower)
            
            process_yield = (yield_upper - (1 - yield_lower)) * 100
            
            # Calculate DPMO (Defects Per Million Opportunities)
            dpmo = (1 - (yield_upper - (1 - yield_lower))) * 1000000
            
            # Determine Six Sigma level
            if dpmo > 0:
                sigma_level = 0.8406 + np.sqrt(0.8406**2 - (np.log10(dpmo) - 9.654)/3.8)
            else:
                sigma_level = 6.0
            
            # Create visualization
            fig = go.Figure()
            
            # Create x-range for normal distribution
            x_range = np.linspace(
                min(measurements.min(), spec_lower) - 2 * sample_sd,
                max(measurements.max(), spec_upper) + 2 * sample_sd,
                1000
            )
            
            # Add process distribution
            y_norm = stats.norm.pdf(x_range, sample_mean, sample_sd)
            y_norm_scaled = y_norm / y_norm.max() * 10  # Scale for visualization
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y_norm_scaled,
                mode='lines',
                name='Process Distribution',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ))
            
            # Add specification limits
            fig.add_vline(x=spec_lower, line=dict(color='red', width=2),
                         annotation=dict(text=f"LSL: {spec_lower}", showarrow=False))
            
            fig.add_vline(x=spec_upper, line=dict(color='red', width=2),
                         annotation=dict(text=f"USL: {spec_upper}", showarrow=False))
            
            # Add process mean
            fig.add_vline(x=sample_mean, line=dict(color='green', width=2),
                         annotation=dict(text=f"Mean: {sample_mean:.2f}", showarrow=False))
            
            # Add target
            fig.add_vline(x=process_target, line=dict(color='purple', width=2, dash='dash'),
                         annotation=dict(text=f"Target: {process_target}", showarrow=False))
            
            # Add individual measurements
            fig.add_trace(go.Scatter(
                x=measurements, y=np.zeros_like(measurements),
                mode='markers',
                name='Measurements',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # Add out-of-spec measurements in different color
            out_of_spec = (measurements < spec_lower) | (measurements > spec_upper)
            if np.any(out_of_spec):
                fig.add_trace(go.Scatter(
                    x=measurements[out_of_spec], y=np.zeros_like(measurements[out_of_spec]),
                    mode='markers',
                    name='Out of Spec',
                    marker=dict(color='red', size=8, opacity=0.6)
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Process Distribution and Specifications (Cp = {cp:.2f}, Cpk = {cpk:.2f})",
                xaxis_title="Measurement",
                yaxis=dict(showticklabels=False),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create confidence interval visualization
            ci_fig = go.Figure()
            
            # Add mean confidence interval
            ci_fig.add_trace(go.Scatter(
                x=[0.5, 1.5],
                y=[sample_mean, sample_mean],
                mode='lines',
                line=dict(color='blue', width=4),
                name=f"Sample Mean: {sample_mean:.2f}"
            ))
            
            ci_fig.add_trace(go.Scatter(
                x=[1, 1],
                y=[mean_ci_lower, mean_ci_upper],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f"95% CI for Mean: [{mean_ci_lower:.2f}, {mean_ci_upper:.2f}]"
            ))
            
            # Add horizontal error bars
            ci_fig.add_trace(go.Scatter(
                x=[0.9, 1.1],
                y=[mean_ci_lower, mean_ci_lower],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            
            ci_fig.add_trace(go.Scatter(
                x=[0.9, 1.1],
                y=[mean_ci_upper, mean_ci_upper],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            
            # Add prediction interval
            ci_fig.add_trace(go.Scatter(
                x=[2.5, 3.5],
                y=[(pred_lower + pred_upper)/2, (pred_lower + pred_upper)/2],
                mode='lines',
                line=dict(color='green', width=4),
                name=f"Prediction Interval Center"
            ))
            
            ci_fig.add_trace(go.Scatter(
                x=[3, 3],
                y=[pred_lower, pred_upper],
                mode='lines',
                line=dict(color='green', width=2),
                name=f"95% Prediction Interval: [{pred_lower:.2f}, {pred_upper:.2f}]"
            ))
            
            # Add horizontal error bars
            ci_fig.add_trace(go.Scatter(
                x=[2.9, 3.1],
                y=[pred_lower, pred_lower],
                mode='lines',
                line=dict(color='green', width=2),
                showlegend=False
            ))
            
            ci_fig.add_trace(go.Scatter(
                x=[2.9, 3.1],
                y=[pred_upper, pred_upper],
                mode='lines',
                line=dict(color='green', width=2),
                showlegend=False
            ))
            
            # Add specification limits
            ci_fig.add_hline(y=spec_lower, line=dict(color='red', width=2, dash='dash'),
                           annotation=dict(text=f"LSL: {spec_lower}", showarrow=False))
            
            ci_fig.add_hline(y=spec_upper, line=dict(color='red', width=2, dash='dash'),
                           annotation=dict(text=f"USL: {spec_upper}", showarrow=False))
            
            # Update layout
            ci_fig.update_layout(
                title="Confidence Interval for Mean and Prediction Interval for Future Observations",
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1, 3],
                    ticktext=['Process Mean', 'Future Observation'],
                    range=[0, 4]
                ),
                yaxis_title="Measurement",
                height=400
            )
            
            st.plotly_chart(ci_fig, use_container_width=True)
            
            # Create capability indices visualization
            cap_fig = go.Figure()
            
            # Add Cp confidence interval
            cap_fig.add_trace(go.Scatter(
                x=[0.5, 1.5],
                y=[cp, cp],
                mode='lines',
                line=dict(color='blue', width=4),
                name=f"Cp: {cp:.2f}"
            ))
            
            cap_fig.add_trace(go.Scatter(
                x=[1, 1],
                y=[cp_ci_lower, cp_ci_upper],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f"95% CI for Cp: [{cp_ci_lower:.2f}, {cp_ci_upper:.2f}]"
            ))
            
            # Add horizontal error bars
            cap_fig.add_trace(go.Scatter(
                x=[0.9, 1.1],
                y=[cp_ci_lower, cp_ci_lower],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            
            cap_fig.add_trace(go.Scatter(
                x=[0.9, 1.1],
                y=[cp_ci_upper, cp_ci_upper],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            
            # Add Cpk value
            cap_fig.add_trace(go.Scatter(
                x=[2.5, 3.5],
                y=[cpk, cpk],
                mode='lines',
                line=dict(color='green', width=4),
                name=f"Cpk: {cpk:.2f}"
            ))
            
            # Add capability thresholds
            cap_fig.add_hline(y=1.33, line=dict(color='green', width=2, dash='dash'),
                            annotation=dict(text="Capable (Cp/Cpk ≥ 1.33)", showarrow=False))
            
            cap_fig.add_hline(y=1.0, line=dict(color='orange', width=2, dash='dash'),
                            annotation=dict(text="Marginally Capable (Cp/Cpk ≥ 1.0)", showarrow=False))
            
            # Update layout
            cap_fig.update_layout(
                title="Process Capability Indices",
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1, 3],
                    ticktext=['Cp (Process Potential)', 'Cpk (Process Performance)'],
                    range=[0, 4]
                ),
                yaxis_title="Capability Index Value",
                height=400
            )
            
            st.plotly_chart(cap_fig, use_container_width=True)
            
            # Create results cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Process Capability (Cp)",
                    value=f"{cp:.2f}",
                    delta=f"{'Capable' if cp >= 1.33 else 'Marginally Capable' if cp >= 1.0 else 'Not Capable'}"
                )
            
            with col2:
                st.metric(
                    label="Process Performance (Cpk)",
                    value=f"{cpk:.2f}",
                    delta=f"{'Capable' if cpk >= 1.33 else 'Marginally Capable' if cpk >= 1.0 else 'Not Capable'}"
                )
            
            with col3:
                st.metric(
                    label="Process Yield",
                    value=f"{process_yield:.2f}%",
                    delta=f"{sigma_level:.2f} sigma"
                )
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Defects Per Million (DPMO)",
                    value=f"{dpmo:.0f}"
                )
            
            with col2:
                st.metric(
                    label="Standard Deviation (95% CI)",
                    value=f"{sample_sd:.3f}",
                    delta=f"[{sd_ci_lower:.3f}, {sd_ci_upper:.3f}]"
                )
            
            with col3:
                dpm_target = 3.4 if sigma_level < 6 else dpmo
                st.metric(
                    label="Distance to Six Sigma",
                    value=f"{abs(6 - sigma_level):.2f} σ",
                    delta=f"{dpmo - dpm_target:,.0f} DPMO reduction needed"
                )
            
            # Interpretation and recommendations
            if cpk >= 1.33:
                capability_status = "capable"
                recommendation = "Maintain current process controls. Consider reducing inspection frequency or implementing process monitoring."
            elif cpk >= 1.0:
                capability_status = "marginally capable"
                recommendation = "Implement statistical process control (SPC) to monitor for process shifts. Consider process improvement initiatives to increase Cpk."
            else:
                capability_status = "not capable"
                recommendation = "Process improvement is required. Focus on reducing variation and/or centering the process mean between specification limits."
            
            # Centering issue
            if abs(sample_mean - process_target) > sample_sd:
                centering = "The process is not well-centered. Adjusting the process mean closer to the target could significantly improve capability."
            else:
                centering = "The process is reasonably well-centered."
            
            # Variation issue
            if cp < 1.0:
                variation = "Process variation is too high relative to specification width. Reducing variation should be a priority."
            else:
                variation = "Process variation is acceptable relative to specification width."
            
            st.markdown(f"""
            ### Process Capability Analysis Summary
            
            **Process Status**: The process is **{capability_status}** of meeting specifications.
            
            **Key Findings**:
            
            1. **Centering**: {centering}
            2. **Variation**: {variation}
            3. **Yield**: The estimated process yield is {process_yield:.2f}%, which corresponds to a {sigma_level:.2f} sigma level.
            4. **Defect Rate**: The process is producing approximately {dpmo:.0f} defects per million opportunities.
            
            **Recommendations**:
            
            {recommendation}
            
            **Statistical Considerations**:
            
            1. The 95% confidence interval for the process mean is [{mean_ci_lower:.2f}, {mean_ci_upper:.2f}]
            2. The 95% confidence interval for Cp is [{cp_ci_lower:.2f}, {cp_ci_upper:.2f}]
            3. With 95% confidence, future individual measurements will fall within [{pred_lower:.2f}, {pred_upper:.2f}]
            
            **Business Impact**:
            
            {"The process is producing too many defects, which will lead to increased inspection costs, scrap, rework, and potential customer returns." if process_yield < 99 else "The process is producing acceptable quality levels, minimizing costs related to scrap, rework, and customer returns."}
            """)
            
            # Educational content on process capability
            st.markdown("""
            ### Understanding Process Capability
            
            **Key Process Capability Indices**:
            
            - **Cp (Process Potential)**: Measures the potential capability of the process based only on process variation
              - Cp = (USL - LSL) / (6 * σ)
              - Ignores process centering
            
            - **Cpk (Process Performance)**: Measures the actual capability of the process, accounting for both variation and centering
              - Cpk = min[(USL - μ) / (3 * σ), (μ - LSL) / (3 * σ)]
              - Always less than or equal to Cp
            
            **Capability Thresholds**:
            
            | Capability Index | Interpretation | Process Yield |
            |------------------|----------------|--------------|
            | Cpk < 1.0        | Not capable    | < 99.73%     |
            | 1.0 ≤ Cpk < 1.33 | Marginally capable | 99.73% - 99.99% |
            | 1.33 ≤ Cpk < 1.67 | Capable       | 99.99% - 99.9997% |
            | Cpk ≥ 1.67       | Highly capable | > 99.9997%  |
            
            **Six Sigma Level**:
            
            The sigma level describes how many standard deviations (σ) fit between the process mean and the nearest specification limit:
            
            | Sigma Level | Defects Per Million | Process Yield |
            |-------------|---------------------|---------------|
            | 3σ          | 66,807              | 93.32%        |
            | 4σ          | 6,210               | 99.38%        |
            | 5σ          | 233                 | 99.977%       |
            | 6σ          | 3.4                 | 99.9997%      |
            
            **Role of Confidence Intervals**:
            
            Confidence intervals quantify the uncertainty in capability estimates due to sampling error. They help determine:
            
            1. Whether a process is truly capable based on limited data
            2. How many samples are needed to make reliable capability assessments
            3. The risk of making incorrect decisions about process acceptability
            """)

# References & Resources Module
elif nav == "References & Resources":
    st.header("References & Resources")
    
    st.markdown("""
    ### Textbooks
    
    - Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury Press.
    - Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
    - DeGroot, M. H., & Schervish, M. J. (2012). *Probability and Statistics* (4th ed.). Pearson.
    - Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
    - Agresti, A., & Coull, B. A. (1998). Approximate is better than "exact" for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119-126.
    
    ### Research Papers
    
    - Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209-212.
    - Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404-413.
    - Neyman, J. (1937). Outline of a theory of statistical estimation based on the classical theory of probability. *Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 236(767), 333-380.
    - Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. *Biometrika*, 34(1/2), 28-35.
    - Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101-133.
    
    ### Online Resources
    
    - [Statistical Inference Course by Johns Hopkins University on Coursera](https://www.coursera.org/learn/statistical-inference)
    - [Confidence Intervals (Yale University)](http://www.stat.yale.edu/Courses/1997-98/101/confint.htm)
    - [Understanding Confidence Intervals (Institute for Digital Research and Education, UCLA)](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-the-differences-between-one-tailed-and-two-tailed-tests/)
    - [NIST/SEMATECH e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/)
    - [Statistical Engineering Division at NIST](https://www.nist.gov/itl/sed)
    
    ### Software Tools
    
    - R packages: `stats`, `boot`, `MASS`, `binom`
    - Python libraries: `scipy.stats`, `statsmodels`, `bootstrapped`
    - Specialized tools: `SAS`, `SPSS`, `Minitab`
    
    ### Interactive Tutorials
    
    - [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/frequentist-inference/index.html)
    - [Interpreting Confidence Intervals (WISE Project)](https://wise.cgu.edu/)
    - [Confidence Intervals (StatKey)](http://www.lock5stat.com/StatKey/)
    
    ### Additional Learning Materials
    
    - [Khan Academy - Confidence Intervals](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample)
    - [OpenIntro Statistics](https://www.openintro.org/book/os/)
    - [StatQuest with Josh Starmer (YouTube Channel)](https://www.youtube.com/c/joshstarmer)
    """)
    
    st.markdown("""
    ### About This App
    
    This app was created as a comprehensive educational tool to help understand confidence intervals and their applications. It includes:
    
    - **Theoretical foundations**: Mathematical derivations and properties of confidence intervals
    - **Interactive simulations**: Visualizations to build intuition about sampling distributions and interval behavior
    - **Advanced methods**: Exploration of non-standard techniques for complex scenarios
    - **Real-world applications**: Practical examples from various domains
    
    The app is designed for students, researchers, and practitioners at different levels of statistical knowledge, from beginners to advanced users.
    
    All simulations and visualizations are generated on-the-fly using Streamlit, Plotly, NumPy, SciPy, and other Python libraries.
    """)
    
    st.markdown("""
    ### Acknowledgments
    
    Special thanks to the open source communities behind:
    
    - [Streamlit](https://streamlit.io/)
    - [Plotly](https://plotly.com/)
    - [NumPy](https://numpy.org/)
    - [SciPy](https://scipy.org/)
    - [pandas](https://pandas.pydata.org/)
    - [Statsmodels](https://www.statsmodels.org/)
    
    And to the countless researchers and educators who have contributed to the development and understanding of confidence intervals over the past century.
    """)