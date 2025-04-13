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

st.set_page_config(
    page_title="Advanced Methods - CI Explorer",
    page_icon="📊",
    layout="wide",
)

# Custom CSS
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
            
            # Add vertical line for true mean (without annotation)
            fig.add_vline(x=true_mean, line=dict(color='black', width=2, dash='dash'))
            fig.add_annotation(
                text=f"True Mean: {true_mean}", 
                x=true_mean, 
                y=1.05, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for sample mean (without annotation)
            fig.add_vline(x=sample_mean, line=dict(color='red', width=2))
            fig.add_annotation(
                text=f"Sample Mean: {sample_mean:.4f}", 
                x=sample_mean, 
                y=0.95, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for prior mean if applicable (without annotation)
            if prior_type != "Non-informative":
                fig.add_vline(x=prior_mean, line=dict(color='green', width=2, dash='dash'))
                fig.add_annotation(
                    text=f"Prior Mean: {prior_mean}", 
                    x=prior_mean, 
                    y=0.85, 
                    xref="x", 
                    yref="paper", 
                    showarrow=False
                )

            # Add vertical line for posterior mean/mode (without annotation)
            fig.add_vline(x=post_mean, line=dict(color='blue', width=2))
            fig.add_annotation(
                text=f"Posterior Mean: {post_mean:.4f}", 
                x=post_mean, 
                y=0.75, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )


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
            
            # Add vertical line for true mean (without annotation)
            int_fig.add_vline(x=true_mean, line=dict(color='black', width=2, dash='dash'))

            # Add annotation separately
            int_fig.add_annotation(
                text=f"True Mean: {true_mean}", 
                x=true_mean, 
                y=1.05, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )
            
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

            # Add vertical line for true proportion (without annotation)
            fig.add_vline(x=true_prop, line=dict(color='black', width=2, dash='dash'))
            fig.add_annotation(
                text=f"True Prop: {true_prop}", 
                x=true_prop, 
                y=1.05, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for sample proportion (without annotation)
            fig.add_vline(x=sample_prop, line=dict(color='red', width=2))
            fig.add_annotation(
                text=f"Sample Prop: {sample_prop:.4f}", 
                x=sample_prop, 
                y=0.95, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for posterior mean (without annotation)
            fig.add_vline(x=post_mean, line=dict(color='blue', width=2))
            fig.add_annotation(
                text=f"Posterior Mean: {post_mean:.4f}", 
                x=post_mean, 
                y=0.85, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for posterior mode if applicable (without annotation)
            if post_alpha > 1 and post_beta > 1:
                fig.add_vline(x=post_mode, line=dict(color='purple', width=2, dash='dot'))
                fig.add_annotation(
                    text=f"Posterior Mode: {post_mode:.4f}", 
                    x=post_mode, 
                    y=0.75, 
                    xref="x", 
                    yref="paper", 
                    showarrow=False
                )
            
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

            # Add vertical line for true proportion (without annotation)
            int_fig.add_vline(x=true_prop, line=dict(color='black', width=2, dash='dash'))

            # Add annotation separately 
            int_fig.add_annotation(
                text=f"True Proportion: {true_prop}", 
                x=true_prop, 
                y=1.05, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )
            
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
            
            # Add vertical line for true rate (without annotation)
            fig.add_vline(x=true_rate, line=dict(color='black', width=2, dash='dash'))
            fig.add_annotation(
                text=f"True Rate: {true_rate}", 
                x=true_rate, 
                y=1.05, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for sample rate (without annotation)
            fig.add_vline(x=sample_rate, line=dict(color='red', width=2))
            fig.add_annotation(
                text=f"Sample Rate: {sample_rate:.4f}", 
                x=sample_rate, 
                y=0.95, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for posterior mean (without annotation)
            fig.add_vline(x=post_mean, line=dict(color='blue', width=2))
            fig.add_annotation(
                text=f"Posterior Mean: {post_mean:.4f}", 
                x=post_mean, 
                y=0.85, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

            # Add vertical line for posterior mode if applicable (without annotation)
            if post_alpha > 1:
                fig.add_vline(x=post_mode, line=dict(color='purple', width=2, dash='dot'))
                fig.add_annotation(
                    text=f"Posterior Mode: {post_mode:.4f}", 
                    x=post_mode, 
                    y=0.75, 
                    xref="x", 
                    yref="paper", 
                    showarrow=False
                )
            
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
            # Add vertical line for true rate (without annotation)
            int_fig.add_vline(x=true_rate, line=dict(color='black', width=2, dash='dash'))

            # Add annotation separately
            int_fig.add_annotation(
                text=f"True Rate: {true_rate}", 
                x=true_rate, 
                y=1.05, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )
            
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
        # Implementation for multiple testing adjustment simulation
        st.info("This section will demonstrate how to adjust confidence intervals for multiple comparisons.")
        
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
    
    # Define custom function for profile likelihood calculation
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
        try:
            lower_result = minimize_scalar(
                lambda v: np.abs(profile_obj(v)), 
                bounds=(mle_params[param_idx] * 0.1, mle_params[param_idx]),
                method='bounded'
            )
            lower_bound = lower_result.x
        except:
            lower_bound = mle_params[param_idx] * 0.5  # Fallback if optimization fails
        
        # Find the upper bound
        try:
            upper_result = minimize_scalar(
                lambda v: np.abs(profile_obj(v)), 
                bounds=(mle_params[param_idx], mle_params[param_idx] * 5),
                method='bounded'
            )
            upper_bound = upper_result.x
        except:
            upper_bound = mle_params[param_idx] * 1.5  # Fallback if optimization fails
        
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
    
    # Different model type implementations
    if model_type == "Logistic Regression":
        # Additional options for logistic regression
        intercept = st.slider("Intercept (β₀)", -5.0, 5.0, 0.0, 0.5)
        add_quadratic = st.checkbox("Add quadratic term", value=False)
        
        if st.button("Generate Profile Likelihood Intervals", key="gen_profile"):
            st.info("This feature will demonstrate profile likelihood intervals for logistic regression models.")
    
    elif model_type == "Exponential Decay":
        # Additional options for exponential decay
        noise_level = st.slider("Noise level", 0.01, 1.0, 0.1, 0.01)
        
        if st.button("Generate Profile Likelihood Intervals", key="gen_profile"):
            # Generate data for exponential decay
            np.random.seed(None)
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
            try:
                profile_results = profile_likelihood_interval(
                    x, y, exp_decay_model, init_params, 1, 'Decay Rate', conf_level
                )
                
                # Display results
                st.success("Profile likelihood intervals calculated successfully!")
                
                # Create visualization
                st.subheader("Exponential Decay Model Fit")
                
                # Plot data and fitted model
                fig = go.Figure()
                
                # Add observed data points
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name='Observed Data',
                    marker=dict(color='blue', size=8)
                ))
                
                # Add true model
                fig.add_trace(go.Scatter(
                    x=x, y=y_true,
                    mode='lines',
                    name='True Model',
                    line=dict(color='green', width=2)
                ))
                
                # Add fitted model using MLE
                y_fit = profile_results['full_mle'][0] * np.exp(-profile_results['full_mle'][1] * x)
                fig.add_trace(go.Scatter(
                    x=x, y=y_fit,
                    mode='lines',
                    name='Fitted Model (MLE)',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Exponential Decay Model Fit',
                    xaxis_title='x',
                    yaxis_title='y',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create interval comparison visualization
                st.subheader("Confidence Interval Comparison")
                
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
                        profile_results['wald_lower'] <= param_value <= profile_results['wald_upper'],
                        profile_results['profile_lower'] <= param_value <= profile_results['profile_upper']
                    ],
                    'Symmetric': [
                        np.isclose(profile_results['mle'] - profile_results['wald_lower'], 
                                 profile_results['wald_upper'] - profile_results['mle'], rtol=0.05),
                        np.isclose(profile_results['mle'] - profile_results['profile_lower'], 
                                 profile_results['profile_upper'] - profile_results['mle'], rtol=0.05)
                    ]
                })
                
                st.dataframe(comparison_df.style.format({
                    'Lower Bound': '{:.4f}',
                    'Upper Bound': '{:.4f}',
                    'Width': '{:.4f}'
                }))
                
                # Visualization of intervals
                int_fig = go.Figure()
                
                # Add intervals as segments
                methods = ['Wald-type', 'Profile Likelihood']
                y_positions = [1, 2]
                lower_bounds = [profile_results['wald_lower'], profile_results['profile_lower']]
                upper_bounds = [profile_results['wald_upper'], profile_results['profile_upper']]
                colors = ['blue', 'red']
                
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
                        x=lower_bounds[i] - 0.05,
                        y=y_positions[i],
                        text=f"{lower_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="right"
                    )
                    
                    int_fig.add_annotation(
                        x=upper_bounds[i] + 0.05,
                        y=y_positions[i],
                        text=f"{upper_bounds[i]:.4f}",
                        showarrow=False,
                        xanchor="left"
                    )
                
                # Add vertical lines for true value and MLE
                # Add vertical line for true value (without annotation)
                int_fig.add_vline(x=param_value, line=dict(color='black', width=2, dash='dash'))
                int_fig.add_annotation(
                    text=f"True Value: {param_value}", 
                    x=param_value, 
                    y=1.05, 
                    xref="x", 
                    yref="paper", 
                    showarrow=False
                )

                # Add vertical line for MLE (without annotation)
                int_fig.add_vline(x=profile_results['mle'], line=dict(color='green', width=2))
                int_fig.add_annotation(
                    text=f"MLE: {profile_results['mle']:.4f}", 
                    x=profile_results['mle'], 
                    y=0.95, 
                    xref="x", 
                    yref="paper", 
                    showarrow=False
                )
                
                int_fig.update_layout(
                    title=f'Comparison of {conf_level*100:.0f}% Confidence Intervals for Decay Rate',
                    xaxis_title='Parameter Value',
                    yaxis=dict(
                        tickmode='array',
                        tickvals=y_positions,
                        ticktext=methods,
                        showgrid=False
                    ),
                    height=300
                )
                
                st.plotly_chart(int_fig, use_container_width=True)
                
                # Interpretation
                st.markdown(f"""
                ### Profile Likelihood vs. Wald-type Confidence Intervals
                
                **Key results for the decay rate parameter**:
                
                - Maximum Likelihood Estimate (MLE): {profile_results['mle']:.4f}
                - True parameter value: {param_value}
                - Profile likelihood {conf_level*100:.0f}% CI: [{profile_results['profile_lower']:.4f}, {profile_results['profile_upper']:.4f}]
                - Standard Wald-type {conf_level*100:.0f}% CI: [{profile_results['wald_lower']:.4f}, {profile_results['wald_upper']:.4f}]
                
                **Comparison**:
                
                - The profile likelihood interval is {'wider' if (profile_results['profile_upper'] - profile_results['profile_lower']) > (profile_results['wald_upper'] - profile_results['wald_lower']) else 'narrower'} than the Wald interval.
                - The profile likelihood interval is {'symmetric' if np.isclose(profile_results['mle'] - profile_results['profile_lower'], profile_results['profile_upper'] - profile_results['mle'], rtol=0.05) else 'asymmetric'} around the MLE.
                - For nonlinear models like exponential decay, profile likelihood intervals often provide better coverage, especially with small samples or parameters near boundaries.
                """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Profile likelihood calculation can be numerically challenging. Try adjusting the model parameters or sample size.")
    
    elif model_type == "Weibull Survival":
        # Additional options for weibull survival
        censoring_rate = st.slider("Censoring rate", 0.0, 0.8, 0.3, 0.05)
        shape_param = st.slider("Shape parameter", 0.5, 5.0, 1.5, 0.1)
        
        if st.button("Generate Profile Likelihood Intervals", key="gen_profile"):
            st.info("This feature will demonstrate profile likelihood intervals for survival models.")

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
        st.info("This feature will demonstrate simultaneous confidence bands for regression functions.")

# At the bottom of your app.py file, add this code:

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; padding: 10px;'>
        <p>Designed and developed by Vishal Bharti © 2025 | PhD-Level Confidence Intervals Explorer</p>
    </div>
    """, 
    unsafe_allow_html=True
)