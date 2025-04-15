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
from force_visible_math import setup_math_rendering
from latex_helper import add_latex_styles
st.set_page_config(
    page_title="Advanced Methods - CI Explorer",
    page_icon="📊",
    layout="wide",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)
add_latex_styles()
setup_math_rendering()

# Apply the custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


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
        # Set random seed
        np.random.seed(None)
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if simulation_type == "Independent Means":
            # Generate independent group data
            true_means = np.zeros(n_parameters)
            true_means[0] = effect_size  # Set first group to have an effect
            
            # Generate samples for each group
            samples = []
            sample_means = []
            sample_sds = []
            
            for i in range(n_parameters):
                sample = np.random.normal(true_means[i], 1.0, sample_size)
                samples.append(sample)
                sample_means.append(np.mean(sample))
                sample_sds.append(np.std(sample, ddof=1))
            
            # Calculate t-critical values
            t_crit_unadjusted = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
            
            # Calculate adjusted critical values
            if adjustment_method == "Bonferroni":
                alpha_adjusted = (1 - conf_level) / n_parameters
                t_crit_adjusted = stats.t.ppf(1 - alpha_adjusted/2, sample_size - 1)
                method_name = "Bonferroni"
                
            elif adjustment_method == "Šidák":
                alpha_adjusted = 1 - (1 - (1 - conf_level))**(1/n_parameters)
                t_crit_adjusted = stats.t.ppf(1 - alpha_adjusted/2, sample_size - 1)
                method_name = "Šidák"
                
            else:  # "Simultaneous Confidence Regions"
                # Use Tukey's method for all-pairwise comparisons
                # For this simulation we'll approximate with the maximum t-statistic
                t_crit_adjusted = stats.t.ppf(1 - (1 - conf_level)/(2*n_parameters), sample_size - 1)
                method_name = "Tukey-equivalent"
            
            # Calculate confidence intervals
            ci_lower_unadjusted = []
            ci_upper_unadjusted = []
            ci_lower_adjusted = []
            ci_upper_adjusted = []
            
            for i in range(n_parameters):
                margin_unadjusted = t_crit_unadjusted * sample_sds[i] / np.sqrt(sample_size)
                margin_adjusted = t_crit_adjusted * sample_sds[i] / np.sqrt(sample_size)
                
                ci_lower_unadjusted.append(sample_means[i] - margin_unadjusted)
                ci_upper_unadjusted.append(sample_means[i] + margin_unadjusted)
                
                ci_lower_adjusted.append(sample_means[i] - margin_adjusted)
                ci_upper_adjusted.append(sample_means[i] + margin_adjusted)
            
            # Check if intervals contain true means
            contains_true_unadjusted = []
            contains_true_adjusted = []
            
            for i in range(n_parameters):
                contains_true_unadjusted.append(ci_lower_unadjusted[i] <= true_means[i] <= ci_upper_unadjusted[i])
                contains_true_adjusted.append(ci_lower_adjusted[i] <= true_means[i] <= ci_upper_adjusted[i])
            
            # Calculate familywise error rate
            fwer_unadjusted = not all(contains_true_unadjusted)
            fwer_adjusted = not all(contains_true_adjusted)
            
            # Create visualization of confidence intervals
            fig = go.Figure()
            
            # Add horizontal line at y=0 (true mean for most groups)
            fig.add_hline(y=0, line=dict(color='green', width=1, dash='dash'),
                        annotation=dict(text="True Mean (Control)", showarrow=False))
            
            if effect_size > 0:
                # Add horizontal line for the group with effect
                fig.add_hline(y=effect_size, line=dict(color='red', width=1, dash='dash'),
                            annotation=dict(text=f"True Mean (Effect: {effect_size})", showarrow=False))
            
            # Add unadjusted intervals
            for i in range(n_parameters):
                fig.add_trace(go.Scatter(
                    x=[i, i],
                    y=[ci_lower_unadjusted[i], ci_upper_unadjusted[i]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Unadjusted CIs' if i == 0 else None,
                    showlegend=(i == 0)
                ))
                
                # Add point for sample mean
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[sample_means[i]],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Sample Means' if i == 0 else None,
                    showlegend=(i == 0)
                ))
            
            # Add adjusted intervals
            for i in range(n_parameters):
                fig.add_trace(go.Scatter(
                    x=[i+0.2, i+0.2],
                    y=[ci_lower_adjusted[i], ci_upper_adjusted[i]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'{method_name} Adjusted CIs' if i == 0 else None,
                    showlegend=(i == 0)
                ))
                
                # Add point for sample mean
                fig.add_trace(go.Scatter(
                    x=[i+0.2],
                    y=[sample_means[i]],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Sample Means (Adjusted)' if i == 0 else None,
                    showlegend=(i == 0)
                ))
            
            # Update layout
            fig.update_layout(
                title='Multiple Confidence Intervals Comparison',
                xaxis_title='Parameter Index',
                yaxis_title='Parameter Value',
                height=500,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(n_parameters)),
                    ticktext=[f'μ{i+1}' for i in range(n_parameters)]
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a comparison table
            data = {
                'Parameter': [f'μ{i+1}' for i in range(n_parameters)],
                'True Value': true_means,
                'Sample Mean': sample_means,
                'Unadjusted Lower': ci_lower_unadjusted,
                'Unadjusted Upper': ci_upper_unadjusted,
                'Unadjusted Width': [u - l for u, l in zip(ci_upper_unadjusted, ci_lower_unadjusted)],
                'Unadjusted Contains True': contains_true_unadjusted,
                f'{method_name} Lower': ci_lower_adjusted,
                f'{method_name} Upper': ci_upper_adjusted,
                f'{method_name} Width': [u - l for u, l in zip(ci_upper_adjusted, ci_lower_adjusted)],
                f'{method_name} Contains True': contains_true_adjusted
            }
            
            comp_df = pd.DataFrame(data)
            
            st.subheader("Confidence Interval Comparison")
            st.dataframe(comp_df.style.format({
                'True Value': '{:.4f}',
                'Sample Mean': '{:.4f}',
                'Unadjusted Lower': '{:.4f}',
                'Unadjusted Upper': '{:.4f}',
                'Unadjusted Width': '{:.4f}',
                f'{method_name} Lower': '{:.4f}',
                f'{method_name} Upper': '{:.4f}',
                f'{method_name} Width': '{:.4f}'
            }))
            
            # Create a summary card with key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Unadjusted Interval Width (Avg)",
                    value=f"{np.mean([u - l for u, l in zip(ci_upper_unadjusted, ci_lower_unadjusted)]):.4f}"
                )
            
            with col2:
                st.metric(
                    label=f"{method_name} Interval Width (Avg)",
                    value=f"{np.mean([u - l for u, l in zip(ci_upper_adjusted, ci_lower_adjusted)]):.4f}",
                    delta=f"{np.mean([u - l for u, l in zip(ci_upper_adjusted, ci_lower_adjusted)]) - np.mean([u - l for u, l in zip(ci_upper_unadjusted, ci_lower_unadjusted)]):.4f}"
                )
            
            with col3:
                st.metric(
                    label="Width Increase Ratio",
                    value=f"{np.mean([u - l for u, l in zip(ci_upper_adjusted, ci_lower_adjusted)]) / np.mean([u - l for u, l in zip(ci_upper_unadjusted, ci_lower_unadjusted)]):.2f}x"
                )
            
            # Individual vs. family-wise error
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Individual Error Rate",
                    value=f"{sum([not x for x in contains_true_unadjusted]) / n_parameters:.4f}"
                )
            
            with col2:
                st.metric(
                    label="Family-wise Error (Unadjusted)",
                    value=f"{int(fwer_unadjusted)}",
                    delta=f"{'Error' if fwer_unadjusted else 'No Error'}"
                )
            
            # Add interpretation
            st.subheader("Interpretation")
            
            # Adjustment factor calculation for explanation
            if adjustment_method == "Bonferroni":
                factor_formula = "m"
                factor_value = n_parameters
            elif adjustment_method == "Šidák":
                factor_formula = "1 - (1 - α)^(1/m)"
                factor_value = 1 / (1 - (1 - (1 - conf_level))**(1/n_parameters))
            else:
                factor_formula = "approximation"
                factor_value = t_crit_adjusted / t_crit_unadjusted
            
            st.markdown(f"""
            ### Multiple Testing Adjustment for Confidence Intervals
            
            In this simulation, we constructed {n_parameters} confidence intervals simultaneously, with one group having a true effect size of {effect_size}.
            
            **Key findings:**
            
            1. **Unadjusted intervals**: The standard {conf_level*100:.0f}% confidence intervals have an expected individual coverage of {conf_level*100:.0f}%, but the family-wise coverage (probability that ALL intervals contain their true parameters) is much lower.
            
            2. **Adjusted intervals using {method_name}**: These intervals are wider to ensure that the family-wise coverage is at least {conf_level*100:.0f}%.
            
            3. **Width comparison**: The {method_name}-adjusted intervals are approximately {factor_value:.2f} times wider than unadjusted intervals, reflecting the adjustment factor of {factor_formula}.
            
            4. **Error rates**:
               - Individual error rate (unadjusted): {sum([not x for x in contains_true_unadjusted]) / n_parameters:.4f}
               - Family-wise error occurred (unadjusted): {"Yes" if fwer_unadjusted else "No"}
               - Family-wise error occurred (adjusted): {"Yes" if fwer_adjusted else "No"}
            
            **Practical implications:**
            
            When making inference about multiple parameters simultaneously, unadjusted confidence intervals can lead to misleading conclusions due to increased family-wise error rates. The {method_name} adjustment provides protection against this inflated error at the cost of wider intervals and reduced power.
            
            **Method details:**
            
            - **Bonferroni**: Divides α by the number of comparisons (α/m), simple but conservative
            - **Šidák**: Uses 1-(1-α)^(1/m), slightly less conservative than Bonferroni 
            - **Simultaneous Confidence Regions**: Based on the joint distribution of test statistics, can be more powerful for correlated parameters
            
            **Recommendation:**
            
            Choose the adjustment method based on:
            1. Number of parameters (larger m → bigger difference between methods)
            2. Correlation structure among parameters (higher correlation → Simultaneous methods more efficient)
            3. Specific inference goals (individual vs. joint conclusions)
            """)
            
        elif simulation_type == "Regression Coefficients":
            # Set up the design matrix
            p = n_parameters  # Number of predictors
            n = sample_size   # Number of observations
            
            # Generate correlated predictors
            if x_correlation != 0:
                # Create correlation matrix
                corr_matrix = np.ones((p, p)) * x_correlation
                np.fill_diagonal(corr_matrix, 1.0)
                
                # Convert correlation to covariance matrix
                cov_matrix = corr_matrix  # Assuming standardized variables
                
                # Generate multivariate normal predictors
                X = np.random.multivariate_normal(np.zeros(p), cov_matrix, n)
            else:
                # Generate independent predictors
                X = np.random.normal(0, 1, (n, p))
            
            # Create true coefficient vector (one non-zero coefficient)
            beta_true = np.zeros(p)
            beta_true[0] = effect_size
            
            # Generate response variable
            y = X @ beta_true + np.random.normal(0, 1, n)
            
            # Add intercept
            X_with_intercept = np.column_stack((np.ones(n), X))
            beta_true = np.insert(beta_true, 0, 0)  # Add intercept (zero)
            
            # Fit regression model
            model = sm.OLS(y, X_with_intercept)
            results = model.fit()
            
            # Get coefficient estimates and standard errors
            beta_hat = results.params
            se = results.bse
            
            # Calculate unadjusted confidence intervals
            t_crit_unadjusted = stats.t.ppf(1 - (1 - conf_level)/2, n - p - 1)
            
            ci_lower_unadjusted = beta_hat - t_crit_unadjusted * se
            ci_upper_unadjusted = beta_hat + t_crit_unadjusted * se
            
            # Calculate adjusted critical values
            if adjustment_method == "Bonferroni":
                alpha_adjusted = (1 - conf_level) / (p + 1)  # Including intercept
                t_crit_adjusted = stats.t.ppf(1 - alpha_adjusted/2, n - p - 1)
                method_name = "Bonferroni"
                
            elif adjustment_method == "Šidák":
                alpha_adjusted = 1 - (1 - (1 - conf_level))**(1/(p + 1))
                t_crit_adjusted = stats.t.ppf(1 - alpha_adjusted/2, n - p - 1)
                method_name = "Šidák"
                
            else:  # "Simultaneous Confidence Regions"
                # Using Scheffé method for regression coefficients
                f_crit = stats.f.ppf(conf_level, p + 1, n - p - 1)
                t_crit_adjusted = np.sqrt((p + 1) * f_crit)
                method_name = "Scheffé"
            
            # Calculate adjusted confidence intervals
            ci_lower_adjusted = beta_hat - t_crit_adjusted * se
            ci_upper_adjusted = beta_hat + t_crit_adjusted * se
            
            # Check if intervals contain true coefficients
            contains_true_unadjusted = (ci_lower_unadjusted <= beta_true) & (beta_true <= ci_upper_unadjusted)
            contains_true_adjusted = (ci_lower_adjusted <= beta_true) & (beta_true <= ci_upper_adjusted)
            
            # Calculate familywise error rate
            fwer_unadjusted = not np.all(contains_true_unadjusted)
            fwer_adjusted = not np.all(contains_true_adjusted)
            
            # Create visualization of confidence intervals
            fig = go.Figure()
            
            # Add horizontal line at y=0 (true value for most coefficients)
            fig.add_hline(y=0, line=dict(color='green', width=1, dash='dash'),
                        annotation=dict(text="True Value (Zero)", showarrow=False))
            
            if effect_size > 0:
                # Add horizontal line for the coefficient with effect
                fig.add_hline(y=effect_size, line=dict(color='red', width=1, dash='dash'),
                            annotation=dict(text=f"True Value (Effect: {effect_size})", showarrow=False))
            
            # Add unadjusted intervals
            for i in range(p + 1):
                fig.add_trace(go.Scatter(
                    x=[i, i],
                    y=[ci_lower_unadjusted[i], ci_upper_unadjusted[i]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Unadjusted CIs' if i == 0 else None,
                    showlegend=(i == 0)
                ))
                
                # Add point for coefficient estimate
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[beta_hat[i]],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Coefficient Estimates' if i == 0 else None,
                    showlegend=(i == 0)
                ))
            
            # Add adjusted intervals
            for i in range(p + 1):
                fig.add_trace(go.Scatter(
                    x=[i+0.2, i+0.2],
                    y=[ci_lower_adjusted[i], ci_upper_adjusted[i]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'{method_name} Adjusted CIs' if i == 0 else None,
                    showlegend=(i == 0)
                ))
                
                # Add point for coefficient estimate
                fig.add_trace(go.Scatter(
                    x=[i+0.2],
                    y=[beta_hat[i]],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Coefficient Estimates (Adjusted)' if i == 0 else None,
                    showlegend=(i == 0)
                ))
            
            # Update layout
            fig.update_layout(
                title='Multiple Confidence Intervals for Regression Coefficients',
                xaxis_title='Coefficient Index',
                yaxis_title='Coefficient Value',
                height=500,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(p + 1)),
                    ticktext=['Intercept'] + [f'β{i}' for i in range(1, p + 1)]
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a comparison table
            data = {
                'Coefficient': ['Intercept'] + [f'β{i}' for i in range(1, p + 1)],
                'True Value': beta_true,
                'Estimate': beta_hat,
                'Std Error': se,
                'Unadjusted Lower': ci_lower_unadjusted,
                'Unadjusted Upper': ci_upper_unadjusted,
                'Unadjusted Width': ci_upper_unadjusted - ci_lower_unadjusted,
                'Unadjusted Contains True': contains_true_unadjusted,
                f'{method_name} Lower': ci_lower_adjusted,
                f'{method_name} Upper': ci_upper_adjusted,
                f'{method_name} Width': ci_upper_adjusted - ci_lower_adjusted,
                f'{method_name} Contains True': contains_true_adjusted
            }
            
            comp_df = pd.DataFrame(data)
            
            st.subheader("Confidence Interval Comparison")
            st.dataframe(comp_df.style.format({
                'True Value': '{:.4f}',
                'Estimate': '{:.4f}',
                'Std Error': '{:.4f}',
                'Unadjusted Lower': '{:.4f}',
                'Unadjusted Upper': '{:.4f}',
                'Unadjusted Width': '{:.4f}',
                f'{method_name} Lower': '{:.4f}',
                f'{method_name} Upper': '{:.4f}',
                f'{method_name} Width': '{:.4f}'
            }))
            
            # Create a summary card with key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Unadjusted Interval Width (Avg)",
                    value=f"{np.mean(ci_upper_unadjusted - ci_lower_unadjusted):.4f}"
                )
            
            with col2:
                st.metric(
                    label=f"{method_name} Interval Width (Avg)",
                    value=f"{np.mean(ci_upper_adjusted - ci_lower_adjusted):.4f}",
                    delta=f"{np.mean(ci_upper_adjusted - ci_lower_adjusted) - np.mean(ci_upper_unadjusted - ci_lower_unadjusted):.4f}"
                )
            
            with col3:
                st.metric(
                    label="Width Increase Ratio",
                    value=f"{np.mean(ci_upper_adjusted - ci_lower_adjusted) / np.mean(ci_upper_unadjusted - ci_lower_unadjusted):.2f}x"
                )
            
            # Individual vs. family-wise error
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Individual Error Rate",
                    value=f"{np.sum(~contains_true_unadjusted) / (p + 1):.4f}"
                )
            
            with col2:
                st.metric(
                    label="Family-wise Error (Unadjusted)",
                    value=f"{int(fwer_unadjusted)}",
                    delta=f"{'Error' if fwer_unadjusted else 'No Error'}"
                )
            
            # Add interpretation
            st.subheader("Interpretation")
            
            # Adjustment factor calculation for explanation
            if adjustment_method == "Bonferroni":
                factor_desc = f"m = {p+1} (number of coefficients including intercept)"
            elif adjustment_method == "Šidák":
                factor_desc = f"1-(1-α)^(1/m) where m = {p+1}"
            else:  # Scheffé
                factor_desc = f"√(p × F(p,n-p)) where p = {p+1} and n = {n}"
            
            st.markdown(f"""
            ### Multiple Testing Adjustment for Regression Coefficients
            
            In this regression model with {p+1} coefficients (including intercept), we used multiple testing adjustments to control the family-wise error rate for confidence intervals.
            
            **Key findings:**
            
            1. **Correlation structure**: The predictors have a correlation of {x_correlation}, which affects the efficiency of different adjustment methods.
            
            2. **Coefficient of interest**: β₁ has a true effect of {effect_size}, while all other coefficients are 0.
            
            3. **Adjusted vs. unadjusted intervals**:
               - The {method_name}-adjusted intervals are approximately {t_crit_adjusted/t_crit_unadjusted:.2f} times wider than the unadjusted intervals
               - The adjustment is based on {factor_desc}
            
            4. **Error rates**:
               - Individual error rate (unadjusted): {np.sum(~contains_true_unadjusted) / (p + 1):.4f}
               - Family-wise error occurred (unadjusted): {"Yes" if fwer_unadjusted else "No"}
               - Family-wise error occurred (adjusted): {"Yes" if fwer_adjusted else "No"}
            
            **Method comparisons:**
            
            - **Bonferroni** is easy to implement but conservative, especially with many coefficients
            - **Šidák** is slightly less conservative than Bonferroni but still ignores correlation
            - **Scheffé** accounts for the joint distribution and performs better with correlated predictors
            
            **Practical implications:**
            
            When performing regression analysis with multiple coefficients, especially when making statements about all coefficients simultaneously, adjusted confidence intervals provide proper control of family-wise error rate. However, this comes at the cost of wider intervals and reduced power to detect non-zero effects.
            
            For focused hypotheses about specific coefficients, consider using unadjusted intervals or more targeted adjustment methods.
            """)
            
            # Show correlation matrix if correlation is non-zero
            if x_correlation != 0:
                st.subheader("Predictor Correlation Matrix")
                
                # Calculate observed correlation matrix
                X_corr = np.corrcoef(X, rowvar=False)
                
                # Create heatmap
                corr_fig = go.Figure(data=go.Heatmap(
                    z=X_corr,
                    x=[f'X{i+1}' for i in range(p)],
                    y=[f'X{i+1}' for i in range(p)],
                    colorscale='RdBu_r',
                    zmin=-1,
                    zmax=1
                ))
                
                corr_fig.update_layout(
                    title='Predictor Correlation Matrix',
                    height=400
                )
                
                st.plotly_chart(corr_fig, use_container_width=True)
        
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
        noise_level = st.slider("Noise level", 0.1, 2.0, 1.0, 0.1)
        
        if st.button("Generate Profile Likelihood Intervals", key="gen_logistic_profile"):
            # Generate data for logistic regression
            np.random.seed(None)
            
            # Generate predictors
            x = np.linspace(-3, 3, sample_size)
            
            # Generate linear predictor
            if add_quadratic:
                # Add quadratic term
                linear_pred = intercept + param_value * x + 0.5 * x**2
                model_description = f"logit(p) = {intercept:.2f} + {param_value:.2f}x + 0.5x²"
            else:
                # Simple linear model
                linear_pred = intercept + param_value * x
                model_description = f"logit(p) = {intercept:.2f} + {param_value:.2f}x"
            
            # Convert to probabilities
            p = 1 / (1 + np.exp(-linear_pred))
            
            # Generate binary outcomes
            y = np.random.binomial(1, p)
            
            # Fit logistic regression model
            import statsmodels.api as sm
            
            # Prepare design matrix
            if add_quadratic:
                X = sm.add_constant(np.column_stack((x, x**2)))
                param_index = 1  # Index of the coefficient we're interested in
            else:
                X = sm.add_constant(x)
                param_index = 1  # Index of the coefficient we're interested in
            
            # Fit the model
            logit_model = sm.Logit(y, X)
            result = logit_model.fit(disp=0)  # Suppress convergence messages
            
            # Extract key information
            coefs = result.params
            std_errors = result.bse
            conf_int_wald = result.conf_int(alpha=1-conf_level)
            
            # Calculate Wald confidence interval
            wald_lower = conf_int_wald[param_index][0]
            wald_upper = conf_int_wald[param_index][1]
            
            # Calculate profile likelihood confidence interval
            
            # Define a function to compute profile likelihood for different parameter values
            def profile_likelihood(param_value_to_test):
                # Fix the parameter of interest and optimize over the others
                def neg_loglike(other_params):
                    # Reconstruct full parameter vector
                    if add_quadratic:
                        full_params = np.array([other_params[0], param_value_to_test, other_params[1]])
                    else:
                        full_params = np.array([other_params[0], param_value_to_test])
                    
                    # Compute negative log-likelihood
                    p_pred = 1 / (1 + np.exp(-np.dot(X, full_params)))
                    p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)  # Avoid log(0)
                    return -np.sum(y * np.log(p_pred) + (1 - y) * np.log(1 - p_pred))
                
                # Initial guess for other parameters
                if add_quadratic:
                    init_other_params = np.array([coefs[0], coefs[2]])
                else:
                    init_other_params = np.array([coefs[0]])
                
                # Optimize the conditional log-likelihood
                import scipy.optimize as opt
                result = opt.minimize(neg_loglike, init_other_params)
                
                # Return the profiled negative log-likelihood
                return result.fun
            
            # Calculate the maximum log-likelihood
            max_loglike = -logit_model.loglike(coefs)
            
            # Critical value for the likelihood ratio test
            crit_value = stats.chi2.ppf(conf_level, 1) / 2
            
            # Find the profile likelihood confidence interval
            from scipy.optimize import minimize_scalar, brentq
            
            # Function to optimize: difference between profile likelihood and critical value
            def profile_diff(param_val):
                return profile_likelihood(param_val) - (max_loglike + crit_value)
            
            # Find lower bound
            try:
                # Try to find where the profile likelihood equals the critical value
                lower_bound = brentq(profile_diff, coefs[param_index] - 5*std_errors[param_index], coefs[param_index])
            except:
                # Fallback: Just search in an interval
                lower_result = minimize_scalar(
                    lambda v: abs(profile_diff(v)),
                    bounds=(coefs[param_index] - 5*std_errors[param_index], coefs[param_index]),
                    method='bounded'
                )
                lower_bound = lower_result.x
            
            # Find upper bound
            try:
                upper_bound = brentq(profile_diff, coefs[param_index], coefs[param_index] + 5*std_errors[param_index])
            except:
                upper_result = minimize_scalar(
                    lambda v: abs(profile_diff(v)),
                    bounds=(coefs[param_index], coefs[param_index] + 5*std_errors[param_index]),
                    method='bounded'
                )
                upper_bound = upper_result.x
            
            # Create visualization of the model fit
            model_fig = go.Figure()
            
            # Calculate predicted probabilities
            p_pred = 1 / (1 + np.exp(-np.dot(X, coefs)))
            
            # Create color scale based on predicted probabilities
            colors = np.array(['blue' if val == 0 else 'red' for val in y])
            sizes = np.array([8 if val == 0 else 10 for val in y])
            
            # Add scatter plot of original data
            model_fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=colors, size=sizes, opacity=0.7),
                name='Observed Data'
            ))
            
            # Add line for predicted probabilities
            model_fig.add_trace(go.Scatter(
                x=x, y=p_pred,
                mode='lines',
                line=dict(color='green', width=2),
                name='Predicted Probability'
            ))
            
            # Add line for true probabilities
            model_fig.add_trace(go.Scatter(
                x=x, y=p,
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='True Probability'
            ))
            
            model_fig.update_layout(
                title=f'Logistic Regression Model: {model_description}',
                xaxis_title='x',
                yaxis_title='Probability/Outcome',
                height=400,
                yaxis=dict(range=[-0.1, 1.1])
            )
            
            st.plotly_chart(model_fig, use_container_width=True)
            
            # Create visualization of profile likelihood
            profile_fig = go.Figure()
            
            # Calculate profile likelihood for a range of parameter values
            param_range = np.linspace(
                min(lower_bound, wald_lower) - 0.5,
                max(upper_bound, wald_upper) + 0.5,
                100
            )
            
            profile_vals = []
            for param_val in param_range:
                profile_vals.append(profile_likelihood(param_val))
                
            # Normalize for better visualization
            profile_vals = np.array(profile_vals)
            profile_vals_norm = profile_vals - min(profile_vals)
            profile_vals_norm = profile_vals_norm / max(profile_vals_norm)
            
            # Add profile likelihood curve
            profile_fig.add_trace(go.Scatter(
                x=param_range, y=profile_vals_norm,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Profile Likelihood'
            ))
            
            # Add horizontal line at critical value
            crit_norm = (max_loglike + crit_value - min(profile_vals)) / max(profile_vals_norm)
            profile_fig.add_hline(y=crit_norm, line=dict(color='red', width=2, dash='dash'),
                            annotation=dict(text=f"{conf_level*100:.0f}% Threshold", showarrow=False))
            
            # Add vertical lines for MLE and bounds
            profile_fig.add_vline(x=coefs[param_index], line=dict(color='green', width=2),
                            annotation=dict(text=f"MLE: {coefs[param_index]:.4f}", showarrow=False))
            
            profile_fig.add_vline(x=lower_bound, line=dict(color='blue', width=2, dash='dash'),
                            annotation=dict(text=f"Lower: {lower_bound:.4f}", showarrow=False))
            
            profile_fig.add_vline(x=upper_bound, line=dict(color='blue', width=2, dash='dash'),
                            annotation=dict(text=f"Upper: {upper_bound:.4f}", showarrow=False))
            
            # Add vertical lines for Wald interval
            profile_fig.add_vline(x=wald_lower, line=dict(color='orange', width=2, dash='dot'),
                            annotation=dict(text=f"Wald Lower: {wald_lower:.4f}", showarrow=False))
            
            profile_fig.add_vline(x=wald_upper, line=dict(color='orange', width=2, dash='dot'),
                            annotation=dict(text=f"Wald Upper: {wald_upper:.4f}", showarrow=False))
            
            profile_fig.add_vline(x=param_value, line=dict(color='black', width=2, dash='dash'),
                            annotation=dict(text=f"True Value: {param_value:.4f}", showarrow=False))
            
            profile_fig.update_layout(
                title='Profile Likelihood for Slope Parameter',
                xaxis_title='Parameter Value',
                yaxis_title='Normalized Profile Likelihood',
                height=400,
                yaxis=dict(range=[0, 1.1])
            )
            
            st.plotly_chart(profile_fig, use_container_width=True)
            
            # Create interval comparison visualization
            interval_fig = go.Figure()
            
            # Add intervals as segments
            methods = ['Wald', 'Profile Likelihood']
            y_positions = [1, 2]
            lower_bounds = [wald_lower, lower_bound]
            upper_bounds = [wald_upper, upper_bound]
            colors = ['orange', 'blue']
            
            for i, method_name in enumerate(methods):
                interval_fig.add_trace(go.Scatter(
                    x=[lower_bounds[i], upper_bounds[i]],
                    y=[y_positions[i], y_positions[i]],
                    mode='lines',
                    name=method_name,
                    line=dict(color=colors[i], width=4)
                ))
                
                # Add endpoints as markers
                interval_fig.add_trace(go.Scatter(
                    x=[lower_bounds[i], upper_bounds[i]],
                    y=[y_positions[i], y_positions[i]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(color=colors[i], size=8)
                ))
                
                # Add labels for bounds
                interval_fig.add_annotation(
                    x=lower_bounds[i] - 0.2,
                    y=y_positions[i],
                    text=f"{lower_bounds[i]:.4f}",
                    showarrow=False,
                    xanchor="right"
                )
                
                interval_fig.add_annotation(
                    x=upper_bounds[i] + 0.2,
                    y=y_positions[i],
                    text=f"{upper_bounds[i]:.4f}",
                    showarrow=False,
                    xanchor="left"
                )
            
            # Add vertical line for true value
            interval_fig.add_vline(x=param_value, line=dict(color='black', width=2, dash='dash'))
            interval_fig.add_annotation(
                text=f"True Value: {param_value}", 
                x=param_value, 
                y=3, 
                showarrow=False
            )
            
            # Add vertical line for MLE
            interval_fig.add_vline(x=coefs[param_index], line=dict(color='green', width=2))
            interval_fig.add_annotation(
                text=f"MLE: {coefs[param_index]:.4f}", 
                x=coefs[param_index], 
                y=0, 
                showarrow=False
            )
            
            interval_fig.update_layout(
                title=f'Comparison of {conf_level*100:.0f}% Confidence Intervals for Slope Parameter',
                xaxis_title='Parameter Value',
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_positions,
                    ticktext=methods,
                    showgrid=False
                ),
                height=300
            )
            
            st.plotly_chart(interval_fig, use_container_width=True)
            
            # Create results table
            results_df = pd.DataFrame({
                'Method': ['Wald', 'Profile Likelihood'],
                'Lower Bound': [wald_lower, lower_bound],
                'Upper Bound': [wald_upper, upper_bound],
                'Width': [wald_upper - wald_lower, upper_bound - lower_bound],
                'Contains True Value': [
                    wald_lower <= param_value <= wald_upper,
                    lower_bound <= param_value <= upper_bound
                ],
                'Symmetric Around MLE': [
                    np.isclose(coefs[param_index] - wald_lower, wald_upper - coefs[param_index], rtol=0.05),
                    np.isclose(coefs[param_index] - lower_bound, upper_bound - coefs[param_index], rtol=0.05)
                ]
            })
            
            st.subheader("Confidence Interval Comparison")
            st.dataframe(results_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Width': '{:.4f}'
            }))
            
            # Add interpretation
            st.subheader("Interpretation")
            
            wald_contains = wald_lower <= param_value <= wald_upper
            profile_contains = lower_bound <= param_value <= upper_bound
            width_diff = (upper_bound - lower_bound) - (wald_upper - wald_lower)
            
            st.markdown(f"""
            ### Profile Likelihood vs. Wald Confidence Intervals
            
            In this example, we've fitted a logistic regression model with the true slope parameter set to {param_value}.
            
            **Key findings**:
            
            1. **Maximum Likelihood Estimate (MLE)**: The estimated slope is {coefs[param_index]:.4f}, which is {'close to' if abs(coefs[param_index] - param_value) < 0.3*std_errors[param_index] else 'somewhat different from'} the true value of {param_value}.
            
            2. **Confidence Intervals**:
            - Wald {conf_level*100:.0f}% CI: [{wald_lower:.4f}, {wald_upper:.4f}], width: {wald_upper - wald_lower:.4f}
            - Profile Likelihood {conf_level*100:.0f}% CI: [{lower_bound:.4f}, {upper_bound:.4f}], width: {upper_bound - lower_bound:.4f}
            
            3. **Comparison**:
            - The profile likelihood interval is {width_diff:.4f} units {'wider' if width_diff > 0 else 'narrower'} than the Wald interval.
            - The Wald interval is {'symmetric' if np.isclose(coefs[param_index] - wald_lower, wald_upper - coefs[param_index], rtol=0.05) else 'asymmetric'} around the MLE.
            - The profile likelihood interval is {'symmetric' if np.isclose(coefs[param_index] - lower_bound, upper_bound - coefs[param_index], rtol=0.05) else 'asymmetric'} around the MLE.
            - The true parameter value is {'contained in both intervals' if wald_contains and profile_contains else 'contained only in the profile likelihood interval' if profile_contains else 'contained only in the Wald interval' if wald_contains else 'not contained in either interval'}.
            
            **Why Profile Likelihood?**
            
            Profile likelihood confidence intervals are particularly valuable for logistic regression because:
            
            1. They don't rely on asymptotic normality assumptions, which can be problematic for small samples
            2. They respect the natural parameter bounds and constraints of the model
            3. They can capture asymmetry in the likelihood surface, which Wald intervals ignore
            4. They generally have better coverage properties, especially for parameters near boundaries
            
            **Practical Guidance**:
            
            - For routine analyses with large samples, Wald intervals may be sufficient
            - For small samples, extreme proportions, or parameters near boundaries, profile likelihood intervals are recommended
            - If computational resources permit, profile likelihood intervals provide more accurate coverage and better reflect the uncertainty in parameter estimates
            """)
    
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
    # Additional options for Weibull survival model
        shape_param = st.slider("Shape parameter", 0.5, 5.0, 1.5, 0.1)
        censoring_rate = st.slider("Censoring rate", 0.0, 0.8, 0.3, 0.05)
        
        if st.button("Generate Profile Likelihood Intervals", key="gen_weibull_profile"):
            # Generate data for Weibull survival analysis
            np.random.seed(None)
            
            # Generate survival times from Weibull distribution
            # Weibull parameters: scale=param_value, shape=shape_param
            # Survival time T ~ Weibull(scale=param_value, shape=shape_param)
            true_scale = param_value
            true_shape = shape_param
            
            # Generate uncensored survival times
            u = np.random.uniform(0, 1, sample_size)
            survival_times = true_scale * (-np.log(1 - u))**(1/true_shape)
            
            # Generate censoring times (exponential distribution)
            # Adjust censoring rate by changing the scale parameter
            censoring_scale = np.quantile(survival_times, 1 - censoring_rate) * 1.5
            censoring_times = np.random.exponential(scale=censoring_scale, size=sample_size)
            
            # Determine observed times and censoring indicators
            observed_times = np.minimum(survival_times, censoring_times)
            censoring_indicators = (survival_times <= censoring_times).astype(int)
            
            # Function to compute negative log-likelihood for Weibull model
            def neg_loglik_weibull(params):
                scale, shape = params
                
                # Avoid negative or zero parameters
                if scale <= 0 or shape <= 0:
                    return 1e9  # Return large value for invalid parameters
                
                # Compute log-likelihood
                ll = 0
                
                # Contribution from uncensored observations (failures)
                uncensored = (censoring_indicators == 1)
                if np.any(uncensored):
                    ll += np.sum(np.log(shape) + shape * np.log(scale) + 
                                (shape - 1) * np.log(observed_times[uncensored]) - 
                                (observed_times[uncensored] / scale)**shape)
                
                # Contribution from censored observations
                censored = (censoring_indicators == 0)
                if np.any(censored):
                    ll += np.sum(-(observed_times[censored] / scale)**shape)
                
                return -ll
            
            # Fit Weibull model using Maximum Likelihood
            from scipy.optimize import minimize
            
            # Initial parameter guesses based on transformed data
            # Using median estimate method
            median_time = np.median(observed_times[censoring_indicators == 1])
            init_scale = median_time / (np.log(2)**(1/1.5))  # Assuming shape around 1.5
            init_shape = 1.5
            
            # Find MLE estimates
            result = minimize(neg_loglik_weibull, [init_scale, init_shape], 
                            method='L-BFGS-B', bounds=[(0.001, None), (0.001, None)])
            
            # Extract ML estimates
            mle_scale, mle_shape = result.x
            mle_loglik = -result.fun
            
            # Calculate Hessian numerically for standard errors
            from scipy.optimize import approx_fprime
            
            # Function to compute Hessian matrix numerically
            def hessian(func, x, epsilon=1e-5):
                n = len(x)
                H = np.zeros((n, n))
                f_x = func(x)
                
                for i in range(n):
                    x_plus = x.copy()
                    x_plus[i] += epsilon
                    f_plus = func(x_plus)
                    
                    x_minus = x.copy()
                    x_minus[i] -= epsilon
                    f_minus = func(x_minus)
                    
                    H[i, i] = (f_plus - 2 * f_x + f_minus) / (epsilon**2)
                    
                    for j in range(i+1, n):
                        x_plus_plus = x.copy()
                        x_plus_plus[i] += epsilon
                        x_plus_plus[j] += epsilon
                        f_pp = func(x_plus_plus)
                        
                        x_plus_minus = x.copy()
                        x_plus_minus[i] += epsilon
                        x_plus_minus[j] -= epsilon
                        f_pm = func(x_plus_minus)
                        
                        x_minus_plus = x.copy()
                        x_minus_plus[i] -= epsilon
                        x_minus_plus[j] += epsilon
                        f_mp = func(x_minus_plus)
                        
                        x_minus_minus = x.copy()
                        x_minus_minus[i] -= epsilon
                        x_minus_minus[j] -= epsilon
                        f_mm = func(x_minus_minus)
                        
                        H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                        H[j, i] = H[i, j]
                
                return H
            
            # Compute Hessian at MLE
            H = hessian(neg_loglik_weibull, result.x)
            
            # Invert to get variance-covariance matrix
            try:
                var_cov = np.linalg.inv(H)
            except:
                # Add small regularization if matrix is singular
                var_cov = np.linalg.inv(H + np.eye(2) * 1e-6)
            
            # Extract standard errors
            se_scale = np.sqrt(var_cov[0, 0])
            se_shape = np.sqrt(var_cov[1, 1])
            
            # Calculate Wald confidence intervals
            z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
            wald_lower_scale = mle_scale - z_crit * se_scale
            wald_upper_scale = mle_scale + z_crit * se_scale
            
            # Profile likelihood function for scale parameter
            def profile_likelihood_scale(scale_value):
                """Calculate profile likelihood for scale parameter by optimizing over shape"""
                def neg_loglik_shape(shape):
                    return neg_loglik_weibull([scale_value, shape])
                
                # Optimize over shape
                shape_result = minimize(neg_loglik_shape, mle_shape, 
                                        method='L-BFGS-B', bounds=[(0.001, None)])
                
                return -shape_result.fun  # Return profile log-likelihood
            
            # Calculate profile likelihood-based confidence interval
            # Critical value for likelihood ratio test
            chi2_crit = stats.chi2.ppf(conf_level, 1)
            loglik_threshold = mle_loglik - chi2_crit/2
            
            # Function to find profile likelihood bounds
            def profile_diff(scale_value):
                return profile_likelihood_scale(scale_value) - loglik_threshold
            
            # Find profile likelihood bounds
            from scipy.optimize import brentq
            
            # Try to find lower bound
            try:
                profile_lower = brentq(profile_diff, max(0.001, mle_scale - 4*se_scale), mle_scale)
            except:
                # Fallback: Wald lower bound
                profile_lower = wald_lower_scale
            
            # Try to find upper bound
            try:
                profile_upper = brentq(profile_diff, mle_scale, mle_scale + 4*se_scale)
            except:
                # Fallback: Wald upper bound
                profile_upper = wald_upper_scale
            
            # Create visualization of data and fitted model
            model_fig = go.Figure()
            
            # Create Kaplan-Meier survival curve from data
            from lifelines import KaplanMeierFitter
            
            kmf = KaplanMeierFitter()
            kmf.fit(observed_times, event_observed=censoring_indicators)
            
            # Add Kaplan-Meier estimate
            km_x = kmf.survival_function_.index.values
            km_y = kmf.survival_function_["KM_estimate"].values
            
            model_fig.add_trace(go.Scatter(
                x=km_x, y=km_y,
                mode='lines+markers',
                name='Kaplan-Meier Estimate',
                line=dict(color='blue', width=2)
            ))
            
            # Add fitted Weibull survival function
            model_t = np.linspace(0, max(observed_times) * 1.2, 100)
            model_surv = np.exp(-(model_t / mle_scale)**mle_shape)
            
            model_fig.add_trace(go.Scatter(
                x=model_t, y=model_surv,
                mode='lines',
                name=f'Fitted Weibull (scale={mle_scale:.2f}, shape={mle_shape:.2f})',
                line=dict(color='red', width=2)
            ))
            
            # Add true Weibull survival function
            true_surv = np.exp(-(model_t / true_scale)**true_shape)
            
            model_fig.add_trace(go.Scatter(
                x=model_t, y=true_surv,
                mode='lines',
                name=f'True Weibull (scale={true_scale:.2f}, shape={true_shape:.2f})',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Add censored observations
            cens_times = observed_times[censoring_indicators == 0]
            cens_y = np.interp(cens_times, model_t, model_surv)
            
            model_fig.add_trace(go.Scatter(
                x=cens_times, y=cens_y,
                mode='markers',
                name='Censored Observations',
                marker=dict(symbol='circle-open', size=10, color='blue')
            ))
            
            model_fig.update_layout(
                title='Weibull Survival Model with Censoring',
                xaxis_title='Time',
                yaxis_title='Survival Probability',
                height=400,
                yaxis=dict(range=[0, 1.05])
            )
            
            st.plotly_chart(model_fig, use_container_width=True)
            
            # Create profile likelihood visualization
            profile_fig = go.Figure()
            
            # Calculate profile likelihood for a range of scale values
            scale_range = np.linspace(
                max(0.1, min(profile_lower, wald_lower_scale) * 0.8),
                max(profile_upper, wald_upper_scale) * 1.2,
                50
            )
            
            profile_loglik = []
            for scale_val in scale_range:
                profile_loglik.append(profile_likelihood_scale(scale_val))
            
            # Normalize for visualization
            profile_loglik = np.array(profile_loglik)
            profile_norm = (profile_loglik - min(profile_loglik)) / (max(profile_loglik) - min(profile_loglik))
            
            # Add profile likelihood curve
            profile_fig.add_trace(go.Scatter(
                x=scale_range, y=profile_norm,
                mode='lines',
                name='Profile Likelihood',
                line=dict(color='blue', width=2)
            ))
            
            # Add threshold line
            threshold_norm = (loglik_threshold - min(profile_loglik)) / (max(profile_loglik) - min(profile_loglik))
            profile_fig.add_hline(y=threshold_norm, line=dict(color='red', width=2, dash='dash'),
                            annotation=dict(text=f"{conf_level*100:.0f}% Threshold", showarrow=False))
            
            # Add vertical lines
            # MLE
            profile_fig.add_vline(x=mle_scale, line=dict(color='green', width=2),
                            annotation=dict(text=f"MLE: {mle_scale:.4f}", showarrow=False))
            
            # Profile likelihood bounds
            profile_fig.add_vline(x=profile_lower, line=dict(color='blue', width=2, dash='dash'),
                            annotation=dict(text=f"Lower: {profile_lower:.4f}", showarrow=False))
            
            profile_fig.add_vline(x=profile_upper, line=dict(color='blue', width=2, dash='dash'),
                            annotation=dict(text=f"Upper: {profile_upper:.4f}", showarrow=False))
            
            # Wald bounds
            profile_fig.add_vline(x=wald_lower_scale, line=dict(color='orange', width=2, dash='dot'),
                            annotation=dict(text=f"Wald Lower: {wald_lower_scale:.4f}", showarrow=False))
            
            profile_fig.add_vline(x=wald_upper_scale, line=dict(color='orange', width=2, dash='dot'),
                            annotation=dict(text=f"Wald Upper: {wald_upper_scale:.4f}", showarrow=False))
            
            # True value
            profile_fig.add_vline(x=true_scale, line=dict(color='black', width=2, dash='dash'),
                            annotation=dict(text=f"True Value: {true_scale:.4f}", showarrow=False))
            
            profile_fig.update_layout(
                title='Profile Likelihood for Weibull Scale Parameter',
                xaxis_title='Scale Parameter',
                yaxis_title='Normalized Profile Likelihood',
                height=400,
                yaxis=dict(range=[0, 1.05])
            )
            
            st.plotly_chart(profile_fig, use_container_width=True)
            
            # Create interval comparison visualization
            interval_fig = go.Figure()
            
            # Add intervals as segments
            methods = ['Wald', 'Profile Likelihood']
            y_positions = [1, 2]
            lower_bounds = [wald_lower_scale, profile_lower]
            upper_bounds = [wald_upper_scale, profile_upper]
            colors = ['orange', 'blue']
            
            for i, method_name in enumerate(methods):
                interval_fig.add_trace(go.Scatter(
                    x=[lower_bounds[i], upper_bounds[i]],
                    y=[y_positions[i], y_positions[i]],
                    mode='lines',
                    name=method_name,
                    line=dict(color=colors[i], width=4)
                ))
                
                # Add endpoints as markers
                interval_fig.add_trace(go.Scatter(
                    x=[lower_bounds[i], upper_bounds[i]],
                    y=[y_positions[i], y_positions[i]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(color=colors[i], size=8)
                ))
                
                # Add labels for bounds
                interval_fig.add_annotation(
                    x=lower_bounds[i] - 0.2,
                    y=y_positions[i],
                    text=f"{lower_bounds[i]:.4f}",
                    showarrow=False,
                    xanchor="right"
                )
                
                interval_fig.add_annotation(
                    x=upper_bounds[i] + 0.2,
                    y=y_positions[i],
                    text=f"{upper_bounds[i]:.4f}",
                    showarrow=False,
                    xanchor="left"
                )
            
            # Add vertical line for true value
            interval_fig.add_vline(x=true_scale, line=dict(color='black', width=2, dash='dash'))
            interval_fig.add_annotation(
                text=f"True Value: {true_scale}", 
                x=true_scale, 
                y=3, 
                showarrow=False
            )
            
            # Add vertical line for MLE
            interval_fig.add_vline(x=mle_scale, line=dict(color='green', width=2))
            interval_fig.add_annotation(
                text=f"MLE: {mle_scale:.4f}", 
                x=mle_scale, 
                y=0, 
                showarrow=False
            )
            
            interval_fig.update_layout(
                title=f'Comparison of {conf_level*100:.0f}% Confidence Intervals for Scale Parameter',
                xaxis_title='Scale Parameter',
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_positions,
                    ticktext=methods,
                    showgrid=False
                ),
                height=300
            )
            
            st.plotly_chart(interval_fig, use_container_width=True)
            
            # Create results summary table
            results_df = pd.DataFrame({
                'Parameter': ['Scale', 'Shape'],
                'True Value': [true_scale, true_shape],
                'MLE': [mle_scale, mle_shape],
                'Standard Error': [se_scale, se_shape],
                'Wald CI': [f'[{wald_lower_scale:.4f}, {wald_upper_scale:.4f}]', ''],
                'Profile CI': [f'[{profile_lower:.4f}, {profile_upper:.4f}]', '']
            })
            
            st.subheader("Parameter Estimates")
            st.dataframe(results_df)
            
            # Create comparison table
            comp_df = pd.DataFrame({
                'Method': ['Wald', 'Profile Likelihood'],
                'Lower Bound': [wald_lower_scale, profile_lower],
                'Upper Bound': [wald_upper_scale, profile_upper],
                'Width': [wald_upper_scale - wald_lower_scale, profile_upper - profile_lower],
                'Contains True Value': [
                    wald_lower_scale <= true_scale <= wald_upper_scale,
                    profile_lower <= true_scale <= profile_upper
                ],
                'Symmetric Around MLE': [
                    np.isclose(mle_scale - wald_lower_scale, wald_upper_scale - mle_scale, rtol=0.05),
                    np.isclose(mle_scale - profile_lower, profile_upper - mle_scale, rtol=0.05)
                ]
            })
            
            st.subheader("Confidence Interval Comparison")
            st.dataframe(comp_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Width': '{:.4f}'
            }))
            
            # Add interpretation
            st.subheader("Interpretation")
            
            cenr_percent = 100 * (1 - censoring_indicators.mean())
            wald_contains = wald_lower_scale <= true_scale <= wald_upper_scale
            profile_contains = profile_lower <= true_scale <= profile_upper
            width_diff = (profile_upper - profile_lower) - (wald_upper_scale - wald_lower_scale)
            
            st.markdown(f"""
            ### Profile Likelihood vs. Wald Confidence Intervals for Survival Analysis
            
            In this example, we've fitted a Weibull survival model with {sample_size} observations, of which {cenr_percent:.1f}% were censored.
            
            **Key findings**:
            
            1. **Maximum Likelihood Estimate (MLE)**: The estimated scale parameter is {mle_scale:.4f}, which is {'close to' if abs(mle_scale - true_scale) < 0.3*se_scale else 'somewhat different from'} the true value of {true_scale}. The estimated shape parameter is {mle_shape:.4f} (true value: {true_shape}).
            
            2. **Confidence Intervals for Scale Parameter**:
            - Wald {conf_level*100:.0f}% CI: [{wald_lower_scale:.4f}, {wald_upper_scale:.4f}], width: {wald_upper_scale - wald_lower_scale:.4f}
            - Profile Likelihood {conf_level*100:.0f}% CI: [{profile_lower:.4f}, {profile_upper:.4f}], width: {profile_upper - profile_lower:.4f}
            
            3. **Comparison**:
            - The profile likelihood interval is {width_diff:.4f} units {'wider' if width_diff > 0 else 'narrower'} than the Wald interval.
            - The Wald interval is {'symmetric' if np.isclose(mle_scale - wald_lower_scale, wald_upper_scale - mle_scale, rtol=0.05) else 'asymmetric'} around the MLE.
            - The profile likelihood interval is {'symmetric' if np.isclose(mle_scale - profile_lower, profile_upper - mle_scale, rtol=0.05) else 'asymmetric'} around the MLE.
            - The true parameter value is {'contained in both intervals' if wald_contains and profile_contains else 'contained only in the profile likelihood interval' if profile_contains else 'contained only in the Wald interval' if wald_contains else 'not contained in either interval'}.
            
            **Why Profile Likelihood for Survival Analysis?**
            
            Profile likelihood confidence intervals are particularly valuable for survival models because:
            
            1. **Handling Censoring**: Survival data often involves censoring, which can lead to asymmetric likelihood functions
            2. **Parameter Constraints**: Survival parameters (scale, shape) must be positive, and profile likelihood naturally respects these bounds
            3. **Small-to-Moderate Samples**: With limited data and censoring, asymptotic normality assumptions of Wald intervals may not hold
            4. **Nuisance Parameters**: In survival analysis, profile likelihood effectively accounts for uncertainty in nuisance parameters
            
            **Impact of Censoring**:
            
            This dataset has {cenr_percent:.1f}% censored observations. {'Higher censoring rates typically lead to wider confidence intervals and greater differences between Wald and profile likelihood methods.' if cenr_percent > 20 else 'Even with this moderate censoring level, differences between the methods can be observed.'}
            
            **Clinical Relevance**:
            
            In survival analysis applications (such as clinical trials, reliability engineering, or actuarial science), the scale parameter often represents a characteristic time value (e.g., median survival time). Accurate confidence intervals for this parameter are critical for:
            
            - Predicting patient outcomes and planning treatment protocols
            - Estimating product reliability and warranty periods
            - Determining risk factors and pricing in insurance
            
            **Recommendation**:
            
            For Weibull survival models, especially with:
            - Moderate to high censoring rates
            - Small to moderate sample sizes
            - When precision in parameter estimation is critical
            
            Profile likelihood intervals provide more accurate uncertainty quantification than standard Wald-type intervals.
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
    

    # Function for generating and displaying simultaneous confidence bands
    def generate_confidence_bands(function_type, sample_size, conf_level, noise_level, 
                                true_intercept=1.0, true_slope=0.5, poly_degree=3, 
                                show_extrapolation=False, bandwidth=0.2, band_method="Working-Hotelling"):
        """Generate and plot simultaneous confidence bands for regression functions"""
        
        # Set random seed for reproducibility
        np.random.seed(None)
        
        # Generate x values evenly spaced in [0, 10]
        x_min, x_max = 0, 10
        x = np.linspace(x_min, x_max, sample_size)
        
        # Generate true function values based on function type
        if function_type == "Linear Regression":
            y_true = true_intercept + true_slope * x
            function_label = f"True function: y = {true_intercept:.2f} + {true_slope:.2f}x"
        
        elif function_type == "Polynomial Regression":
            # Generate coefficients for the polynomial (decreasing magnitude with degree)
            coeffs = [true_intercept, true_slope]  # First two coefficients
            for i in range(2, poly_degree + 1):
                # Alternate sign and decrease magnitude with degree
                coeffs.append(((-1)**(i) * 0.1 * (poly_degree - i + 1)))
            
            # Generate y values from polynomial
            y_true = np.zeros_like(x)
            for i, coef in enumerate(coeffs):
                y_true += coef * x**i
            
            # Create a nice label for the polynomial
            terms = [f"{coeffs[0]:.2f}"]
            for i in range(1, len(coeffs)):
                if coeffs[i] >= 0:
                    terms.append(f"+ {coeffs[i]:.2f}x^{i}")
                else:
                    terms.append(f"- {abs(coeffs[i]):.2f}x^{i}")
            function_label = f"True function: y = {' '.join(terms)}"
        
        elif function_type == "Nonparametric Regression":
            # Nonparametric: use a sinusoidal function with some additional features
            y_true = true_intercept + np.sin(x) + 0.5 * np.sin(2*x) + true_slope * x/5
            function_label = "True function: Nonparametric (sinusoidal with trend)"
        
        # Add random noise to generate observed data
        y_obs = y_true + np.random.normal(0, noise_level, size=sample_size)
        
        # Create extrapolation region if requested
        if show_extrapolation:
            x_range = x_max - x_min
            x_extended = np.linspace(x_min - 0.2*x_range, x_max + 0.2*x_range, 200)
        else:
            x_extended = np.linspace(x_min, x_max, 200)
        
        # Create arrays for fitted values and confidence bands
        y_fit = np.zeros_like(x_extended)
        lower_band = np.zeros_like(x_extended)
        upper_band = np.zeros_like(x_extended)
        
        # Compute the fitted model and confidence bands
        if function_type == "Linear Regression":
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_obs)
            
            # Prediction for all points
            y_fit = intercept + slope * x_extended
            
            # Calculate standard error of the prediction
            # First get residual standard error
            y_pred_obs = intercept + slope * x
            residuals = y_obs - y_pred_obs
            residual_std = np.sqrt(np.sum(residuals**2) / (sample_size - 2))
            
            # Standard error for predictions at each x point
            x_mean = np.mean(x)
            x_var = np.sum((x - x_mean)**2)
            
            # Calculate standard error for each point in the extended range
            se_fit = np.array([
                residual_std * np.sqrt(1/sample_size + (x_val - x_mean)**2 / x_var)
                for x_val in x_extended
            ])
            
            # Calculate critical value based on band method
            if band_method == "Working-Hotelling":
                # Working-Hotelling bands (based on F-distribution)
                p = 2  # Number of parameters in linear model
                F = stats.f.ppf(conf_level, p, sample_size - p)
                critical_value = np.sqrt(p * F)
            
            elif band_method == "Bonferroni":
                # Bonferroni correction for multiple comparisons
                alpha = 1 - conf_level
                alpha_adjusted = alpha / (2 * len(x_extended))  # Two-sided, multiple points
                critical_value = stats.norm.ppf(1 - alpha_adjusted/2)
            
            else:  # Bootstrap would be more complex, approximate here
                critical_value = stats.norm.ppf((1 + conf_level)/2)
            
            # Calculate confidence bands
            margin = critical_value * se_fit
            lower_band = y_fit - margin
            upper_band = y_fit + margin
            
            # Label for the fitted model
            model_label = f"Fitted: y = {intercept:.4f} + {slope:.4f}x"
        
        elif function_type == "Polynomial Regression":
            # Fit polynomial regression
            poly_model = np.polyfit(x, y_obs, poly_degree)
            
            # Create polynomial function for prediction
            p = np.poly1d(poly_model)
            y_fit = p(x_extended)
            
            # Calculate confidence bands (complex for polynomial)
            # We'll use bootstrapping to approximate the confidence bands
            
            # First define a function to fit polynomial and predict
            def fit_poly_predict(x_data, y_data, degree, x_pred):
                poly = np.polyfit(x_data, y_data, degree)
                p_fitted = np.poly1d(poly)
                return p_fitted(x_pred)
            
            # Bootstrap to generate many fitted curves
            n_bootstrap = 1000
            bootstrap_curves = np.zeros((n_bootstrap, len(x_extended)))
            
            for i in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(sample_size, sample_size, replace=True)
                x_boot = x[indices]
                y_boot = y_obs[indices]
                
                # Fit model to bootstrap sample and predict
                bootstrap_curves[i, :] = fit_poly_predict(x_boot, y_boot, poly_degree, x_extended)
            
            # Calculate confidence bands from bootstrap samples
            if band_method == "Working-Hotelling":
                # Modified for polynomial case
                bootvar = np.var(bootstrap_curves, axis=0)
                se_fit = np.sqrt(bootvar)
                
                # Working-Hotelling bands (wider than pointwise)
                p = poly_degree + 1  # Number of parameters
                F = stats.f.ppf(conf_level, p, sample_size - p)
                critical_value = np.sqrt(p * F)
                
                margin = critical_value * se_fit
            
            elif band_method == "Bonferroni":
                # Bonferroni bands
                alpha = 1 - conf_level
                lower_percentile = alpha / 2 * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                # Adjust for multiple testing
                lower_percentile = lower_percentile / len(x_extended)
                upper_percentile = 100 - (100 - upper_percentile) / len(x_extended)
                
                # Get percentiles from bootstrap samples
                lower_band = np.percentile(bootstrap_curves, lower_percentile, axis=0)
                upper_band = np.percentile(bootstrap_curves, upper_percentile, axis=0)
                margin = None  # Not used in this case
            
            else:  # Standard bootstrap
                # Get percentiles directly from bootstrap samples
                lower_band = np.percentile(bootstrap_curves, (1 - conf_level) / 2 * 100, axis=0)
                upper_band = np.percentile(bootstrap_curves, (1 + conf_level) / 2 * 100, axis=0)
                margin = None  # Not used in this case
            
            # If margin was calculated, apply it
            if margin is not None:
                lower_band = y_fit - margin
                upper_band = y_fit + margin
            
            # Format the polynomial equation for display
            poly_eq = f"{poly_model[0]:.4f}"
            for i in range(1, len(poly_model)):
                if poly_model[i] >= 0:
                    poly_eq += f" + {poly_model[i]:.4f}x^{poly_degree-i}"
                else:
                    poly_eq += f" - {abs(poly_model[i]):.4f}x^{poly_degree-i}"
            
            model_label = f"Fitted: y = {poly_eq}"
        
        elif function_type == "Nonparametric Regression":
            # Kernel regression with variable bandwidth
            
            # Define Gaussian kernel function
            def gaussian_kernel(u):
                return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
            
            # Nadaraya-Watson estimator for kernel regression
            def kernel_regression(x_data, y_data, x_pred, bandwidth):
                n = len(x_data)
                y_pred = np.zeros_like(x_pred)
                
                for i, x_i in enumerate(x_pred):
                    # Calculate weights for each data point
                    weights = gaussian_kernel((x_i - x_data) / bandwidth)
                    # Compute weighted average
                    if np.sum(weights) > 0:
                        y_pred[i] = np.sum(weights * y_data) / np.sum(weights)
                    else:
                        y_pred[i] = np.mean(y_data)
                
                return y_pred
            
            # Fit nonparametric model
            y_fit = kernel_regression(x, y_obs, x_extended, bandwidth)
            
            # For confidence bands, we'll use bootstrap
            n_bootstrap = 1000
            bootstrap_curves = np.zeros((n_bootstrap, len(x_extended)))
            
            for i in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(sample_size, sample_size, replace=True)
                x_boot = x[indices]
                y_boot = y_obs[indices]
                
                # Fit model to bootstrap sample and predict
                bootstrap_curves[i, :] = kernel_regression(x_boot, y_boot, x_extended, bandwidth)
            
            # Calculate confidence bands based on method
            if band_method == "Working-Hotelling":
                # For nonparametric, this is an approximation
                bootvar = np.var(bootstrap_curves, axis=0)
                se_fit = np.sqrt(bootvar)
                
                # Working-Hotelling bands (conservative for nonparametric)
                critical_value = 2.8  # Higher than parametric
                
                margin = critical_value * se_fit
                lower_band = y_fit - margin
                upper_band = y_fit + margin
            
            elif band_method == "Bonferroni":
                # Bonferroni bands
                alpha = 1 - conf_level
                lower_percentile = alpha / 2 / len(x_extended) * 100
                upper_percentile = 100 - lower_percentile
                
                # Get percentiles from bootstrap samples
                lower_band = np.percentile(bootstrap_curves, max(0.1, lower_percentile), axis=0)
                upper_band = np.percentile(bootstrap_curves, min(99.9, upper_percentile), axis=0)
            
            else:  # Standard bootstrap
                # Get percentiles directly from bootstrap samples
                lower_band = np.percentile(bootstrap_curves, (1 - conf_level) / 2 * 100, axis=0)
                upper_band = np.percentile(bootstrap_curves, (1 + conf_level) / 2 * 100, axis=0)
            
            model_label = f"Fitted: Nonparametric (bandwidth={bandwidth})"
        
        # Create the plot with observed data, true function, fitted function, and confidence bands
        fig = go.Figure()
        
        # Add observed data points
        fig.add_trace(go.Scatter(
            x=x, y=y_obs,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Observed Data'
        ))
        
        # Add true function
        if function_type == "Linear Regression":
            true_y_extended = true_intercept + true_slope * x_extended
        elif function_type == "Polynomial Regression":
            true_y_extended = np.array([sum([coeffs[i] * xx**i for i in range(len(coeffs))]) for xx in x_extended])
        else:  # Nonparametric
            true_y_extended = true_intercept + np.sin(x_extended) + 0.5 * np.sin(2*x_extended) + true_slope * x_extended/5
        
        fig.add_trace(go.Scatter(
            x=x_extended, y=true_y_extended,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name=function_label
        ))
        
        # Add fitted function
        fig.add_trace(go.Scatter(
            x=x_extended, y=y_fit,
            mode='lines',
            line=dict(color='red', width=2),
            name=model_label
        ))
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=x_extended, y=upper_band,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            name=f'{conf_level*100:.0f}% Upper Band'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_extended, y=lower_band,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            fill='tonexty',  # Fill between this trace and the previous one
            fillcolor='rgba(255, 0, 0, 0.2)',
            name=f'{conf_level*100:.0f}% Lower Band'
        ))
        
        # Add vertical lines to demarcate extrapolation regions if applicable
        if show_extrapolation:
            fig.add_vline(x=x_min, line=dict(color='black', width=1, dash='dot'))
            fig.add_vline(x=x_max, line=dict(color='black', width=1, dash='dot'))
            
            # Add annotations for extrapolation regions
            fig.add_annotation(
                x=(x_min - 0.2*x_range + x_min)/2, 
                y=y_obs.max(), 
                text="Extrapolation",
                showarrow=False,
                font=dict(color="black")
            )
            
            fig.add_annotation(
                x=(x_max + 0.2*x_range + x_max)/2, 
                y=y_obs.max(), 
                text="Extrapolation",
                showarrow=False,
                font=dict(color="black")
            )
        
        # Update layout with proper legend positioning
        fig.update_layout(
            title=f'Simultaneous {conf_level*100:.0f}% Confidence Bands for {function_type}',
            xaxis_title='x',
            yaxis_title='y',
            # Key changes for legend positioning
            legend=dict(
                x=1.05,       # Position to the right of the plot
                y=0.5,        # Center vertically
                xanchor='left', # Anchor to the left side of the legend box
                yanchor='middle', # Anchor to the middle of the legend box
                bgcolor='rgba(255, 255, 255, 0.7)', # Semi-transparent background
                bordercolor='rgba(0, 0, 0, 0.1)',  # Light border
                borderwidth=1
            ),
            # Increase right margin to make room for the legend
            margin=dict(r=150),  # Add space on the right side for the legend
            width=900,
            height=500,
            template='plotly_white'
        )
        
        # Add explanation for the confidence bands
        if function_type == "Linear Regression":
            explanation = f"""
            ### Linear Regression with {conf_level*100:.0f}% Simultaneous Confidence Bands
            
            These confidence bands provide bounds that contain the entire true regression line with {conf_level*100:.0f}% probability.
            
            **Key characteristics:**
            - **Sample size:** {sample_size} observations
            - **Noise level:** {noise_level}
            - **Band method:** {band_method}
            
            **Interpretation:**
            Simultaneous confidence bands are wider than pointwise confidence intervals because they account for the multiplicity of testing across the entire function domain. We can be {conf_level*100:.0f}% confident that the true regression line lies entirely within the bands.
            
            {f"The bands widen significantly in the extrapolation regions, reflecting increased uncertainty when predicting beyond the observed data range." if show_extrapolation else ""}
            
            **Technical details:**
            The {band_method} method was used to calculate the critical value for the bands. This approach {'controls the family-wise error rate by adjusting for multiple comparisons.' if band_method == 'Bonferroni' else 'is based on the F-distribution and provides exact simultaneous coverage for linear models.' if band_method == 'Working-Hotelling' else 'uses resampling to estimate the joint distribution of residuals.'}
            """
        
        elif function_type == "Polynomial Regression":
            explanation = f"""
            ### Polynomial Regression with {conf_level*100:.0f}% Simultaneous Confidence Bands
            
            These confidence bands provide bounds that contain the entire true polynomial function of degree {poly_degree} with {conf_level*100:.0f}% probability.
            
            **Key characteristics:**
            - **Polynomial degree:** {poly_degree}
            - **Sample size:** {sample_size} observations
            - **Noise level:** {noise_level}
            - **Band method:** {band_method}
            
            **Interpretation:**
            The bands are typically wider than for linear regression due to the increased model flexibility. The width increases with polynomial degree, reflecting greater uncertainty with more complex models.
            
            {f"Polynomial models can behave erratically in extrapolation regions. Note how the confidence bands widen dramatically outside the range of observed data." if show_extrapolation else ""}
            
            **Technical details:**
            For polynomial regression, the bands account for correlation between parameter estimates, which is essential for valid simultaneous inference. The {band_method} method was used to calculate these bands.
            """
        
        elif function_type == "Nonparametric Regression":
            explanation = f"""
            ### Nonparametric Regression with {conf_level*100:.0f}% Simultaneous Confidence Bands
            
            These confidence bands provide bounds that contain the entire true function with {conf_level*100:.0f}% probability, using kernel smoothing methods.
            
            **Key characteristics:**
            - **Bandwidth:** {bandwidth} (controls smoothness)
            - **Sample size:** {sample_size} observations
            - **Noise level:** {noise_level}
            - **Band method:** {band_method}
            
            **Interpretation:**
            Nonparametric confidence bands are typically wider than parametric ones because they make fewer assumptions about the function form. The bandwidth controls the trade-off between bias and variance:
            - Smaller bandwidth → less bias, more variance
            - Larger bandwidth → more bias, less variance
            
            {f"Nonparametric methods are particularly unreliable for extrapolation, as shown by the extremely wide bands outside the data range." if show_extrapolation else ""}
            
            **Technical details:**
            The Nadaraya-Watson estimator with a Gaussian kernel was used for the regression function. The {band_method} method was used to calculate the confidence bands.
            """
        
        return fig, explanation

    if st.button("Generate Simultaneous Confidence Bands", key="gen_bands"):
        # Call the function based on selected options
        if function_type == "Linear Regression":
            fig, explanation = generate_confidence_bands(
                function_type, sample_size, conf_level, noise_level,
                true_intercept=true_intercept, true_slope=true_slope
            )
        elif function_type == "Polynomial Regression":
            fig, explanation = generate_confidence_bands(
                function_type, sample_size, conf_level, noise_level,
                poly_degree=poly_degree, show_extrapolation=extrapolate
            )
        elif function_type == "Nonparametric Regression":
            fig, explanation = generate_confidence_bands(
                function_type, sample_size, conf_level, noise_level,
                bandwidth=bandwidth, band_method=band_method
            )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the explanation
        st.markdown(explanation)

# At the bottom of your app.py file, add this code:

# Add footer
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)