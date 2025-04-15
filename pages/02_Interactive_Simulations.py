import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from custom_styling import get_custom_css, get_footer
from latex_helper import render_latex, render_definition, render_example, render_proof, render_key_equation
from force_visible_math import force_visible_math_mode, inline_math_fix
from force_visible_math import setup_math_rendering
from latex_helper import add_latex_styles
st.set_page_config(
    page_title="Interactive Simulations - CI Explorer",
    page_icon="📊",
    layout="wide",
)


st.markdown(get_custom_css(), unsafe_allow_html=True)
add_latex_styles()
setup_math_rendering()

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
                st.stop()
        
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
                st.stop()
            
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

        # Add vertical lines for CI bounds (without annotations)
        fig.add_vline(x=lower, line=dict(color='red', width=2, dash='dash'))
        # Add annotation separately
        fig.add_annotation(
            text=f"Lower: {lower:.4f}", 
            x=lower, 
            y=1.05, 
            xref="x", 
            yref="paper", 
            showarrow=False
        )

        fig.add_vline(x=upper, line=dict(color='red', width=2, dash='dash'))
        fig.add_annotation(
            text=f"Upper: {upper:.4f}", 
            x=upper, 
            y=1.05, 
            xref="x", 
            yref="paper", 
            showarrow=False
        )

        # Add vertical line for observed statistic (without annotation)
        fig.add_vline(x=observed_stat, line=dict(color='blue', width=2))
        fig.add_annotation(
            text=f"Observed: {observed_stat:.4f}", 
            x=observed_stat, 
            y=0.95, 
            xref="x", 
            yref="paper", 
            showarrow=False
        )

        # Add vertical line for true value if known (without annotation)
        if true_value is not None:
            fig.add_vline(x=true_value, line=dict(color='green', width=2, dash='dot'))
            fig.add_annotation(
                text=f"True: {true_value:.4f}", 
                x=true_value, 
                y=0.85, 
                xref="x", 
                yref="paper", 
                showarrow=False
            )

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
        # Set random seed
        np.random.seed(None)
        
        if transform_type == "Log Transformation for Ratio":
            # Generate data - lognormal distribution produces ratios
            # Mean and SD in log scale that will give desired ratio and CV
            log_mean = np.log(true_ratio) - 0.5 * np.log(1 + cv**2)
            log_sd = np.sqrt(np.log(1 + cv**2))
            
            data = np.random.lognormal(log_mean, log_sd, sample_size)
            
            # Calculate statistics in original scale
            sample_ratio = np.mean(data)
            sample_sd = np.std(data, ddof=1)
            
            # Calculate confidence interval in original scale (non-transformed)
            t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
            margin = t_crit * sample_sd / np.sqrt(sample_size)
            
            ci_lower_orig = sample_ratio - margin
            ci_upper_orig = sample_ratio + margin
            
            # Check if lower bound is negative (shouldn't happen for ratios)
            if ci_lower_orig < 0:
                ci_lower_orig = 0
            
            # Calculate transformed statistics (in log scale)
            log_data = np.log(data)
            log_mean = np.mean(log_data)
            log_sd = np.std(log_data, ddof=1)
            
            # Calculate confidence interval in log scale
            log_margin = t_crit * log_sd / np.sqrt(sample_size)
            log_ci_lower = log_mean - log_margin
            log_ci_upper = log_mean + log_margin
            
            # Transform back to original scale
            ci_lower_trans = np.exp(log_ci_lower)
            ci_upper_trans = np.exp(log_ci_upper)
            
            # Create plot for original data and CI
            fig1 = go.Figure()
            
            # Add histogram of data
            fig1.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                opacity=0.6,
                name="Sample Data",
                marker_color="blue"
            ))
            
            # Add vertical lines for true ratio and sample ratio
            fig1.add_vline(x=true_ratio, line=dict(color="green", width=2, dash="dash"), 
                         annotation=dict(text=f"True ratio: {true_ratio:.2f}", showarrow=False))
            
            fig1.add_vline(x=sample_ratio, line=dict(color="red", width=2), 
                         annotation=dict(text=f"Sample ratio: {sample_ratio:.2f}", showarrow=False))
            
            # Add confidence intervals
            fig1.add_vrect(
                x0=ci_lower_orig, x1=ci_upper_orig,
                fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0,
                annotation=dict(text="Standard CI", showarrow=False)
            )
            
            fig1.add_vrect(
                x0=ci_lower_trans, x1=ci_upper_trans,
                fillcolor="rgba(0, 255, 0, 0.1)", layer="below", line_width=0,
                annotation=dict(text="Log-transformed CI", showarrow=False)
            )
            
            fig1.update_layout(
                title="Distribution of Ratios with Confidence Intervals",
                xaxis_title="Ratio Value",
                yaxis_title="Frequency",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create a comparison plot in log scale
            fig2 = go.Figure()
            
            # Add histogram of log-transformed data
            fig2.add_trace(go.Histogram(
                x=log_data,
                nbinsx=30,
                opacity=0.6,
                name="Log-transformed Data",
                marker_color="purple"
            ))
            
            # Add vertical lines
            fig2.add_vline(x=np.log(true_ratio), line=dict(color="green", width=2, dash="dash"), 
                         annotation=dict(text=f"Log(true ratio): {np.log(true_ratio):.2f}", showarrow=False))
            
            fig2.add_vline(x=log_mean, line=dict(color="red", width=2), 
                         annotation=dict(text=f"Log(sample ratio): {log_mean:.2f}", showarrow=False))
            
            # Add confidence interval
            fig2.add_vrect(
                x0=log_ci_lower, x1=log_ci_upper,
                fillcolor="rgba(128, 0, 128, 0.1)", layer="below", line_width=0,
                annotation=dict(text="CI in log scale", showarrow=False)
            )
            
            fig2.update_layout(
                title="Log-transformed Distribution with Confidence Interval",
                xaxis_title="Log(Ratio)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Create a comparison table
            comp_df = pd.DataFrame({
                'Method': ['Standard', 'Log-transformed'],
                'Lower Bound': [ci_lower_orig, ci_lower_trans],
                'Upper Bound': [ci_upper_orig, ci_upper_trans],
                'Interval Width': [ci_upper_orig - ci_lower_orig, ci_upper_trans - ci_lower_trans],
                'Symmetric Around Mean': [
                    f"{abs((sample_ratio - ci_lower_orig) - (ci_upper_orig - sample_ratio)) < 0.001}",
                    f"{abs((np.exp(log_mean) - ci_lower_trans) - (ci_upper_trans - np.exp(log_mean))) < 0.001}"
                ],
                'Contains True Value': [
                    f"{ci_lower_orig <= true_ratio <= ci_upper_orig}",
                    f"{ci_lower_trans <= true_ratio <= ci_upper_trans}"
                ]
            })
            
            st.subheader("Comparison of Confidence Interval Methods")
            st.dataframe(comp_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Interval Width': '{:.4f}'
            }))
            
            # Add interpretation
            st.subheader("Interpretation")
            
            standard_contains = ci_lower_orig <= true_ratio <= ci_upper_orig
            log_contains = ci_lower_trans <= true_ratio <= ci_upper_trans
            
            st.markdown(f"""
            #### Key Findings:
            
            1. **Standard CI**: [{ci_lower_orig:.4f}, {ci_upper_orig:.4f}]
               - {"Contains" if standard_contains else "Does not contain"} the true ratio of {true_ratio}
               - {'Symmetric around the sample mean' if abs((sample_ratio - ci_lower_orig) - (ci_upper_orig - sample_ratio)) < 0.001 else 'Not symmetric around the sample mean'}
            
            2. **Log-transformed CI**: [{ci_lower_trans:.4f}, {ci_upper_trans:.4f}]
               - {"Contains" if log_contains else "Does not contain"} the true ratio of {true_ratio}
               - {'Symmetric in log scale, but asymmetric in original scale' if abs((log_mean - log_ci_lower) - (log_ci_upper - log_mean)) < 0.001 else 'Not symmetric in log scale'}
            
            #### Why Log Transformation Works Better for Ratios:
            
            - Ratios are naturally bounded at zero and right-skewed
            - Log transformation makes the distribution more symmetric and closer to normal
            - In log scale, multiplicative relationships become additive, better matching normal theory
            - The log-transformed CI respects the natural boundary (ratio > 0)
            
            For this simulation with a true ratio of {true_ratio} and CV of {cv}:
            
            - {"Both methods captured the true value" if standard_contains and log_contains else 
               "Only the log-transformed method captured the true value" if log_contains else 
               "Only the standard method captured the true value" if standard_contains else
               "Neither method captured the true value"} 
            - The {"log-transformed" if ci_upper_trans - ci_lower_trans < ci_upper_orig - ci_lower_orig else "standard"} method produced a narrower interval
            - With small sample sizes or higher variability, the advantages of transformation become more pronounced
            """)
            
        elif transform_type == "Logit Transformation for Proportion":
            # Generate binomial data
            data = np.random.binomial(1, true_prop, sample_size)
            sample_prop = np.mean(data)
            
            # Standard Wald interval
            z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
            std_se = np.sqrt(sample_prop * (1 - sample_prop) / sample_size)
            
            ci_lower_std = max(0, sample_prop - z_crit * std_se)
            ci_upper_std = min(1, sample_prop + z_crit * std_se)
            
            # Logit-transformed interval
            # Avoid 0 and 1 for logit transformation
            if sample_prop == 0:
                sample_prop = 0.5 / sample_size
            elif sample_prop == 1:
                sample_prop = 1 - 0.5 / sample_size
                
            # Logit transformation
            logit_prop = np.log(sample_prop / (1 - sample_prop))
            
            # Variance of logit-transformed estimate using delta method
            logit_se = np.sqrt(1 / (sample_size * sample_prop * (1 - sample_prop)))
            
            # Confidence interval in logit scale
            logit_ci_lower = logit_prop - z_crit * logit_se
            logit_ci_upper = logit_prop + z_crit * logit_se
            
            # Transform back to probability scale
            ci_lower_logit = 1 / (1 + np.exp(-logit_ci_lower))
            ci_upper_logit = 1 / (1 + np.exp(-logit_ci_upper))
            
            # Wilson score interval (for comparison)
            denominator = 1 + z_crit**2 / sample_size
            center = (sample_prop + z_crit**2 / (2 * sample_size)) / denominator
            wilson_margin = z_crit * np.sqrt(sample_prop * (1 - sample_prop) / sample_size + z_crit**2 / (4 * sample_size**2)) / denominator
            
            ci_lower_wilson = max(0, center - wilson_margin)
            ci_upper_wilson = min(1, center + wilson_margin)
            
            # Create visualization
            fig = go.Figure()
            
            # Create range for probability scale
            p_range = np.linspace(0, 1, 1000)
            
            # Calculate standard error function across the range
            se_range = np.sqrt(p_range * (1 - p_range) / sample_size)
            
            # Create confidence band for standard approach
            upper_band = np.minimum(1, p_range + z_crit * se_range)
            lower_band = np.maximum(0, p_range - z_crit * se_range)
            
            # Add bands
            fig.add_trace(go.Scatter(
                x=p_range, y=upper_band,
                mode='lines', line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=p_range, y=lower_band,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)',
                name='Standard CI Band'
            ))
            
            # Calculate logit-transformed bands
            # Avoid boundary issues
            p_range_logit = np.linspace(0.001, 0.999, 1000)
            logit_range = np.log(p_range_logit / (1 - p_range_logit))
            
            # SE in logit scale
            logit_se_range = np.sqrt(1 / (sample_size * p_range_logit * (1 - p_range_logit)))
            
            # CI in logit scale
            logit_upper = logit_range + z_crit * logit_se_range
            logit_lower = logit_range - z_crit * logit_se_range
            
            # Transform back to probability scale
            p_upper_logit = 1 / (1 + np.exp(-logit_upper))
            p_lower_logit = 1 / (1 + np.exp(-logit_lower))
            
            # Add logit bands
            fig.add_trace(go.Scatter(
                x=p_range_logit, y=p_upper_logit,
                mode='lines', line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=p_range_logit, y=p_lower_logit,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0, 128, 0, 0.1)',
                name='Logit-transformed CI Band'
            ))
            
            # Add reference line where upper = lower (CI width = 0)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', line=dict(color='black', width=1, dash='dash'),
                name='Reference Line'
            ))
            
            # Add point for actual sample
            fig.add_trace(go.Scatter(
                x=[sample_prop], y=[sample_prop],
                mode='markers', marker=dict(color='red', size=10),
                name=f'Sample Proportion: {sample_prop:.3f}'
            ))
            
            # Add vertical lines for confidence intervals
            fig.add_trace(go.Scatter(
                x=[sample_prop, sample_prop], y=[ci_lower_std, ci_upper_std],
                mode='lines', line=dict(color='red', width=2),
                name='Standard CI'
            ))
            
            fig.add_trace(go.Scatter(
                x=[sample_prop, sample_prop], y=[ci_lower_logit, ci_upper_logit],
                mode='lines', line=dict(color='green', width=2),
                name='Logit-transformed CI'
            ))
            
            fig.add_trace(go.Scatter(
                x=[sample_prop, sample_prop], y=[ci_lower_wilson, ci_upper_wilson],
                mode='lines', line=dict(color='blue', width=2),
                name='Wilson score CI'
            ))
            
            # Add true proportion
            fig.add_trace(go.Scatter(
                x=[true_prop], y=[true_prop],
                mode='markers', marker=dict(color='black', size=10, symbol='x'),
                name=f'True Proportion: {true_prop:.3f}'
            ))
            
            # Update layout
            fig.update_layout(
                title='Confidence Interval Bands for Proportions',
                xaxis_title='Proportion',
                yaxis_title='Proportion',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a comparison table
            comp_df = pd.DataFrame({
                'Method': ['Standard (Wald)', 'Logit-transformed', 'Wilson score'],
                'Lower Bound': [ci_lower_std, ci_lower_logit, ci_lower_wilson],
                'Upper Bound': [ci_upper_std, ci_upper_logit, ci_upper_wilson],
                'Interval Width': [
                    ci_upper_std - ci_lower_std, 
                    ci_upper_logit - ci_lower_logit,
                    ci_upper_wilson - ci_lower_wilson
                ],
                'Contains True Value': [
                    f"{ci_lower_std <= true_prop <= ci_upper_std}",
                    f"{ci_lower_logit <= true_prop <= ci_upper_logit}",
                    f"{ci_lower_wilson <= true_prop <= ci_upper_wilson}"
                ]
            })
            
            st.subheader("Comparison of Confidence Interval Methods")
            st.dataframe(comp_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Interval Width': '{:.4f}'
            }))
            
            # Add interpretation
            st.subheader("Interpretation")
            
            std_contains = ci_lower_std <= true_prop <= ci_upper_std
            logit_contains = ci_lower_logit <= true_prop <= ci_upper_logit
            wilson_contains = ci_lower_wilson <= true_prop <= ci_upper_wilson
            
            near_boundary = sample_prop < 0.1 or sample_prop > 0.9
            
            st.markdown(f"""
            #### Key Findings:
            
            1. **Standard (Wald) CI**: [{ci_lower_std:.4f}, {ci_upper_std:.4f}]
               - {"Contains" if std_contains else "Does not contain"} the true proportion of {true_prop}
               - Known to perform poorly near boundaries (0 or 1)
            
            2. **Logit-transformed CI**: [{ci_lower_logit:.4f}, {ci_upper_logit:.4f}]
               - {"Contains" if logit_contains else "Does not contain"} the true proportion of {true_prop}
               - Better respects the [0,1] boundaries
               - Asymmetric in original scale (wider toward the middle of [0,1])
            
            3. **Wilson score CI**: [{ci_lower_wilson:.4f}, {ci_upper_wilson:.4f}]
               - {"Contains" if wilson_contains else "Does not contain"} the true proportion of {true_prop}
               - Generally recommended for most practical applications
               - Performs well across the entire [0,1] range
            
            #### Why Transformations Help for Proportions:
            
            - The binomial distribution becomes increasingly skewed as the proportion approaches 0 or 1
            - The logit transformation maps [0,1] to (-∞,∞), allowing for symmetric intervals in the transformed space
            - Near the boundaries, standard intervals can extend beyond [0,1], leading to invalid confidence limits
            - Logit transformation automatically respects the natural boundary constraints
            
            {"The advantages of transformation are particularly evident in this case where the sample proportion is near a boundary." if near_boundary else "With proportions near 0.5, all methods tend to perform similarly, but differences become more pronounced near the boundaries."}
            """)
            
        elif transform_type == "Fisher's Z Transformation for Correlation":
            # Generate bivariate normal data with specified correlation
            n = sample_size
            rho = true_corr
            
            # Create covariance matrix
            cov_matrix = np.array([[1, rho], [rho, 1]])
            
            # Generate correlated data
            data = np.random.multivariate_normal([0, 0], cov_matrix, n)
            x = data[:, 0]
            y = data[:, 1]
            
            # Calculate sample correlation
            sample_corr = np.corrcoef(x, y)[0, 1]
            
            # Standard CI for correlation (using approximate formula)
            # The approximation is not great for extreme correlations or small samples
            se_r = np.sqrt((1 - sample_corr**2) / (n - 2))
            t_crit = stats.t.ppf(1 - (1 - conf_level)/2, n - 2)
            
            ci_lower_std = max(-1, sample_corr - t_crit * se_r)
            ci_upper_std = min(1, sample_corr + t_crit * se_r)
            
            # Fisher's Z transformation
            # Transform r to z
            z = np.arctanh(sample_corr)
            
            # Standard error in z-space is constant
            se_z = 1 / np.sqrt(n - 3)
            
            # CI in z-space
            z_crit = stats.norm.ppf(1 - (1 - conf_level)/2)
            ci_lower_z = z - z_crit * se_z
            ci_upper_z = z + z_crit * se_z
            
            # Transform back to r-space
            ci_lower_fisher = np.tanh(ci_lower_z)
            ci_upper_fisher = np.tanh(ci_upper_z)
            
            # Create scatter plot with correlation
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Sample Data'
            ))
            
            # Add regression line
            line_x = np.array([min(x), max(x)])
            line_y = sample_corr * line_x  # Standardized data, so intercept is 0
            
            fig1.add_trace(go.Scatter(
                x=line_x, y=line_y,
                mode='lines', line=dict(color='red', width=2),
                name=f'r = {sample_corr:.3f}'
            ))
            
            fig1.update_layout(
                title=f'Scatter Plot (r = {sample_corr:.3f}, true ρ = {true_corr:.3f})',
                xaxis_title='X',
                yaxis_title='Y',
                height=400
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create visualization comparing standard and Fisher's z CIs
            fig2 = go.Figure()
            
            # Create range of correlations
            r_range = np.linspace(-0.999, 0.999, 1000)
            
            # Calculate standard error for each r
            se_range = np.sqrt((1 - r_range**2) / (n - 2))
            
            # Standard CI bounds
            upper_std = np.minimum(1, r_range + t_crit * se_range)
            lower_std = np.maximum(-1, r_range - t_crit * se_range)
            
            # Fisher's z transformation
            z_range = np.arctanh(r_range)
            upper_z = z_range + z_crit * se_z
            lower_z = z_range - z_crit * se_z
            
            # Back-transform to r
            upper_fisher = np.tanh(upper_z)
            lower_fisher = np.tanh(lower_z)
            
            # Add standard CI band
            fig2.add_trace(go.Scatter(
                x=r_range, y=upper_std,
                mode='lines', line=dict(width=0),
                showlegend=False
            ))
            
            fig2.add_trace(go.Scatter(
                x=r_range, y=lower_std,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)',
                name='Standard CI Band'
            ))
            
            # Add Fisher's z CI band
            fig2.add_trace(go.Scatter(
                x=r_range, y=upper_fisher,
                mode='lines', line=dict(width=0),
                showlegend=False
            ))
            
            fig2.add_trace(go.Scatter(
                x=r_range, y=lower_fisher,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0, 128, 0, 0.1)',
                name="Fisher's z CI Band"
            ))
            
            # Add reference line
            fig2.add_trace(go.Scatter(
                x=[-1, 1], y=[-1, 1],
                mode='lines', line=dict(color='black', width=1, dash='dash'),
                name='Reference Line'
            ))
            
            # Add point for actual sample
            fig2.add_trace(go.Scatter(
                x=[sample_corr], y=[sample_corr],
                mode='markers', marker=dict(color='red', size=10),
                name=f'Sample r: {sample_corr:.3f}'
            ))
            
            # Add true correlation
            fig2.add_trace(go.Scatter(
                x=[true_corr], y=[true_corr],
                mode='markers', marker=dict(color='black', size=10, symbol='x'),
                name=f'True ρ: {true_corr:.3f}'
            ))
            
            # Update layout
            fig2.update_layout(
                title="Confidence Interval Bands for Correlation Coefficient",
                xaxis_title='Correlation (r)',
                yaxis_title='Correlation (r)',
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Create a figure comparing the CIs for this specific sample
            fig3 = go.Figure()
            
            # Add CI intervals as horizontal lines
            fig3.add_trace(go.Scatter(
                x=[ci_lower_std, ci_upper_std], y=[1, 1],
                mode='lines', line=dict(color='red', width=4),
                name='Standard CI'
            ))
            
            fig3.add_trace(go.Scatter(
                x=[ci_lower_fisher, ci_upper_fisher], y=[2, 2],
                mode='lines', line=dict(color='green', width=4),
                name="Fisher's z CI"
            ))
            
            # Add sample correlation
            fig3.add_vline(x=sample_corr, line=dict(color='blue', width=2),
                         annotation=dict(text=f"Sample r: {sample_corr:.3f}", showarrow=False))
            
            # Add true correlation
            fig3.add_vline(x=true_corr, line=dict(color='black', width=2, dash='dash'),
                         annotation=dict(text=f"True ρ: {true_corr:.3f}", showarrow=False))
            
            # Update layout
            fig3.update_layout(
                title=f"{conf_level*100:.0f}% Confidence Intervals for Correlation",
                xaxis_title='Correlation Coefficient',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2],
                    ticktext=['Standard', "Fisher's z"],
                    showgrid=False
                ),
                height=300,
                xaxis=dict(range=[-1, 1])
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Create a comparison table
            comp_df = pd.DataFrame({
                'Method': ['Standard', "Fisher's z"],
                'Lower Bound': [ci_lower_std, ci_lower_fisher],
                'Upper Bound': [ci_upper_std, ci_upper_fisher],
                'Interval Width': [ci_upper_std - ci_lower_std, ci_upper_fisher - ci_lower_fisher],
                'Symmetric Around r': [
                    f"{abs((sample_corr - ci_lower_std) - (ci_upper_std - sample_corr)) < 0.001}",
                    f"{abs((sample_corr - ci_lower_fisher) - (ci_upper_fisher - sample_corr)) < 0.001}"
                ],
                'Contains True Value': [
                    f"{ci_lower_std <= true_corr <= ci_upper_std}",
                    f"{ci_lower_fisher <= true_corr <= ci_upper_fisher}"
                ]
            })
            
            st.subheader("Comparison of Confidence Interval Methods")
            st.dataframe(comp_df.style.format({
                'Lower Bound': '{:.4f}',
                'Upper Bound': '{:.4f}',
                'Interval Width': '{:.4f}'
            }))
            
            # Add interpretation
            st.subheader("Interpretation")
            
            std_contains = ci_lower_std <= true_corr <= ci_upper_std
            fisher_contains = ci_lower_fisher <= true_corr <= ci_upper_fisher
            high_corr = abs(sample_corr) > 0.7
            
            st.markdown(f"""
            #### Key Findings:
            
            1. **Standard CI**: [{ci_lower_std:.4f}, {ci_upper_std:.4f}]
               - {"Contains" if std_contains else "Does not contain"} the true correlation of {true_corr}
               - Based on the t-distribution approximation
               - {'Symmetric around the sample correlation' if abs((sample_corr - ci_lower_std) - (ci_upper_std - sample_corr)) < 0.01 else 'Not symmetric around the sample correlation'}
               - Can extend beyond [-1, 1] for extreme correlations (requiring truncation)
            
            2. **Fisher's z CI**: [{ci_lower_fisher:.4f}, {ci_upper_fisher:.4f}]
               - {"Contains" if fisher_contains else "Does not contain"} the true correlation of {true_corr}
               - Based on variance-stabilizing transformation
               - Always within [-1, 1] without truncation
               - {'Symmetric in z-space, but asymmetric in r-space' if abs(ci_upper_z - z) - abs(z - ci_lower_z) < 0.01 else 'Not symmetric in z-space'}
            
            #### Why Fisher's z Transformation Works Better:
            
            - The sampling distribution of the correlation coefficient becomes increasingly skewed as |r| approaches 1
            - The variance of r depends on the true correlation (larger variance when ρ is close to 0)
            - Fisher's z transformation stabilizes the variance (SE is approximately 1/√(n-3) regardless of the true correlation)
            - Fisher's z transformation automatically respects the [-1, 1] boundaries
            
            {"The advantages of Fisher's z transformation are particularly evident for high correlations like in this sample." if high_corr else "Even for moderate correlations, Fisher's z transformation provides more accurate coverage."}
            
            #### Practical recommendation:
            
            - For small samples (n < 50) or high correlations (|r| > 0.5), Fisher's z transformation is strongly recommended
            - The standard method is adequate for large samples with moderate correlations
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
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Set random seed
        np.random.seed(None)
        
        # Define distributions
        if distribution == "Normal":
            dist_name = "Normal"
            def generate_sample(n):
                return np.random.normal(0, 1, n)
            skewness = 0
            kurtosis = 3  # Normal kurtosis
            
        elif distribution == "Skewed (Log-normal)":
            dist_name = "Log-normal"
            # Parameters chosen to have mean ≈ 0 and variance ≈ 1 after transformation
            sigma = 0.6
            mu = -sigma**2/2
            def generate_sample(n):
                return np.random.lognormal(mu, sigma, n) - np.exp(mu + sigma**2/2)
            skewness = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)
            kurtosis = np.exp(4*sigma**2) + 2*np.exp(3*sigma**2) + 3*np.exp(2*sigma**2) - 6
            
        elif distribution == "Heavy-tailed (t with df=3)":
            dist_name = "t(3)"
            def generate_sample(n):
                return stats.t.rvs(3, size=n) * np.sqrt(3/5)  # Scale to variance=1
            skewness = 0
            kurtosis = float('inf')  # Undefined for df < 5
            
        elif distribution == "Bimodal":
            dist_name = "Bimodal Mixture"
            def generate_sample(n):
                comp1 = np.random.normal(-1.5, 0.5, n)
                comp2 = np.random.normal(1.5, 0.5, n)
                mix = np.random.binomial(1, 0.5, n)
                return comp1 * (1 - mix) + comp2 * mix
            skewness = 0  # Symmetric mixture
            kurtosis = 1.8  # Lower than normal (flatter)
            
        elif distribution == "Contaminated Normal":
            dist_name = "Contaminated Normal"
            def generate_sample(n):
                main = np.random.normal(0, 0.9, n)
                outliers = np.random.normal(0, 5, n)
                mix = np.random.binomial(1, 0.05, n)  # 5% outliers
                return main * (1 - mix) + outliers * mix
            skewness = 0  # Symmetric
            kurtosis = 15  # Much higher than normal (heavy tails)
        
        # Run simulations
        contains_true = 0
        interval_widths = []
        medians = []
        
        # Generate a fixed "population" to compare with
        population = generate_sample(10000)
        true_mean = np.mean(population)
        true_median = np.median(population)
        true_variance = np.var(population, ddof=1)
        
        for i in range(n_sims):
            # Update progress
            if i % 50 == 0:  # Update every 50 iterations
                progress_bar.progress((i + 1) / n_sims)
                status_text.text(f"Running simulation {i+1}/{n_sims}")
            
            # Generate sample
            sample = generate_sample(sample_size)
            
            if parameter == "Mean":
                # Calculate sample statistics
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                
                # Calculate t-interval
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                margin = t_crit * sample_std / np.sqrt(sample_size)
                
                lower = sample_mean - margin
                upper = sample_mean + margin
                
                # Check if interval contains true value
                contains_true += (lower <= true_mean <= upper)
                interval_widths.append(upper - lower)
                
            elif parameter == "Median":
                # Calculate bootstrap confidence interval for median
                bootstrap_medians = []
                n_boot = 1000
                
                for _ in range(n_boot):
                    boot_sample = np.random.choice(sample, size=sample_size, replace=True)
                    bootstrap_medians.append(np.median(boot_sample))
                
                # Percentile interval
                lower = np.percentile(bootstrap_medians, 100 * (1 - conf_level) / 2)
                upper = np.percentile(bootstrap_medians, 100 * (1 - (1 - conf_level) / 2))
                
                # Check if interval contains true value
                contains_true += (lower <= true_median <= upper)
                interval_widths.append(upper - lower)
                medians.append(np.median(sample))
                
            elif parameter == "Variance":
                # Calculate sample variance
                sample_var = np.var(sample, ddof=1)
                
                # Chi-square interval
                chi2_lower = stats.chi2.ppf((1 - conf_level)/2, sample_size - 1)
                chi2_upper = stats.chi2.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                
                upper = (sample_size - 1) * sample_var / chi2_lower
                lower = (sample_size - 1) * sample_var / chi2_upper
                
                # Check if interval contains true value
                contains_true += (lower <= true_variance <= upper)
                interval_widths.append(upper - lower)
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
        # Calculate actual coverage and interval width
        actual_coverage = contains_true / n_sims * 100
        avg_width = np.mean(interval_widths)
        
        # Create results display
        st.success(f"Simulation complete! {n_sims} intervals generated.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=f"Actual Coverage of {parameter}",
                value=f"{actual_coverage:.1f}%",
                delta=f"{actual_coverage - conf_level*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Average Interval Width",
                value=f"{avg_width:.4f}"
            )
        
        # Create additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Distribution Skewness",
                value=f"{skewness:.4f}"
            )
        
        with col2:
            st.metric(
                label="Distribution Kurtosis",
                value=f"{kurtosis if kurtosis != float('inf') else '∞'}"
            )
        
        with col3:
            st.metric(
                label="Sample Size",
                value=f"{sample_size}"
            )
        
        # Create coverage visualization
        fig1 = go.Figure()
        
        # Add bar for actual coverage
        fig1.add_trace(go.Bar(
            x=['Actual Coverage'],
            y=[actual_coverage],
            name='Actual Coverage',
            marker_color='blue'
        ))
        
        # Add bar for nominal coverage
        fig1.add_trace(go.Bar(
            x=['Nominal Coverage'],
            y=[conf_level * 100],
            name='Nominal Coverage',
            marker_color='green'
        ))
        
        fig1.update_layout(
            title=f'Coverage Probability for {parameter} ({dist_name} Distribution)',
            yaxis=dict(title='Coverage (%)', range=[min(actual_coverage, conf_level*100)*0.95, 
                                               max(actual_coverage, conf_level*100)*1.05]),
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Generate data for distribution visualization
        x = np.linspace(-4, 4, 1000)
        normal_pdf = stats.norm.pdf(x)
        
        # Get histogram data for the actual distribution
        sample_large = generate_sample(10000)
        hist_data, hist_bins = np.histogram(sample_large, bins=50, density=True)
        hist_x = (hist_bins[:-1] + hist_bins[1:]) / 2
        
        # Create distribution visualization
        fig2 = go.Figure()
        
        # Add normal distribution curve
        fig2.add_trace(go.Scatter(
            x=x, y=normal_pdf,
            mode='lines',
            name='Standard Normal',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # Add histogram of actual distribution
        fig2.add_trace(go.Bar(
            x=hist_x, y=hist_data,
            name=dist_name,
            marker_color='rgba(255, 0, 0, 0.5)',
            marker_line_width=0
        ))
        
        fig2.update_layout(
            title=f'Comparison of {dist_name} vs. Normal Distribution',
            xaxis_title='Value',
            yaxis_title='Density',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Create visualization of sample statistics
        if parameter == "Median":
            # Create a histogram of sample medians
            fig3 = go.Figure()
            
            fig3.add_trace(go.Histogram(
                x=medians,
                nbinsx=30,
                marker_color='rgba(0, 128, 0, 0.5)',
                name='Sample Medians'
            ))
            
            # Add vertical line for true median
            fig3.add_vline(x=true_median, line=dict(color='red', width=2),
                         annotation=dict(text=f"True Median: {true_median:.4f}", showarrow=False))
            
            fig3.update_layout(
                title='Sampling Distribution of Median',
                xaxis_title='Median Value',
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Add interpretation
        st.subheader("Interpretation")
        
        if abs(actual_coverage - conf_level*100) < 2:
            coverage_quality = "excellent"
        elif abs(actual_coverage - conf_level*100) < 5:
            coverage_quality = "good"
        else:
            coverage_quality = "poor"
        
        if parameter == "Mean":
            if distribution == "Normal":
                interpretation = """
                For normally distributed data, the t-interval provides excellent coverage for the mean at any sample size. This is because the t-distribution exactly models the sampling distribution of the standardized mean for normal data.
                """
            elif distribution == "Skewed (Log-normal)":
                interpretation = """
                For skewed distributions like log-normal, the actual coverage of t-intervals for the mean can be below the nominal level, especially with small samples. This is because the Central Limit Theorem requires larger samples to overcome skewness in the parent distribution.
                """
            elif distribution == "Heavy-tailed (t with df=3)":
                interpretation = """
                With heavy-tailed distributions like the t with 3 degrees of freedom, traditional t-intervals can have poor coverage because outliers can drastically affect the sample mean and standard deviation. Larger samples or robust methods would be more appropriate.
                """
            elif distribution == "Bimodal":
                interpretation = """
                For bimodal distributions, t-intervals may have reasonably good coverage for the mean if the modes are symmetric. However, the mean itself may not be a representative measure of central tendency for such data.
                """
            else:  # Contaminated Normal
                interpretation = """
                With contaminated normal distributions (containing outliers), t-intervals tend to have below-nominal coverage because outliers inflate the standard deviation without providing proportional information about the mean.
                """
                
        elif parameter == "Median":
            if distribution == "Normal":
                interpretation = """
                For normal distributions, bootstrap confidence intervals for the median work well, though they may be wider than needed since the mean would be more efficient for symmetric data.
                """
            elif distribution == "Skewed (Log-normal)":
                interpretation = """
                For skewed distributions, bootstrap confidence intervals for the median often perform better than parametric intervals for the mean, since the median is less affected by extreme values in the tail.
                """
            elif distribution == "Heavy-tailed (t with df=3)":
                interpretation = """
                With heavy-tailed distributions, the median is a more robust measure of central tendency than the mean, and bootstrap intervals typically provide good coverage even with moderate sample sizes.
                """
            elif distribution == "Bimodal":
                interpretation = """
                For bimodal distributions, confidence intervals for the median may have good coverage but the median itself might not be the most informative summary statistic, as it falls between the two modes.
                """
            else:  # Contaminated Normal
                interpretation = """
                With contaminated normal distributions, bootstrap confidence intervals for the median are much more reliable than intervals for the mean, since the median is robust to outliers.
                """
                
        elif parameter == "Variance":
            if distribution == "Normal":
                interpretation = """
                For normal distributions, chi-square intervals for the variance have exact coverage regardless of sample size.
                """
            elif distribution == "Skewed (Log-normal)":
                interpretation = """
                For skewed distributions, standard chi-square intervals for the variance tend to have poor coverage because the sampling distribution of the variance is no longer chi-square distributed.
                """
            elif distribution == "Heavy-tailed (t with df=3)":
                interpretation = """
                With heavy-tailed distributions, chi-square intervals for the variance perform very poorly because outliers have an extreme effect on the variance, and the sampling distribution is far from chi-square.
                """
            elif distribution == "Bimodal":
                interpretation = """
                For bimodal distributions, standard variance intervals may have below-nominal coverage as the chi-square approximation becomes less accurate.
                """
            else:  # Contaminated Normal
                interpretation = """
                With contaminated normal distributions (containing outliers), chi-square intervals for the variance typically have very poor coverage due to the extreme influence of outliers on the variance estimation.
                """
        
        st.markdown(f"""
        ### Impact of Non-normality on {parameter} Confidence Intervals
        
        **Key Results**:
        
        - **Distribution**: {dist_name} (Skewness: {skewness:.2f}, Kurtosis: {kurtosis if kurtosis != float('inf') else '∞'})
        - **Nominal coverage**: {conf_level*100:.1f}%
        - **Actual coverage**: {actual_coverage:.1f}%
        - **Coverage quality**: {coverage_quality.title()}
        - **Average interval width**: {avg_width:.4f}
        
        **Specific interpretation for this scenario**:
        
        {interpretation}
        
        **General recommendations for non-normal data**:
        
        1. **For the mean**: 
           - Increase sample size to rely on the Central Limit Theorem
           - Consider bootstrap confidence intervals
           - For heavily skewed data, consider transforming the data
        
        2. **For the median**:
           - Bootstrap or rank-based methods generally work well
           - More robust to outliers and skewness than the mean
        
        3. **For the variance**:
           - Standard methods are highly sensitive to non-normality
           - Consider robust variance estimators or bootstrap methods
           - Data transformations may help in some cases
        
        Remember that no confidence interval method is universally best - the choice depends on the specific properties of your data and the parameter of interest.
        """)
        
        # Compare with normal data coverage
        st.subheader("Comparison with Normal Data")
        
        # Quick simulation with normal data
        normal_contains = 0
        normal_widths = []
        
        for i in range(min(1000, n_sims)):  # Limit to 1000 for speed
            normal_sample = np.random.normal(0, 1, sample_size)
            
            if parameter == "Mean":
                normal_mean = np.mean(normal_sample)
                normal_std = np.std(normal_sample, ddof=1)
                
                t_crit = stats.t.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                margin = t_crit * normal_std / np.sqrt(sample_size)
                
                lower = normal_mean - margin
                upper = normal_mean + margin
                
                normal_contains += (lower <= 0 <= upper)
                normal_widths.append(upper - lower)
                
            elif parameter == "Median":
                # Using similar bootstrap approach
                boot_medians = []
                n_boot = 200  # Reduced for speed
                
                for _ in range(n_boot):
                    boot_sample = np.random.choice(normal_sample, size=sample_size, replace=True)
                    boot_medians.append(np.median(boot_sample))
                
                lower = np.percentile(boot_medians, 100 * (1 - conf_level) / 2)
                upper = np.percentile(boot_medians, 100 * (1 - (1 - conf_level) / 2))
                
                normal_contains += (lower <= 0 <= upper)
                normal_widths.append(upper - lower)
                
            elif parameter == "Variance":
                sample_var = np.var(normal_sample, ddof=1)
                
                chi2_lower = stats.chi2.ppf((1 - conf_level)/2, sample_size - 1)
                chi2_upper = stats.chi2.ppf(1 - (1 - conf_level)/2, sample_size - 1)
                
                upper = (sample_size - 1) * sample_var / chi2_lower
                lower = (sample_size - 1) * sample_var / chi2_upper
                
                normal_contains += (lower <= 1 <= upper)
                normal_widths.append(upper - lower)
        
        normal_coverage = normal_contains / min(1000, n_sims) * 100
        normal_width = np.mean(normal_widths)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Coverage with Normal Data",
                value=f"{normal_coverage:.1f}%",
                delta=f"{normal_coverage - actual_coverage:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Width Ratio (Non-normal / Normal)",
                value=f"{avg_width / normal_width:.2f}",
                delta=f"{avg_width - normal_width:.4f}"
            )
        
        st.markdown(f"""
        The confidence interval for the {parameter} is {"wider" if avg_width > normal_width else "narrower"} with the {dist_name} distribution compared to normal data, while providing {"better" if actual_coverage > normal_coverage else "worse"} coverage.
        
        This demonstrates that non-normality can significantly affect both the coverage and precision of standard confidence interval methods.
        """)

# Add footer
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)