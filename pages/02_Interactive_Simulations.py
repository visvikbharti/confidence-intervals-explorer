import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Interactive Simulations - CI Explorer",
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
        # Implementation of transformations and simulations
        st.info("This section will simulate confidence intervals with various transformations.")

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
        # Implementation of non-normality impact simulation
        st.info("This section will demonstrate the effect of non-normality on confidence interval performance.")

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