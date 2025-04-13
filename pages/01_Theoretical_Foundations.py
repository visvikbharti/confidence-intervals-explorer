import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Theoretical Foundations - CI Explorer",
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

st.title("Theoretical Foundations of Confidence Intervals")

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
        
        # Add horizontal line for true parameter (without annotation)
        fig.add_hline(y=true_mu, line=dict(color='red', width=2, dash='dash'))

        # Add annotation separately
        fig.add_annotation(
            text="True μ = 50", 
            x=1.02, 
            y=true_mu, 
            xref="paper", 
            yref="y", 
            showarrow=False
        )
        
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
        
        $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$
        
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
        
        # Interactive demonstration - to be implemented similar to the known variance case
        # with t-distribution instead of normal

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

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; padding: 10px;'>
        <p>Designed and developed by Vishal Bharti © 2025 | PhD-Level Confidence Intervals Explorer</p>
    </div>
    """, 
    unsafe_allow_html=True
)