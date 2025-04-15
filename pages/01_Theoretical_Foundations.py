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
    page_title="Theoretical Foundations - CI Explorer",
    page_icon="📊",
    layout="wide",
)

# Then after st.set_page_config():
st.markdown(get_custom_css(), unsafe_allow_html=True)
add_latex_styles()
setup_math_rendering()

# Apply the custom CSS - this is critical for proper math rendering
st.markdown(get_custom_css(), unsafe_allow_html=True)
force_visible_math_mode()
inline_math_fix()

st.title("Theoretical Foundations of Confidence Intervals")

tabs = st.tabs(["Definitions & Properties", "Statistical Theory", "Derivations", "Optimality", "Interpretation"])

with tabs[0]:  # Definitions & Properties
    st.subheader("Formal Definitions")
    
    # Use the render_definition helper instead of raw markdown for better theming
    render_definition("""
    <strong>Definition 1:</strong> A confidence interval for a parameter $\\theta$ is a pair of statistics $L(X)$ and $U(X)$ such that:
    
    $$P_{\\theta}(L(X) \\leq \\theta \\leq U(X)) \\geq 1-\\alpha \\quad \\forall \\theta \\in \\Theta$$
    
    where $1-\\alpha$ is the confidence level, and $\\Theta$ is the parameter space.
    """)
    
    render_definition("""
    <strong>Definition 2:</strong> A random interval $[L(X), U(X)]$ is an <em>exact</em> confidence interval if:
    
    $$P_{\\theta}(L(X) \\leq \\theta \\leq U(X)) = 1-\\alpha \\quad \\forall \\theta \\in \\Theta$$
    """)
    
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
    
    st.markdown("### Pivotal Quantities")
    
    st.markdown("""
    A **pivotal quantity** is a function of both the data and the parameter whose distribution does not depend on any unknown parameter.
    """)
    
    render_definition("""
    <strong>Definition:</strong> A statistic $Q(X, \\theta)$ is a pivotal quantity if its distribution is the same for all $\\theta \\in \\Theta$.
    """)
    
    render_example("""
    <strong>Example:</strong> For a normal sample with unknown mean $\\mu$ and known variance $\\sigma^2$, the quantity
    $$Z = \\frac{\\bar{X} - \\mu}{\\sigma/\\sqrt{n}}$$
    follows a standard normal distribution regardless of the value of $\\mu$, making it a pivotal quantity.
    """)
    
    st.markdown("### Relationship with Hypothesis Testing")

    st.markdown("There is a fundamental duality between confidence intervals and hypothesis testing:")

    render_definition("""
    A $(1-\\alpha)$ confidence interval for $\\theta$ contains precisely those values that would not be rejected by a level-$\\alpha$ test of $H_0: \\theta = \\theta_0$
    """)

    st.markdown("This relationship can be expressed formally as: θ₀ ∈ CI₁₋ₐ(X) if and only if the test of H₀: θ = θ₀ is not rejected at level α")

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
    
    elif interval_type == "Binomial Proportion":
        st.markdown(r"""
        ### Derivation of Confidence Intervals for Binomial Proportion
        
        For a binomial random variable $X \sim \text{Bin}(n, p)$ where $p$ is the unknown proportion:
        
        ### 1. Wald (Standard) Interval
        
        The Wald interval is based on the normal approximation to the binomial distribution.
        
        #### Step 1: Define the sample proportion
        
        The sample proportion is $\hat{p} = X/n$, with:
        
        $$E(\hat{p}) = p$$
        $$\text{Var}(\hat{p}) = \frac{p(1-p)}{n}$$
        
        #### Step 2: Apply the Central Limit Theorem
        
        For large $n$, by the Central Limit Theorem:
        
        $$\frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}} \stackrel{\text{approx}}{\sim} N(0, 1)$$
        
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
        
        - Lower bound: $p_L = \text{Beta}_{\alpha/2}(k, n-k+1)$
        - Upper bound: $p_U = \text{Beta}_{1-\alpha/2}(k+1, n-k)$
        
        where $\text{Beta}_q(a, b)$ is the $q$ quantile of the Beta$(a, b)$ distribution.
        
        **Properties:**
        - Guaranteed coverage: Always has coverage probability $\geq 1-\alpha$
        - Often conservative (wider than necessary)
        - Always within [0,1]
        - Based directly on the binomial distribution, not approximations
        
        ### Mathematical Proof for Coverage Properties
        
        Let's give a formal proof of why the Clopper-Pearson interval guarantees coverage.
        
        For $X \sim \text{Bin}(n, p)$, the Clopper-Pearson interval $[L(X), U(X)]$ has the property:
        
        $$P_p(L(X) \leq p \leq U(X)) \geq 1-\alpha \quad \forall p \in [0,1]$$
        
        **Proof:**
        
        For a given $k = 0, 1, ..., n$:
        
        $L(k)$ is chosen such that $P(X \geq k | p = L(k)) = \alpha/2$
        
        $U(k)$ is chosen such that $P(X \leq k | p = U(k)) = \alpha/2$
        
        We need to show that $P_p(L(X) \leq p \leq U(X)) \geq 1-\alpha$ for any $p \in [0,1]$.
        
        For a fixed $p$:
        
        1. If $p < L(k)$, then $P(X \geq k | p) < P(X \geq k | L(k)) = \alpha/2$ (since probability increases with $p$)
        
        2. If $p > U(k)$, then $P(X \leq k | p) < P(X \leq k | U(k)) = \alpha/2$
        
        3. For $p$ to be outside the interval, either $p < L(X)$ or $p > U(X)$
        
        4. The probability of $p < L(X)$ is:
        $$P_p(p < L(X)) = P_p(\text{all } k \text{ where } p < L(k)) \leq \alpha/2$$
        
        5. Similarly, $P_p(p > U(X)) \leq \alpha/2$
        
        6. By the union bound:
        $$P_p(p < L(X) \text{ or } p > U(X)) \leq \alpha/2 + \alpha/2 = \alpha$$
        
        7. Therefore:
        $$P_p(L(X) \leq p \leq U(X)) = 1 - P_p(p < L(X) \text{ or } p > U(X)) \geq 1 - \alpha$$
        
        ### Coverage Comparison and Professional Applications
        
        For a $(1-\alpha)$ confidence interval, the actual coverage probability is:
        
        $$C(p) = \sum_{k=0}^n I_{[k \in \{k: p \in [L(k), U(k)]\}]} \binom{n}{k} p^k (1-p)^{n-k}$$
        
        where $I$ is the indicator function, and $L(k)$ and $U(k)$ are the lower and upper bounds when $X = k$.
        
        **Industry Applications:**
        
        In professional settings, the choice of methods depends on the specific context:
        
        1. **Clinical Trials**: Regulatory agencies often require Clopper-Pearson intervals for safety outcomes to be conservative
        
        2. **Quality Control**: Wilson methods are preferred for manufacturing processes where both accuracy and efficiency are important
        
        3. **A/B Testing**: For web experiments with large samples, Wald intervals with continuity correction are efficient
        
        4. **Risk Assessment**: For rare events, Clopper-Pearson or Bayesian methods provide safer bounds on risk probabilities
        
        **Recommendations for Professional Use:**
        
        - **High-Stakes Decisions**: Use Clopper-Pearson for guaranteed coverage
        - **Large Samples**: Wald intervals are computationally efficient
        - **Balanced Approach**: Wilson score intervals offer a good trade-off between accuracy and width
        - **Small Samples or Extreme Proportions**: Avoid Wald intervals; use Wilson or exact methods
        """, unsafe_allow_html=True)
        
        # Interactive demonstration
        st.subheader("Interactive Demonstration")
        
        col1, col2 = st.columns(2)
        with col1:
            demo_p = st.slider("True population proportion (p)", 0.01, 0.99, 0.3, 0.01)
            demo_n = st.slider("Sample size (n)", 5, 200, 30, 5)
        with col2:
            demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
            demo_method = st.selectbox("Interval method", ["Wald", "Wilson", "Clopper-Pearson"])
        
        if st.button("Generate Sample and Confidence Interval", key="gen_binom_ci"):
            # Generate sample
            np.random.seed(None)  # Use a different seed each time
            sample = np.random.binomial(1, demo_p, demo_n)
            successes = np.sum(sample)
            sample_prop = successes / demo_n
            
            # Calculate CI based on selected method
            z_critical = stats.norm.ppf(1 - demo_alpha/2)
            
            if demo_method == "Wald":
                # Wald interval
                std_err = np.sqrt(sample_prop * (1 - sample_prop) / demo_n)
                margin = z_critical * std_err
                ci_lower = max(0, sample_prop - margin)
                ci_upper = min(1, sample_prop + margin)
                method_name = "Wald"
                
            elif demo_method == "Wilson":
                # Wilson score interval
                denominator = 1 + z_critical**2 / demo_n
                center = (sample_prop + z_critical**2 / (2 * demo_n)) / denominator
                margin = z_critical * np.sqrt(sample_prop * (1 - sample_prop) / demo_n + z_critical**2 / (4 * demo_n**2)) / denominator
                ci_lower = max(0, center - margin)
                ci_upper = min(1, center + margin)
                method_name = "Wilson Score"
                
            else:  # "Clopper-Pearson"
                # Clopper-Pearson exact interval
                ci_lower = stats.beta.ppf(demo_alpha/2, successes, demo_n - successes + 1) if successes > 0 else 0
                ci_upper = stats.beta.ppf(1 - demo_alpha/2, successes + 1, demo_n - successes) if successes < demo_n else 1
                method_name = "Clopper-Pearson (Exact)"
            
            # Display results
            st.markdown(f"""
            **Sample Results**:
            - Number of trials (n): {demo_n}
            - Number of successes: {successes}
            - Sample proportion: {sample_prop:.4f}
            
            **{method_name} {(1-demo_alpha)*100:.0f}% Confidence Interval**:
            - Lower bound: {ci_lower:.4f}
            - Upper bound: {ci_upper:.4f}
            - Interval width: {ci_upper - ci_lower:.4f}
            """)
            
            # Visualization
            fig = go.Figure()
            
            # Add vertical line for sample proportion
            fig.add_vline(x=sample_prop, line=dict(color='blue', width=2),
                        annotation=dict(text=f"Sample proportion: {sample_prop:.4f}", showarrow=False))
            
            # Add vertical line for true proportion
            fig.add_vline(x=demo_p, line=dict(color='green', width=2, dash='dash'),
                        annotation=dict(text=f"True proportion: {demo_p}", showarrow=False))
            
            # Add confidence interval
            fig.add_shape(
                type="rect",
                x0=ci_lower, x1=ci_upper,
                y0=0.2, y1=0.8,
                fillcolor="rgba(0, 100, 80, 0.2)",
                line=dict(color="rgba(0, 100, 80, 0.4)", width=2),
            )
            
            # Add annotation for confidence interval
            fig.add_annotation(
                x=(ci_lower + ci_upper)/2, y=0.5,
                text=f"{(1-demo_alpha)*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
                showarrow=False,
                font=dict(color="rgba(0, 100, 80, 1)")
            )
            
            # Update layout
            fig.update_layout(
                title=f"{method_name} Confidence Interval for Proportion",
                xaxis=dict(title="Proportion", range=[0, 1]),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of results
            contains_true = ci_lower <= demo_p <= ci_upper
            st.markdown(f"""
            **Interpretation**:
            
            The {method_name} {(1-demo_alpha)*100:.0f}% confidence interval is [{ci_lower:.4f}, {ci_upper:.4f}].
            
            This interval {"contains" if contains_true else "does not contain"} the true proportion ({demo_p}).
            
            In repeated sampling, approximately {(1-demo_alpha)*100:.0f}% of such intervals would contain the true parameter.
            
            **Method Characteristics**:
            
            {
            "The Wald interval is simple and based on normal approximation. It can perform poorly for small samples or extreme proportions." if demo_method == "Wald" else
            "The Wilson score interval generally has better coverage properties than the Wald interval, especially for small samples or proportions near 0 or 1." if demo_method == "Wilson" else
            "The Clopper-Pearson interval guarantees at least the nominal coverage probability, though it's often conservative (wider than necessary)."
            }
            """)

    elif interval_type == "Difference of Means":
        st.markdown(r"""
        ### Derivation of Confidence Intervals for Difference of Means
        
        We'll derive confidence intervals for the difference of means $\mu_1 - \mu_2$ from two independent populations.
        
        ### 1. Known Variances Case
        
        Assume we have two independent random samples:
        - $X_1, X_2, \ldots, X_{n_1} \sim N(\mu_1, \sigma_1^2)$
        - $Y_1, Y_2, \ldots, Y_{n_2} \sim N(\mu_2, \sigma_2^2)$
        
        with known variances $\sigma_1^2$ and $\sigma_2^2$.
        
        #### Step 1: Define the estimator
        
        The natural estimator for $\mu_1 - \mu_2$ is $\bar{X} - \bar{Y}$, where:
        
        $$\bar{X} = \frac{1}{n_1}\sum_{i=1}^{n_1} X_i \quad \text{and} \quad \bar{Y} = \frac{1}{n_2}\sum_{i=1}^{n_2} Y_i$$
        
        #### Step 2: Find the distribution of the estimator
        
        Since $\bar{X} \sim N\left(\mu_1, \frac{\sigma_1^2}{n_1}\right)$ and $\bar{Y} \sim N\left(\mu_2, \frac{\sigma_2^2}{n_2}\right)$, and the samples are independent, we have:
        
        $$\bar{X} - \bar{Y} \sim N\left(\mu_1 - \mu_2, \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}\right)$$
        
        #### Step 3: Standardize to get a pivotal quantity
        
        The standardized statistic follows a standard normal distribution:
        
        $$Z = \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} \sim N(0, 1)$$
        
        #### Step 4: Construct the confidence interval
        
        For a $(1-\alpha)$ confidence level, we have:
        
        $$P\left(-z_{\alpha/2} \leq \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} \leq z_{\alpha/2}\right) = 1-\alpha$$
        
        Solving for $\mu_1 - \mu_2$:
        
        $$P\left((\bar{X} - \bar{Y}) - z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}} \leq \mu_1 - \mu_2 \leq (\bar{X} - \bar{Y}) + z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}\right) = 1-\alpha$$
        
        #### Step 5: Write the confidence interval
        
        A $(1-\alpha)$ confidence interval for $\mu_1 - \mu_2$ is:
        
        $$(\bar{X} - \bar{Y}) \pm z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}$$
        
        ### 2. Unknown But Equal Variances Case
        
        Now assume $\sigma_1^2 = \sigma_2^2 = \sigma^2$ (unknown).
        
        #### Step 1: Pool the samples to estimate the common variance
        
        The pooled estimator of $\sigma^2$ is:
        
        $$S_p^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}$$
        
        where $S_1^2$ and $S_2^2$ are the sample variances.
        
        #### Step 2: Find the distribution of the standardized statistic
        
        Under the equal variance assumption, we have:
        
        $$T = \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{S_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t_{n_1 + n_2 - 2}$$
        
        This follows a t-distribution with $n_1 + n_2 - 2$ degrees of freedom.
        
        #### Step 3: Construct the confidence interval
        
        For a $(1-\alpha)$ confidence level:
        
        $$P\left(-t_{\alpha/2, n_1 + n_2 - 2} \leq \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{S_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \leq t_{\alpha/2, n_1 + n_2 - 2}\right) = 1-\alpha$$
        
        #### Step 4: Write the confidence interval
        
        A $(1-\alpha)$ confidence interval for $\mu_1 - \mu_2$ is:
        
        $$(\bar{X} - \bar{Y}) \pm t_{\alpha/2, n_1 + n_2 - 2} \times S_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$
        
        ### 3. Unknown and Unequal Variances Case (Welch's t-test)
        
        When $\sigma_1^2 \neq \sigma_2^2$ and both are unknown, we need to use Welch's approximation.
        
        #### Step 1: Estimate each variance separately
        
        Use the sample variances $S_1^2$ and $S_2^2$ as estimators.
        
        #### Step 2: Approximate the distribution of the standardized statistic
        
        Welch showed that:
        
        $$T' = \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}$$
        
        approximately follows a t-distribution with degrees of freedom $\nu$ given by the Satterthwaite approximation:
        
        $$\nu \approx \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}$$
        
        #### Step 3: Construct the confidence interval
        
        For a $(1-\alpha)$ confidence level:
        
        $$P\left(-t_{\alpha/2, \nu} \leq \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \leq t_{\alpha/2, \nu}\right) = 1-\alpha$$
        
        #### Step 4: Write the confidence interval
        
        A $(1-\alpha)$ confidence interval for $\mu_1 - \mu_2$ is:
        
        $$(\bar{X} - \bar{Y}) \pm t_{\alpha/2, \nu} \times \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}$$
        
        ### Formal Derivation of Welch's Degrees of Freedom
        
        Let $V = \frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}$. We want to approximate the distribution of $\frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{V}}$.
        
        The key is to find the distribution of $V$. Since $\frac{(n_1-1)S_1^2}{\sigma_1^2} \sim \chi^2_{n_1-1}$ and $\frac{(n_2-1)S_2^2}{\sigma_2^2} \sim \chi^2_{n_2-1}$, we have:
        
        $$V = c_1 U_1 + c_2 U_2$$
        
        where $c_1 = \frac{\sigma_1^2}{n_1(n_1-1)}$, $c_2 = \frac{\sigma_2^2}{n_2(n_2-1)}$, $U_1 \sim \chi^2_{n_1-1}$, and $U_2 \sim \chi^2_{n_2-1}$.
        
        Welch approximated this as a scaled chi-square distribution:
        
        $$\frac{V}{E[V]} \approx \frac{\chi^2_\nu}{\nu}$$
        
        where $\nu$ is chosen to match the first two moments of $V$. This leads to:
        
        $$\nu = \frac{(E[V])^2}{Var(V)} = \frac{\left(\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}\right)^2}{\frac{\sigma_1^4}{n_1^2(n_1-1)} + \frac{\sigma_2^4}{n_2^2(n_2-1)}}$$
        
        Substituting the estimates $S_1^2$ and $S_2^2$ for $\sigma_1^2$ and $\sigma_2^2$, we get the Satterthwaite approximation.
        
        ### Practical Considerations and Recommendations
        
        #### When to Use Each Method:
        
        1. **Known Variances**: Rarely applicable in practice, but provides a theoretical foundation
        
        2. **Equal Variances (Pooled)**: 
        - Use when there's strong reason to believe variances are equal
        - More powerful when the assumption is correct
        - Standard in many experimental designs (ANOVA, etc.)
        
        3. **Welch's Method (Unequal Variances)**:
        - More robust to variance heterogeneity
        - Recommended as the default approach by many statisticians
        - Minimal loss of power even when variances are equal
        
        #### Testing the Equal Variance Assumption:
        
        While tests for equal variances exist (e.g., F-test, Levene's test), basing the choice of t-test on such preliminary tests can lead to increased Type I error rates. Modern practice generally recommends Welch's method as the default.
        
        #### Sample Size and Power Considerations:
        
        For a desired margin of error $E$ in the confidence interval for $\mu_1 - \mu_2$, the required sample sizes are:
        
        $$n_1 = n_2 = \frac{2(z_{\alpha/2})^2(\sigma_1^2 + \sigma_2^2)}{E^2}$$
        
        If the variances differ substantially, optimal allocation gives:
        
        $$\frac{n_1}{n_2} = \frac{\sigma_1}{\sigma_2}$$
        """, unsafe_allow_html=True)
        
        # Interactive demonstration
        st.subheader("Interactive Demonstration")
        
        col1, col2 = st.columns(2)
        with col1:
            demo_mu1 = st.slider("Group 1 mean (μ₁)", -10.0, 10.0, 0.0, 0.5)
            demo_sigma1 = st.slider("Group 1 std dev (σ₁)", 0.1, 5.0, 1.0, 0.1)
            demo_n1 = st.slider("Group 1 sample size (n₁)", 5, 100, 30, 5)
        with col2:
            demo_mu2 = st.slider("Group 2 mean (μ₂)", -10.0, 10.0, 0.0, 0.5)
            demo_sigma2 = st.slider("Group 2 std dev (σ₂)", 0.1, 5.0, 1.0, 0.1)
            demo_n2 = st.slider("Group 2 sample size (n₂)", 5, 100, 30, 5)
        
        demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
        demo_method = st.selectbox("Method", ["Pooled (equal variances)", "Welch (unequal variances)"])
        
        if st.button("Generate Samples and Confidence Interval", key="gen_diff_means"):
            # Generate samples
            np.random.seed(None)  # Use a different seed each time
            sample1 = np.random.normal(demo_mu1, demo_sigma1, demo_n1)
            sample2 = np.random.normal(demo_mu2, demo_sigma2, demo_n2)
            
            # Calculate sample statistics
            x_bar = np.mean(sample1)
            y_bar = np.mean(sample2)
            s1_squared = np.var(sample1, ddof=1)
            s2_squared = np.var(sample2, ddof=1)
            
            # Calculate CI based on selected method
            if demo_method == "Pooled (equal variances)":
                # Pooled variance estimate
                s_pooled_squared = ((demo_n1 - 1) * s1_squared + (demo_n2 - 1) * s2_squared) / (demo_n1 + demo_n2 - 2)
                s_pooled = np.sqrt(s_pooled_squared)
                
                # Standard error of the difference
                se_diff = s_pooled * np.sqrt(1/demo_n1 + 1/demo_n2)
                
                # Critical value
                t_crit = stats.t.ppf(1 - demo_alpha/2, demo_n1 + demo_n2 - 2)
                
                # Confidence interval
                margin = t_crit * se_diff
                ci_lower = (x_bar - y_bar) - margin
                ci_upper = (x_bar - y_bar) + margin
                
                method_name = "Pooled (Equal Variances)"
                df = demo_n1 + demo_n2 - 2
                
            else:  # "Welch (unequal variances)"
                # Standard error of the difference
                se_diff = np.sqrt(s1_squared/demo_n1 + s2_squared/demo_n2)
                
                # Welch-Satterthwaite degrees of freedom
                numerator = (s1_squared/demo_n1 + s2_squared/demo_n2)**2
                denominator = (s1_squared/demo_n1)**2/(demo_n1-1) + (s2_squared/demo_n2)**2/(demo_n2-1)
                df = numerator / denominator
                
                # Critical value
                t_crit = stats.t.ppf(1 - demo_alpha/2, df)
                
                # Confidence interval
                margin = t_crit * se_diff
                ci_lower = (x_bar - y_bar) - margin
                ci_upper = (x_bar - y_bar) + margin
                
                method_name = "Welch (Unequal Variances)"
            
            # Display results
            st.markdown(f"""
            **Sample Statistics**:
            - Group 1: Mean = {x_bar:.4f}, SD = {np.sqrt(s1_squared):.4f}, n = {demo_n1}
            - Group 2: Mean = {y_bar:.4f}, SD = {np.sqrt(s2_squared):.4f}, n = {demo_n2}
            - Observed difference (x̄ - ȳ): {x_bar - y_bar:.4f}
            
            **{method_name} {(1-demo_alpha)*100:.0f}% Confidence Interval**:
            - Lower bound: {ci_lower:.4f}
            - Upper bound: {ci_upper:.4f}
            - Margin of error: {margin:.4f}
            - Degrees of freedom: {df:.2f}
            - t-critical value: {t_crit:.4f}
            """)
            
            # Visualization
            fig = go.Figure()
            
            # Add vertical line for observed difference
            fig.add_vline(x=x_bar - y_bar, line=dict(color='blue', width=2),
                        annotation=dict(text=f"Observed difference: {x_bar - y_bar:.4f}", showarrow=False))
            
            # Add vertical line for true difference
            fig.add_vline(x=demo_mu1 - demo_mu2, line=dict(color='green', width=2, dash='dash'),
                        annotation=dict(text=f"True difference: {demo_mu1 - demo_mu2}", showarrow=False))
            
            # Add confidence interval
            fig.add_shape(
                type="rect",
                x0=ci_lower, x1=ci_upper,
                y0=0.2, y1=0.8,
                fillcolor="rgba(0, 100, 80, 0.2)",
                line=dict(color="rgba(0, 100, 80, 0.4)", width=2),
            )
            
            # Add annotation for confidence interval
            fig.add_annotation(
                x=(ci_lower + ci_upper)/2, y=0.5,
                text=f"{(1-demo_alpha)*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
                showarrow=False,
                font=dict(color="rgba(0, 100, 80, 1)")
            )
            
            # Update layout
            fig.update_layout(
                title=f"{method_name} Confidence Interval for Difference of Means",
                xaxis=dict(title="Difference (μ₁ - μ₂)"),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of results
            contains_true = ci_lower <= (demo_mu1 - demo_mu2) <= ci_upper
            variance_ratio = max(s1_squared, s2_squared) / min(s1_squared, s2_squared)
            
            st.markdown(f"""
            **Interpretation**:
            
            The {method_name} {(1-demo_alpha)*100:.0f}% confidence interval is [{ci_lower:.4f}, {ci_upper:.4f}].
            
            This interval {"contains" if contains_true else "does not contain"} the true difference ({demo_mu1 - demo_mu2}).
            
            In repeated sampling, approximately {(1-demo_alpha)*100:.0f}% of such intervals would contain the true parameter.
            
            **Method Assessment**:
            
            The observed variance ratio is {variance_ratio:.2f}. {
            "Since this ratio is close to 1, the equal variance assumption seems reasonable." if variance_ratio < 2 else
            "Since this ratio is moderately large, Welch's method may be more appropriate." if variance_ratio < 4 else
            "Since this ratio is large, Welch's method is strongly recommended."
            }
            
            {"Welch's method adjusts the degrees of freedom to account for unequal variances, making it more robust when group variances differ." if demo_method == "Welch (unequal variances)" else
            "The pooled method assumes equal variances and generally has more power when this assumption holds."}
            """)
            
            # Add sample histograms with normal overlays
            hist_fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Group 1 (n={demo_n1})", f"Group 2 (n={demo_n2})"])
            
            # Group 1 histogram
            hist_fig.add_trace(
                go.Histogram(x=sample1, name="Group 1", opacity=0.7, nbinsx=20),
                row=1, col=1
            )
            
            # Group 2 histogram
            hist_fig.add_trace(
                go.Histogram(x=sample2, name="Group 2", opacity=0.7, nbinsx=20),
                row=1, col=2
            )
            
            # Add normal overlay for Group 1
            x_range1 = np.linspace(min(sample1), max(sample1), 100)
            y_norm1 = stats.norm.pdf(x_range1, x_bar, np.sqrt(s1_squared))
            y_norm1_scaled = y_norm1 * (demo_n1 / 5)  # Scale for visualization
            
            hist_fig.add_trace(
                go.Scatter(x=x_range1, y=y_norm1_scaled, mode='lines', name="Normal Fit (G1)",
                        line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Add normal overlay for Group 2
            x_range2 = np.linspace(min(sample2), max(sample2), 100)
            y_norm2 = stats.norm.pdf(x_range2, y_bar, np.sqrt(s2_squared))
            y_norm2_scaled = y_norm2 * (demo_n2 / 5)  # Scale for visualization
            
            hist_fig.add_trace(
                go.Scatter(x=x_range2, y=y_norm2_scaled, mode='lines', name="Normal Fit (G2)",
                        line=dict(color='red', width=2)),
                row=1, col=2
            )
            
            hist_fig.update_layout(
                title="Sample Distributions",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(hist_fig, use_container_width=True)


    elif interval_type == "Variance":
        st.markdown(r"""
        ### Derivation of Confidence Intervals for Variance
        
        Here we derive confidence intervals for the variance $\sigma^2$ and standard deviation $\sigma$ of a normal population.
        
        ### 1. Confidence Interval for Variance
        
        Let $X_1, X_2, \ldots, X_n$ be a random sample from a normal distribution $N(\mu, \sigma^2)$ with unknown mean $\mu$ and unknown variance $\sigma^2$.
        
        #### Step 1: Identify a pivotal quantity
        
        For normally distributed data, the following statistic has a chi-square distribution:
        
        $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$
        
        where $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is the sample variance, and $\chi^2_{n-1}$ denotes the chi-square distribution with $n-1$ degrees of freedom.
        
        This is a pivotal quantity because its distribution does not depend on any unknown parameters.
        
        #### Step 2: Find critical values using the chi-square distribution
        
        For a confidence level of $1-\alpha$, we need to find values $a$ and $b$ such that:
        
        $P\left(a \leq \frac{(n-1)S^2}{\sigma^2} \leq b\right) = 1-\alpha$
        
        Using the quantiles of the chi-square distribution, we get:
        
        $P\left(\chi^2_{\alpha/2, n-1} \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, n-1}\right) = 1-\alpha$
        
        where $\chi^2_{p, n-1}$ is the $p$-quantile of the $\chi^2_{n-1}$ distribution.
        
        #### Step 3: Solve for $\sigma^2$
        
        Rearranging the inequalities to isolate $\sigma^2$:
        
        $P\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}} \leq \sigma^2 \leq \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right) = 1-\alpha$
        
        #### Step 4: Write the confidence interval
        
        A $(1-\alpha)$ confidence interval for $\sigma^2$ is:
        
        $\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}, \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right]$
        
        <div class="key-equation">
        $(1-\alpha)$ Confidence Interval for $\sigma^2$:
        $\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}, \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right]$
        </div>
        
        ### 2. Confidence Interval for Standard Deviation
        
        To obtain a confidence interval for the standard deviation $\sigma$, we simply take the square root of the endpoints of the interval for $\sigma^2$.
        
        #### Transformation Method
        
        If $[L, U]$ is a $(1-\alpha)$ confidence interval for $\sigma^2$, then $[\sqrt{L}, \sqrt{U}]$ is a $(1-\alpha)$ confidence interval for $\sigma$.
        
        This follows from the general result that if $g$ is a monotone function and $[L, U]$ is a confidence interval for a parameter $\theta$, then $[g(L), g(U)]$ is a confidence interval for $g(\theta)$.
        
        #### Resulting Confidence Interval
        
        A $(1-\alpha)$ confidence interval for $\sigma$ is:
        
        $\left[\sqrt{\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}}, \sqrt{\frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}}\right]$
        
        <div class="key-equation">
        $(1-\alpha)$ Confidence Interval for $\sigma$:
        $\left[\sqrt{\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}}, \sqrt{\frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}}\right]$
        </div>
        
        ### 3. Formal Proof of the Pivotal Quantity
        
        Here we provide a more rigorous proof of why the statistic $\frac{(n-1)S^2}{\sigma^2}$ follows a chi-square distribution with $n-1$ degrees of freedom.
        
        #### Theorem
        
        If $X_1, X_2, \ldots, X_n$ is a random sample from $N(\mu, \sigma^2)$, then:
        
        $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$
        
        #### Proof
        
        1. First, we standardize the random variables:
        $Z_i = \frac{X_i - \mu}{\sigma} \sim N(0, 1)$
        
        2. The sum of squares of standard normal random variables follows a chi-square distribution:
        $\sum_{i=1}^n Z_i^2 \sim \chi^2_n$
        
        3. We can decompose this sum of squares:
        $\sum_{i=1}^n Z_i^2 = \sum_{i=1}^n \left(\frac{X_i - \mu}{\sigma}\right)^2 = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu)^2$
        
        4. Further decomposition using the identity:
        $\sum_{i=1}^n (X_i - \mu)^2 = \sum_{i=1}^n (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2$
        
        5. We know that:
        $\frac{n(\bar{X} - \mu)^2}{\sigma^2} \sim \chi^2_1$
        
        6. By a theorem on the independence of the sample mean and sample variance in normal distributions:
        $\frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar{X})^2 \perp \frac{n(\bar{X} - \mu)^2}{\sigma^2}$
        
        7. For independent chi-square random variables, the sum is also chi-square with degrees of freedom equal to the sum of the individual degrees of freedom:
        $\frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu)^2 = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar{X})^2 + \frac{n(\bar{X} - \mu)^2}{\sigma^2} \sim \chi^2_{n-1} + \chi^2_1 = \chi^2_n$
        
        8. By rearranging, we get:
        $\frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar{X})^2 = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu)^2 - \frac{n(\bar{X} - \mu)^2}{\sigma^2} \sim \chi^2_n - \chi^2_1 = \chi^2_{n-1}$
        
        9. Since $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$, we have:
        $\frac{(n-1)S^2}{\sigma^2} = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar{X})^2 \sim \chi^2_{n-1}$
        
        ### 4. Robustness and Practical Considerations
        
        #### Non-normality Effects
        
        The intervals derived above rely critically on the assumption that the data come from a normal distribution. When this assumption is violated:
        
        1. The statistic $\frac{(n-1)S^2}{\sigma^2}$ may no longer follow a $\chi^2_{n-1}$ distribution
        2. The actual coverage probability may differ from the nominal level $1-\alpha$
        
        Specific effects of different types of non-normality:
        
        - **Skewness**: Tends to make the intervals too narrow, resulting in undercoverage
        - **Heavy tails**: Can lead to highly variable sample variances and unreliable intervals
        - **Mixture distributions**: May result in bimodal distributions of the sample variance
        
        #### Alternative Approaches
        
        For non-normal data, consider:
        
        1. **Bootstrapping**: Resampling methods to estimate the distribution of $S^2$ empirically
        2. **Transformation**: Apply variance-stabilizing transformations before analysis
        3. **Robust estimators**: Use estimators less sensitive to outliers (e.g., median absolute deviation)
        4. **Modified chi-square approximations**: Adjust degrees of freedom based on estimated kurtosis
        
        #### Sample Size Considerations
        
        The chi-square-based intervals have the following properties:
        
        - **Exact** for normal data, regardless of sample size
        - **Asymmetric**, with the upper bound further from the point estimate than the lower bound
        - **Narrower** for larger sample sizes
        
        For small samples ($n < 30$), the intervals can be quite wide, reflecting the high uncertainty in variance estimation.
        
        ### 5. Applications in Industry and Research
        
        #### Quality Control
        
        In manufacturing, variance confidence intervals are used to:
        
        - Assess process capability and stability
        - Set control limits for monitoring process variation
        - Compare variability between different production methods or batches
        
        #### Metrology and Measurement
        
        In calibration and measurement science:
        
        - Quantify instrument precision and repeatability
        - Report measurement uncertainty in accordance with standards
        - Validate measurement systems
        
        #### Risk Assessment and Reliability Engineering
        
        In reliability analysis:
        
        - Quantify uncertainty in failure rates and lifetime variations
        - Assess confidence in safety margins
        - Support probabilistic risk assessments
        """, unsafe_allow_html=True)
        
        # Interactive demonstration
        st.subheader("Interactive Demonstration")
        
        col1, col2 = st.columns(2)
        with col1:
            demo_mu = st.slider("True population mean (μ)", -10.0, 10.0, 0.0, 0.5)
            demo_sigma = st.slider("True population std dev (σ)", 0.1, 5.0, 1.0, 0.1)
        with col2:
            demo_n = st.slider("Sample size (n)", 5, 100, 20, 5)
            demo_alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
        
        demo_parameter = st.radio("Parameter of interest", ["Variance (σ²)", "Standard Deviation (σ)"])
        
        if st.button("Generate Sample and Confidence Interval", key="gen_var_ci"):
            # Generate sample
            np.random.seed(None)  # Use a different seed each time
            sample = np.random.normal(demo_mu, demo_sigma, demo_n)
            sample_var = np.var(sample, ddof=1)
            sample_std = np.sqrt(sample_var)
            
            # Calculate chi-square critical values
            chi2_lower = stats.chi2.ppf(demo_alpha/2, demo_n-1)
            chi2_upper = stats.chi2.ppf(1-demo_alpha/2, demo_n-1)
            
            # Calculate confidence interval for variance
            var_lower = (demo_n-1) * sample_var / chi2_upper
            var_upper = (demo_n-1) * sample_var / chi2_lower
            
            # Calculate confidence interval for standard deviation
            std_lower = np.sqrt(var_lower)
            std_upper = np.sqrt(var_upper)
            
            # Display results
            st.markdown(f"""
            **Sample Statistics**:
            - Sample size: {demo_n}
            - Sample mean: {np.mean(sample):.4f}
            - Sample variance: {sample_var:.4f}
            - Sample standard deviation: {sample_std:.4f}
            
            **{(1-demo_alpha)*100:.0f}% Confidence Interval for Variance**:
            - Lower bound: {var_lower:.4f}
            - Upper bound: {var_upper:.4f}
            - Interval width: {var_upper - var_lower:.4f}
            
            **{(1-demo_alpha)*100:.0f}% Confidence Interval for Standard Deviation**:
            - Lower bound: {std_lower:.4f}
            - Upper bound: {std_upper:.4f}
            - Interval width: {std_upper - std_lower:.4f}
            """)
            
            # Visualization
            if demo_parameter == "Variance (σ²)":
                param_value = demo_sigma**2
                lower_bound = var_lower
                upper_bound = var_upper
                sample_stat = sample_var
                x_label = "Variance (σ²)"
                true_label = f"True Variance: {param_value:.4f}"
                sample_label = f"Sample Variance: {sample_stat:.4f}"
                ci_label = f"{(1-demo_alpha)*100:.0f}% CI: [{var_lower:.4f}, {var_upper:.4f}]"
            else:
                param_value = demo_sigma
                lower_bound = std_lower
                upper_bound = std_upper
                sample_stat = sample_std
                x_label = "Standard Deviation (σ)"
                true_label = f"True Std Dev: {param_value:.4f}"
                sample_label = f"Sample Std Dev: {sample_stat:.4f}"
                ci_label = f"{(1-demo_alpha)*100:.0f}% CI: [{std_lower:.4f}, {std_upper:.4f}]"
            
            fig = go.Figure()
            
            # Add vertical line for sample statistic
            fig.add_vline(x=sample_stat, line=dict(color='blue', width=2),
                        annotation=dict(text=sample_label, showarrow=False))
            
            # Add vertical line for true parameter
            fig.add_vline(x=param_value, line=dict(color='green', width=2, dash='dash'),
                        annotation=dict(text=true_label, showarrow=False))
            
            # Add confidence interval
            fig.add_shape(
                type="rect",
                x0=lower_bound, x1=upper_bound,
                y0=0.2, y1=0.8,
                fillcolor="rgba(0, 100, 80, 0.2)",
                line=dict(color="rgba(0, 100, 80, 0.4)", width=2),
            )
            
            # Add annotation for confidence interval
            fig.add_annotation(
                x=(lower_bound + upper_bound)/2, y=0.5,
                text=ci_label,
                showarrow=False,
                font=dict(color="rgba(0, 100, 80, 1)")
            )
            
            # Update layout
            fig.update_layout(
                title=f"Confidence Interval for {x_label}",
                xaxis=dict(title=x_label),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add sample histogram
            hist_fig = go.Figure()
            
            hist_fig.add_trace(go.Histogram(
                x=sample,
                nbinsx=min(20, demo_n),
                opacity=0.7,
                name="Sample Data"
            ))
            
            # Add normal density overlay
            x_range = np.linspace(min(sample), max(sample), 100)
            y_norm = stats.norm.pdf(x_range, np.mean(sample), sample_std)
            y_norm_scaled = y_norm * (demo_n / 5)  # Scale for visualization
            
            hist_fig.add_trace(go.Scatter(
                x=x_range, y=y_norm_scaled,
                mode='lines',
                name="Normal Fit",
                line=dict(color='red', width=2)
            ))
            
            hist_fig.update_layout(
                title="Sample Distribution",
                xaxis_title="Value",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(hist_fig, use_container_width=True)
            
            # Add explanation of results
            contains_var = var_lower <= demo_sigma**2 <= var_upper
            contains_std = std_lower <= demo_sigma <= std_upper
            
            if demo_parameter == "Variance (σ²)":
                contains = contains_var
            else:
                contains = contains_std
            
            st.markdown(f"""
            **Interpretation**:
            
            The {(1-demo_alpha)*100:.0f}% confidence interval for {demo_parameter.lower()} is {
            f"[{var_lower:.4f}, {var_upper:.4f}]" if demo_parameter == "Variance (σ²)" else 
            f"[{std_lower:.4f}, {std_upper:.4f}]"
            }.
            
            This interval {"contains" if contains else "does not contain"} the true parameter value ({
            f"{demo_sigma**2:.4f}" if demo_parameter == "Variance (σ²)" else 
            f"{demo_sigma:.4f}"
            }).
            
            **Key Observations**:
            
            1. The confidence interval is **asymmetric** around the sample estimate, reflecting the skewed nature of the chi-square distribution.
            
            2. The interval width depends strongly on sample size. With {demo_n} observations, the ratio of upper to lower bound is {
            f"{var_upper/var_lower:.2f}" if demo_parameter == "Variance (σ²)" else 
            f"{std_upper/std_lower:.2f}"
            }.
            
            3. For variance estimation, larger samples provide much narrower intervals relative to the point estimate.
            
            4. The derived interval is exact for normally distributed data, regardless of sample size (unlike many other intervals that rely on asymptotic properties).
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

# Add footer
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)