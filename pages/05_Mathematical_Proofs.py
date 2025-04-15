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
    page_title="Mathematical Proofs - CI Explorer",
    page_icon="📊",
    layout="wide",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)
add_latex_styles()
setup_math_rendering()
# Custom CSS
# Apply the custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

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
        
        # Add vertical lines for sample mean and CI (FIXED METHOD)
        fig.add_vline(x=sample_mean, line=dict(color='red', width=2))
        fig.add_annotation(text=f"Sample Mean", x=sample_mean, y=max(sampling_dist)*1.05, 
                          xref="x", yref="y", showarrow=False)
        
        fig.add_vline(x=ci_lower, line=dict(color='green', width=2, dash='dash'))
        fig.add_annotation(text=f"Lower CI", x=ci_lower, y=max(sampling_dist)*0.9, 
                          xref="x", yref="y", showarrow=False)
        
        fig.add_vline(x=ci_upper, line=dict(color='green', width=2, dash='dash'))
        fig.add_annotation(text=f"Upper CI", x=ci_upper, y=max(sampling_dist)*0.9, 
                          xref="x", yref="y", showarrow=False)
        
        # Add vertical line for true mean
        fig.add_vline(x=demo_mean, line=dict(color='purple', width=2, dash='dot'))
        fig.add_annotation(text=f"True Mean", x=demo_mean, y=max(sampling_dist)*0.8, 
                          xref="x", yref="y", showarrow=False)
        
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

    # Visualization comparing confidence and credible intervals
    st.subheader("Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prior_strength = st.slider("Prior strength", 0.1, 10.0, 1.0, 0.1, 
                                  help="Higher values mean stronger prior influence")
        prior_center = st.slider("Prior center", -5.0, 5.0, 0.0, 0.5,
                                help="Where the prior is centered")
    
    with col2:
        sample_size = st.slider("Sample size", 5, 50, 10,
                              help="Number of observations")
        true_mean = st.slider("True mean", -5.0, 5.0, 2.0, 0.5,
                            help="Actual parameter value (typically unknown)")
    
    if st.button("Generate Comparison"):
        # Generate data
        np.random.seed(None)
        sigma = 2.0  # Fixed standard deviation
        data = np.random.normal(true_mean, sigma, sample_size)
        x_bar = np.mean(data)
        
        # Frequentist confidence interval
        z_critical = stats.norm.ppf(0.975)  # For 95% CI
        freq_margin = z_critical * sigma / np.sqrt(sample_size)
        freq_lower = x_bar - freq_margin
        freq_upper = x_bar + freq_margin
        
        # Bayesian analysis with normal prior
        prior_variance = (sigma / prior_strength)**2
        likelihood_variance = sigma**2 / sample_size
        
        # Posterior parameters
        posterior_precision = 1/prior_variance + 1/likelihood_variance
        posterior_variance = 1/posterior_precision
        posterior_mean = posterior_variance * (prior_center/prior_variance + x_bar/likelihood_variance)
        
        # Bayesian credible interval
        bayes_margin = z_critical * np.sqrt(posterior_variance)
        bayes_lower = posterior_mean - bayes_margin
        bayes_upper = posterior_mean + bayes_margin
        
        # Create plot for comparison
        x_range = np.linspace(
            min(freq_lower, bayes_lower, prior_center) - 3, 
            max(freq_upper, bayes_upper, prior_center) + 3, 
            1000
        )
        
        # Compute densities
        likelihood = stats.norm.pdf(x_range, x_bar, np.sqrt(likelihood_variance))
        prior = stats.norm.pdf(x_range, prior_center, np.sqrt(prior_variance))
        posterior = stats.norm.pdf(x_range, posterior_mean, np.sqrt(posterior_variance))
        
        # Scale for better visualization
        max_density = max(max(likelihood), max(posterior), max(prior))
        likelihood_scaled = likelihood / max_density
        posterior_scaled = posterior / max_density
        prior_scaled = prior / max_density * 0.5  # Make prior smaller
        
        # Create figure
        fig = go.Figure()
        
        # Add densities
        fig.add_trace(go.Scatter(
            x=x_range, y=likelihood_scaled,
            mode='lines', name='Likelihood',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range, y=prior_scaled,
            mode='lines', name='Prior',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range, y=posterior_scaled,
            mode='lines', name='Posterior',
            line=dict(color='red', width=2)
        ))
        
        # Add vertical lines for true mean
        fig.add_vline(x=true_mean, line=dict(color='black', width=2, dash='dash'))
        fig.add_annotation(text="True Mean", x=true_mean, y=1.05, 
                          xref="x", yref="paper", showarrow=False)
        
        # Add vertical lines for sample mean
        fig.add_vline(x=x_bar, line=dict(color='blue', width=2))
        fig.add_annotation(text="Sample Mean", x=x_bar, y=0.95, 
                          xref="x", yref="paper", showarrow=False)
        
        # Add vertical lines for posterior mean
        fig.add_vline(x=posterior_mean, line=dict(color='red', width=2))
        fig.add_annotation(text="Posterior Mean", x=posterior_mean, y=0.85, 
                          xref="x", yref="paper", showarrow=False)
        
        fig.update_layout(
            title="Comparison of Frequentist and Bayesian Approaches",
            xaxis_title="Parameter Value",
            yaxis_title="Scaled Density",
            height=500,
            yaxis=dict(showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create interval comparison
        interval_fig = go.Figure()
        
        # Add frequentist interval
        interval_fig.add_trace(go.Scatter(
            x=[freq_lower, freq_upper], y=[1, 1],
            mode='lines', name='95% Confidence Interval',
            line=dict(color='blue', width=4)
        ))
        
        # Add Bayesian interval
        interval_fig.add_trace(go.Scatter(
            x=[bayes_lower, bayes_upper], y=[2, 2],
            mode='lines', name='95% Credible Interval',
            line=dict(color='red', width=4)
        ))
        
        # Add vertical line for true value
        interval_fig.add_vline(x=true_mean, line=dict(color='black', width=2, dash='dash'))
        interval_fig.add_annotation(text="True Mean", x=true_mean, y=3, 
                                  xref="x", yref="y", showarrow=False)
        
        interval_fig.update_layout(
            title="Interval Comparison",
            xaxis_title="Parameter Value",
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2],
                ticktext=['Frequentist', 'Bayesian'],
                showgrid=False
            ),
            height=300
        )
        
        st.plotly_chart(interval_fig, use_container_width=True)
        
        # Display results
        st.subheader("Numerical Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sample Mean", f"{x_bar:.4f}")
            st.metric("Posterior Mean", f"{posterior_mean:.4f}")
            st.metric("True Mean", f"{true_mean:.4f}")
        
        with col2:
            st.metric("95% Confidence Interval", f"[{freq_lower:.4f}, {freq_upper:.4f}]")
            st.metric("95% Credible Interval", f"[{bayes_lower:.4f}, {bayes_upper:.4f}]")
            
        # Interpretation
        st.subheader("Interpretation")
        
        freq_contains = freq_lower <= true_mean <= freq_upper
        bayes_contains = bayes_lower <= true_mean <= bayes_upper
        
        st.markdown(f"""
        **Frequentist Interpretation**: 
        
        The 95% confidence interval is [{freq_lower:.4f}, {freq_upper:.4f}]. This means that if we repeated the experiment many times, about 95% of such intervals would contain the true mean. This particular interval {"does" if freq_contains else "does not"} contain the true mean ({true_mean}).
        
        **Bayesian Interpretation**: 
        
        The 95% credible interval is [{bayes_lower:.4f}, {bayes_upper:.4f}]. Given the observed data and our prior beliefs, there is a 95% probability that the true mean lies within this interval. This particular interval {"does" if bayes_contains else "does not"} contain the true mean ({true_mean}).
        
        **Prior Influence**:
        
        The prior had {'strong' if prior_strength > 3 else 'moderate' if prior_strength > 1 else 'weak'} influence on the posterior. As the sample size increases, the influence of the prior diminishes, and the Bayesian credible interval approaches the frequentist confidence interval.
        """)

# At the bottom of your app.py file, add this code:

# Add footer
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)