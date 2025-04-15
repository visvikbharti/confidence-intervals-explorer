import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from custom_styling import get_custom_css, get_footer
from latex_helper import render_latex, render_definition, render_example, render_proof, render_key_equation

st.set_page_config(
    page_title="LaTeX Display Guide - CI Explorer",
    page_icon="📊",
    layout="wide",
)

# Apply the custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("LaTeX Formula Display Guide")
st.markdown("*This guide demonstrates how to properly render LaTeX formulas in both light and dark modes*")

st.markdown("""
## Problem and Solution

Mathematical formulas rendered with LaTeX can be difficult to read in different display modes (light vs. dark).
This guide provides solutions for displaying formulas clearly in both modes using our custom styling.

### How to Use

Import the helper functions at the top of your module:

```python
from latex_helper import render_latex, render_definition, render_example, render_proof, render_key_equation
```

Then use these functions instead of direct Markdown for mathematical content.
""")

st.subheader("Examples")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Standard Markdown (Less Reliable)")
    st.markdown(r"Inline formula: $\bar{X} \sim N(\mu, \sigma^2/n)$")
    st.markdown(r"""
    Display formula: 
    $$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$
    """)
    
    st.markdown("#### Definition with LaTeX")
    st.markdown(r"""
    <div class="definition">
    <strong>Definition 1:</strong> A confidence interval for a parameter $\theta$ is a pair of statistics $L(X)$ and $U(X)$ such that:
    
    $$P_{\theta}(L(X) \leq \theta \leq U(X)) \geq 1-\alpha \quad \forall \theta \in \Theta$$
    
    where $1-\alpha$ is the confidence level, and $\Theta$ is the parameter space.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Improved Rendering (More Reliable)")
    render_latex(r"\bar{X} \sim N(\mu, \sigma^2/n)")
    render_latex(r"\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}", block=True)
    
    st.markdown("#### Definition with Improved Rendering")
    render_definition(r"""
    <strong>Definition 1:</strong> A confidence interval for a parameter $\theta$ is a pair of statistics $L(X)$ and $U(X)$ such that:
    
    $$P_{\theta}(L(X) \leq \theta \leq U(X)) \geq 1-\alpha \quad \forall \theta \in \Theta$$
    
    where $1-\alpha$ is the confidence level, and $\Theta$ is the parameter space.
    """)

st.subheader("Other Styled Components")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Example")
    render_example(r"""
    <strong>Example:</strong> For a normal sample with unknown mean $\mu$ and known variance $\sigma^2$, the quantity
    $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$$
    follows a standard normal distribution regardless of the value of $\mu$, making it a pivotal quantity.
    """)

with col2:
    st.markdown("#### Proof")
    render_proof(r"""
    <strong>Step 1:</strong> Identify a pivotal quantity.
    
    The standardized sample mean follows a standard normal distribution:
    
    $$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$
    
    <strong>Step 2:</strong> Find critical values such that $P(-z_{\alpha/2} \leq Z \leq z_{\alpha/2}) = 1-\alpha$.
    """)

st.markdown("#### Key Equation")
render_key_equation(r"""
<strong>Confidence Interval for Variance:</strong>

$$\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}, \frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}\right]$$

This interval provides a $(1-\alpha)$ confidence level for estimating the population variance $\sigma^2$.
""")

st.subheader("Mathematical Content Stress Test")

st.markdown("The following section includes various complex mathematical expressions to test rendering:")

render_latex(r"\int_{-\infty}^{\infty} e^{-x^2/2} \, dx = \sqrt{2\pi}", block=True)

render_latex(r"\sum_{i=1}^{n} \frac{(X_i - \bar{X})^2}{n-1} = S^2", block=True)

render_latex(r"\binom{n}{k} = \frac{n!}{k!(n-k)!}", block=True)

render_latex(r"f(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}", block=True)

render_key_equation(r"""
<strong>Wilson Score Interval:</strong>

$$\frac{\hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}}}{1 + \frac{z_{\alpha/2}^2}{n}}$$
""")

# Add theme toggle information
st.info("""
**Try toggling between light and dark mode** to see how these formulas appear in both modes.
In Streamlit, you can change the theme using the menu in the top-right corner (≡ → Settings → Theme).
""")

# Add footer
st.markdown("---")
st.markdown(get_footer(), unsafe_allow_html=True)