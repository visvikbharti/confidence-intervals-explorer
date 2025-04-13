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

st.set_page_config(
    page_title="Real-world Applications - CI Explorer",
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
        days_for_mde = (2 * z_crit / abs_diff * 100)**2 * ((observed_control/100) * (1 - observed_control/100) + (observed_variant/100) * (1 - observed_variant/100)) / daily_visitors if abs_diff != 0 else float('inf')
        
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
        
        samples_needed = int(np.ceil((2 * (t_crit)/ effect_size)**2)) if effect_size > 0 else float('inf')
        
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