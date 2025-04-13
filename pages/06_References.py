import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="References & Resources - CI Explorer",
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
    .ref-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #4CAF50;
    }
    .ref-title {
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 5px;
    }
    .ref-authors {
        font-style: italic;
        color: #34495E;
        margin-bottom: 5px;
    }
    .ref-journal {
        color: #7F8C8D;
    }
    .ref-link {
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.header("References & Resources")

tabs = st.tabs(["Textbooks", "Research Papers", "Online Resources", "Software Tools", "About This App"])

with tabs[0]:  # Textbooks
    st.subheader("Textbooks on Confidence Intervals and Statistical Inference")
    
    textbooks = [
        {
            "title": "Statistical Inference",
            "authors": "Casella, G., & Berger, R. L.",
            "year": 2002,
            "publisher": "Duxbury Press",
            "edition": "2nd",
            "description": "A comprehensive introduction to statistical inference that covers confidence intervals in detail. The book presents both theoretical foundations and practical applications."
        },
        {
            "title": "All of Statistics: A Concise Course in Statistical Inference",
            "authors": "Wasserman, L.",
            "year": 2004,
            "publisher": "Springer",
            "description": "A modern approach to statistics that covers confidence intervals, hypothesis testing, and Bayesian methods. Particularly useful for understanding the theoretical underpinnings of confidence intervals."
        },
        {
            "title": "Probability and Statistics",
            "authors": "DeGroot, M. H., & Schervish, M. J.",
            "year": 2012,
            "publisher": "Pearson",
            "edition": "4th",
            "description": "An excellent introduction to probability and statistics with clear explanations of confidence intervals and their interpretations."
        },
        {
            "title": "An Introduction to the Bootstrap",
            "authors": "Efron, B., & Tibshirani, R. J.",
            "year": 1994,
            "publisher": "Chapman & Hall/CRC",
            "description": "The definitive reference on bootstrap methods, including the construction of bootstrap confidence intervals."
        },
        {
            "title": "Statistical Methods in Medical Research",
            "authors": "Armitage, P., Berry, G., & Matthews, J. N. S.",
            "year": 2002,
            "publisher": "Wiley-Blackwell",
            "edition": "4th",
            "description": "A comprehensive resource for statistical methods in medical research, with detailed coverage of confidence intervals for various study designs."
        },
        {
            "title": "Modern Applied Statistics with S",
            "authors": "Venables, W. N., & Ripley, B. D.",
            "year": 2002,
            "publisher": "Springer",
            "edition": "4th",
            "description": "A practical guide to modern statistical methods with R/S-PLUS, including implementation of various confidence interval techniques."
        },
        {
            "title": "Bayesian Data Analysis",
            "authors": "Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B.",
            "year": 2013,
            "publisher": "Chapman & Hall/CRC",
            "edition": "3rd",
            "description": "The authoritative text on Bayesian methods, including credible intervals and their relationship to confidence intervals."
        }
    ]
    
    for book in textbooks:
        with st.expander(f"{book['title']} ({book['year']})"):
            st.write(f"**Authors**: {book['authors']}")
            st.write(f"**Publisher**: {book['publisher']}")
            if 'edition' in book:
                st.write(f"**Edition**: {book['edition']}")
            st.write(f"**Description**: {book['description']}")

with tabs[1]:  # Research Papers
    st.subheader("Key Research Papers on Confidence Intervals")
    
    papers = [
        {
            "title": "Probable inference, the law of succession, and statistical inference",
            "authors": "Wilson, E. B.",
            "year": 1927,
            "journal": "Journal of the American Statistical Association, 22(158), 209-212",
            "description": "Introduced the Wilson score interval for binomial proportions, which has better coverage properties than the standard Wald interval.",
            "doi": "10.1080/01621459.1927.10502953"
        },
        {
            "title": "The use of confidence or fiducial limits illustrated in the case of the binomial",
            "authors": "Clopper, C. J., & Pearson, E. S.",
            "year": 1934,
            "journal": "Biometrika, 26(4), 404-413",
            "description": "Presented the Clopper-Pearson exact interval for binomial proportions, which guarantees the nominal coverage probability.",
            "doi": "10.1093/biomet/26.4.404"
        },
        {
            "title": "Outline of a theory of statistical estimation based on the classical theory of probability",
            "authors": "Neyman, J.",
            "year": 1937,
            "journal": "Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences, 236(767), 333-380",
            "description": "A foundational paper that formalized the concept of confidence intervals in the frequentist framework.",
            "doi": "10.1098/rsta.1937.0005"
        },
        {
            "title": "The generalization of 'Student's' problem when several different population variances are involved",
            "authors": "Welch, B. L.",
            "year": 1947,
            "journal": "Biometrika, 34(1/2), 28-35",
            "description": "Introduced what is now known as the Welch's t-test and confidence interval for the difference between means of two populations with unequal variances.",
            "doi": "10.1093/biomet/34.1-2.28"
        },
        {
            "title": "Approximate is better than 'exact' for interval estimation of binomial proportions",
            "authors": "Agresti, A., & Coull, B. A.",
            "year": 1998,
            "journal": "The American Statistician, 52(2), 119-126",
            "description": "Advocated for the adjusted Wald interval (adding 2 successes and 2 failures) for binomial proportions, showing it has better coverage properties than exact methods for small samples.",
            "doi": "10.1080/00031305.1998.10480550"
        },
        {
            "title": "Interval estimation for a binomial proportion",
            "authors": "Brown, L. D., Cai, T. T., & DasGupta, A.",
            "year": 2001,
            "journal": "Statistical Science, 16(2), 101-133",
            "description": "A comprehensive review and comparison of different confidence interval methods for binomial proportions, with recommendations for practice.",
            "doi": "10.1214/ss/1009213286"
        },
        {
            "title": "Better bootstrap confidence intervals",
            "authors": "DiCiccio, T. J., & Efron, B.",
            "year": 1996,
            "journal": "Journal of the American Statistical Association, 91(434), 1428-1432",
            "description": "Introduced improved bootstrap confidence interval methods, including the BCa (bias-corrected and accelerated) method.",
            "doi": "10.1080/01621459.1996.10476725"
        },
        {
            "title": "On the relation between confidence intervals and tests of significance",
            "authors": "Cox, D. R.",
            "year": 1977,
            "journal": "Biometrika, 64(2), 404-406",
            "description": "Explained the duality between confidence intervals and hypothesis tests, showing how they provide complementary information.",
            "doi": "10.1093/biomet/64.2.404"
        }
    ]
    
    for paper in papers:
        with st.expander(f"{paper['title']} ({paper['year']})"):
            st.write(f"**Authors**: {paper['authors']}")
            st.write(f"**Journal**: {paper['journal']}")
            st.write(f"**Description**: {paper['description']}")
            if 'doi' in paper:
                st.write(f"**DOI**: [{paper['doi']}](https://doi.org/{paper['doi']})")

with tabs[2]:  # Online Resources
    st.subheader("Online Resources for Learning About Confidence Intervals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courses and Tutorials")
        
        st.markdown("""
        - [**Statistical Inference Course**](https://www.coursera.org/learn/statistical-inference) - Johns Hopkins University on Coursera
        - [**Confidence Intervals Tutorial**](http://www.stat.yale.edu/Courses/1997-98/101/confint.htm) - Yale University
        - [**Understanding Confidence Intervals**](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-the-differences-between-one-tailed-and-two-tailed-tests/) - UCLA Institute for Digital Research and Education
        - [**Khan Academy - Confidence Intervals**](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample) - Excellent introductory tutorials
        - [**Seeing Theory**](https://seeing-theory.brown.edu/frequentist-inference/index.html) - Brown University's visual introduction to probability and statistics
        """)
        
        st.subheader("Interactive Tools")
        
        st.markdown("""
        - [**StatKey**](http://www.lock5stat.com/StatKey/) - Online tools for statistical inference including bootstrap confidence intervals
        - [**R-fiddle**](http://www.r-fiddle.org/) - Online R environment for implementing confidence interval calculations
        - [**WISE Project**](https://wise.cgu.edu/) - Web Interface for Statistics Education with interactive tutorials on confidence intervals
        - [**Rossman/Chance Applet Collection**](http://www.rossmanchance.com/applets/) - Interactive applets for exploring statistical concepts
        """)
    
    with col2:
        st.subheader("Reference Materials")
        
        st.markdown("""
        - [**NIST/SEMATECH e-Handbook of Statistical Methods**](https://www.itl.nist.gov/div898/handbook/) - Comprehensive resource on statistical methods
        - [**Statistical Engineering Division at NIST**](https://www.nist.gov/itl/sed) - Applied statistics resources
        - [**ASA Guide on Confidence Intervals**](https://www.amstat.org/asa/files/pdfs/POL-Confidence-Intervals.pdf) - American Statistical Association's guide
        - [**OpenIntro Statistics**](https://www.openintro.org/book/os/) - Free statistics textbook with confidence interval coverage
        """)
        
        st.subheader("Video Resources")
        
        st.markdown("""
        - [**StatQuest with Josh Starmer**](https://www.youtube.com/c/joshstarmer) - Excellent YouTube channel with clear statistical explanations
        - [**Brandon Foltz Statistics**](https://www.youtube.com/user/BCFoltz) - Detailed tutorials on confidence intervals and related topics
        - [**3Blue1Brown**](https://www.youtube.com/c/3blue1brown) - Visual explanations of mathematical concepts underlying statistics
        """)

with tabs[3]:  # Software Tools
    st.subheader("Software Tools for Confidence Interval Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("R Packages")
        
        r_packages = [
            {
                "name": "stats",
                "description": "Base R package with t.test(), binom.test() and other functions for confidence intervals",
                "example": "t.test(x, conf.level = 0.95)$conf.int  # t confidence interval for mean"
            },
            {
                "name": "boot",
                "description": "Functions for bootstrap confidence intervals, including percentile, BCa, and t-bootstrap methods",
                "example": "boot.ci(boot.out, type = c(\"norm\", \"basic\", \"perc\", \"bca\"))"
            },
            {
                "name": "binom",
                "description": "Various confidence interval methods for binomial proportions",
                "example": "binom.confint(x, n, methods = \"all\")  # Compare all methods"
            },
            {
                "name": "PropCIs",
                "description": "Confidence intervals for proportions and related quantities",
                "example": "scoreci(x, n, conf.level = 0.95)  # Wilson score interval"
            },
            {
                "name": "MASS",
                "description": "Confidence and prediction intervals for various regression models",
                "example": "glm.ci(model, level = 0.95)  # CIs for GLM parameters"
            }
        ]
        
        for package in r_packages:
            with st.expander(f"{package['name']}"):
                st.write(f"**Description**: {package['description']}")
                st.code(package['example'], language="r")
    
    with col2:
        st.subheader("Python Libraries")
        
        python_libraries = [
            {
                "name": "scipy.stats",
                "description": "Statistical functions including confidence intervals for various distributions",
                "example": "from scipy import stats\nstats.norm.interval(0.95, loc=mean, scale=se)  # Normal CI"
            },
            {
                "name": "statsmodels",
                "description": "Comprehensive statistics package with advanced confidence interval methods",
                "example": "import statsmodels.api as sm\nresult = sm.OLS(y, X).fit()\nresult.conf_int(alpha=0.05)  # regression CIs"
            },
            {
                "name": "bootstrapped",
                "description": "Simple interface for bootstrap confidence intervals",
                "example": "from bootstrapped.bootstrap import bootstrap\nbootstrap(data, stat_func=np.mean).value"
            },
            {
                "name": "scikit-learn",
                "description": "Machine learning library with some confidence interval capabilities for predictions",
                "example": "from sklearn.ensemble import RandomForestRegressor\nrfr.predict(X_test, return_std=True)  # for predictions + std"
            },
            {
                "name": "pymc",
                "description": "Bayesian modeling and credible intervals",
                "example": "import pymc as pm\nwith model:\n    trace = pm.sample()\n    pm.hdi(trace, hdi_prob=0.95)  # 95% credible interval"
            }
        ]
        
        for library in python_libraries:
            with st.expander(f"{library['name']}"):
                st.write(f"**Description**: {library['description']}")
                st.code(library['example'], language="python")
        
        st.subheader("Specialized Statistical Software")
        
        specialized_tools = [
            "**SAS** - Comprehensive statistical software with extensive confidence interval procedures",
            "**SPSS** - Statistical software with user-friendly confidence interval calculations",
            "**Minitab** - Statistical software popular in industrial and quality control applications",
            "**Stata** - Data analysis software with various confidence interval implementations",
            "**JASP** - Free, open-source alternative to SPSS with Bayesian capabilities"
        ]
        
        for tool in specialized_tools:
            st.markdown(f"- {tool}")

with tabs[4]:  # About This App
    st.subheader("About This Confidence Intervals Explorer")
    
    st.markdown("""
    ### Purpose
    
    This application was created as a comprehensive educational tool to help understand confidence intervals and their applications. It includes:
    
    - **Theoretical foundations**: Mathematical derivations and properties of confidence intervals
    - **Interactive simulations**: Visualizations to build intuition about sampling distributions and interval behavior
    - **Advanced methods**: Exploration of non-standard techniques for complex scenarios
    - **Real-world applications**: Practical examples from various domains
    - **Mathematical proofs**: Formal derivations for students studying statistical theory
    
    The app is designed for students, researchers, and practitioners at different levels of statistical knowledge, from beginners to advanced users.
    
    ### Technical Implementation
    
    All simulations and visualizations are generated on-the-fly using:
    
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive data visualization
    - **NumPy**: Numerical computations
    - **SciPy**: Statistical distributions and tests
    - **Pandas**: Data manipulation
    - **Statsmodels**: Advanced statistical modeling
    
    ### How to Extend This App
    
    This application is designed to be modular and extensible. If you're interested in contributing or extending it:
    
    1. Each module is a separate Python file in the `pages` directory
    2. New methods can be added to existing modules with minimal changes
    3. New application domains can be added as separate files
    4. The structure follows a consistent pattern for easy extension
    
    For more information on how to contribute, please see the README.md file in the project repository.
    
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
    
    st.info("""
    **Citation Information**  
    
    If you use this application in your research or teaching, please cite:  
    "Confidence Intervals Explorer: An Interactive Educational Tool for Understanding Statistical Interval Estimation" (2025)
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