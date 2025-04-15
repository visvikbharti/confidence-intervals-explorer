"""
Enhanced helper functions to ensure mathematical formulas are visible in both light and dark modes
Includes backward compatibility with the original function names
"""

import streamlit as st

def force_visible_math_mode():
    """
    Add this to your pages to ensure formulas are visible in both light and dark modes.
    Call this function at the beginning of your app after setting the page config.
    
    Original function kept for backward compatibility.
    """
    st.markdown("""
    <script>
    // Detect the current theme
    const isDarkTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Add the appropriate class to the document body
    if (isDarkTheme) {
        document.body.classList.add('force-visible-math-dark');
    } else {
        document.body.classList.add('force-visible-math-light');
    }
    
    // Listen for theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        if (event.matches) {
            document.body.classList.remove('force-visible-math-light');
            document.body.classList.add('force-visible-math-dark');
        } else {
            document.body.classList.remove('force-visible-math-dark');
            document.body.classList.add('force-visible-math-light');
        }
    });
    
    // Enhanced dark mode detection for Streamlit
    // This is the new improved part
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.attributeName === 'data-theme') {
                const theme = document.body.getAttribute('data-theme');
                if (theme === 'dark') {
                    document.body.classList.remove('force-visible-math-light');
                    document.body.classList.add('force-visible-math-dark');
                } else {
                    document.body.classList.remove('force-visible-math-dark');
                    document.body.classList.add('force-visible-math-light');
                }
            }
        });
    });
    
    // Start observing theme changes
    observer.observe(document.body, { attributes: true });
    
    // Enhanced function to update math colors
    function updateMathStyles() {
        const isDarkTheme = document.body.getAttribute('data-theme') === 'dark';
        const mathColor = isDarkTheme ? '#ffffff' : '#000000';
        
        document.querySelectorAll('.katex, .katex-display, .katex-html, .katex *').forEach(el => {
            el.style.setProperty('color', mathColor, 'important');
        });
    }
    
    // Run periodically
    setInterval(updateMathStyles, 1000);
    </script>
    """, unsafe_allow_html=True)

def inline_math_fix():
    """
    Apply a direct CSS fix for all inline math elements.
    This is a more aggressive approach that directly targets all LaTeX elements.
    
    Original function kept for backward compatibility.
    """
    st.markdown("""
    <style>
    /* Light mode - force dark text */
    .katex { color: currentColor !important; }
    .katex .mord, 
    .katex .mbin,
    .katex .mrel,
    .katex .mop,
    .katex .mopen,
    .katex .mclose,
    .katex .mpunct,
    .katex .minner,
    .katex .mfrac,
    .katex .msqrt {
        color: currentColor !important;
    }
    
    /* Dark mode - force light text */
    [data-theme="dark"] .katex,
    .stApp[data-theme="dark"] .katex,
    body[data-theme="dark"] .katex {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    [data-theme="dark"] .katex *,
    .stApp[data-theme="dark"] .katex *,
    body[data-theme="dark"] .katex * {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Add subtle background to inline math for better contrast */
    .stMarkdown p .katex {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 3px;
        padding: 0 3px;
        margin: 0 1px;
    }
    </style>
    """, unsafe_allow_html=True)

def add_math_enhancement_script():
    """
    Add a JavaScript function to enhance math rendering in dark mode
    This is a more aggressive approach that should work in most cases
    """
    st.markdown("""
    <script>
    // Function to enhance math rendering
    function enhanceMathRendering() {
        const isDarkTheme = document.body.getAttribute('data-theme') === 'dark';
        const textColor = isDarkTheme ? '#ffffff' : '#000000';
        const bgColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';
        
        // Apply styles to KaTeX elements
        document.querySelectorAll('.katex, .katex-display').forEach(el => {
            el.style.color = textColor;
            el.style.setProperty('color', textColor, 'important');
        });
        
        // Style math blocks for better visibility
        document.querySelectorAll('.math-block').forEach(el => {
            el.style.backgroundColor = isDarkTheme ? 'rgba(30, 30, 30, 0.6)' : 'rgba(245, 245, 245, 0.7)';
            el.style.borderColor = isDarkTheme ? 'rgba(100, 100, 100, 0.6)' : 'rgba(200, 200, 200, 0.6)';
        });
        
        // Style all math elements
        document.querySelectorAll('.katex .mord, .katex .mbin, .katex .mrel, .katex .mop, .katex .mopen, .katex .mclose, .katex .mpunct, .katex .minner, .katex .mfrac').forEach(el => {
            el.style.color = textColor;
            el.style.setProperty('color', textColor, 'important');
        });
    }
    
    // Run on page load and set up observers
    document.addEventListener('DOMContentLoaded', function() {
        // Initial enhancement
        enhanceMathRendering();
        
        // Watch for theme changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.attributeName === 'data-theme') {
                    enhanceMathRendering();
                }
            });
        });
        
        observer.observe(document.body, { attributes: true });
        
        // Watch for new content
        const contentObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length) {
                    enhanceMathRendering();
                }
            });
        });
        
        contentObserver.observe(document.body, { childList: true, subtree: true });
        
        // Periodic check as a fallback
        setInterval(enhanceMathRendering, 2000);
    });
    </script>
    """, unsafe_allow_html=True)

def setup_math_rendering():
    """
    Complete setup for math rendering in both light and dark modes.
    Call this function at the beginning of your app after setting the page config.
    
    This is a new comprehensive function that applies all fixes at once.
    """
    # Apply original functions for backward compatibility
    force_visible_math_mode()
    inline_math_fix()
    
    # Apply additional JavaScript enhancement
    add_math_enhancement_script()
    
    # Add custom MathJax configuration
    st.markdown("""
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        messageStyle: "none",
        tex2jax: {
            inlineMath: [['$','$'], ['\\(','\\)']],
            displayMath: [['$$','$$'], ['\\[','\\]']],
            processEscapes: true
        },
        "HTML-CSS": { 
            availableFonts: ["TeX"], 
            scale: 100,
            styles: {
                ".MathJax": {
                    color: "inherit !important"
                }
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)