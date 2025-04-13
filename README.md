# Confidence Intervals Explorer

An interactive educational application for exploring and understanding confidence intervals in statistics.

**Developed by: Vishal Bharti**
**Version: 1.0**
**Date: 2025-04-12
## Overview

The Confidence Intervals Explorer is a comprehensive educational tool designed to help students, researchers, and practitioners understand the concept of confidence intervals in statistics. This application provides interactive visualizations, simulations, and detailed explanations that bridge the gap between theoretical foundations and practical applications.

## Features

- **Interactive simulations** - Explore how confidence intervals behave under different conditions
- **Visual demonstrations** - See theoretical concepts come to life through intuitive visualizations
- **Advanced statistical methods** - Learn about various confidence interval techniques beyond the basics
- **Real-world applications** - Understand how confidence intervals are used in different fields
- **Mathematical proofs** - Detailed derivations and theoretical foundations
- **Comprehensive references** - Curated list of resources for further learning

## Modules

The application is structured into the following modules:

1. **Theoretical Foundations** - Basic definitions, properties, and interpretations of confidence intervals
2. **Interactive Simulations** - Hands-on demonstrations of coverage properties, sample size effects, and more
3. **Advanced Methods** - Bayesian credible intervals, profile likelihood, multiple testing adjustments
4. **Real-world Applications** - Examples from clinical trials, A/B testing, environmental monitoring, and manufacturing
5. **Mathematical Proofs** - Formal derivations of key confidence interval methods
6. **References & Resources** - Curated list of textbooks, research papers, and online resources

## Installation

### Requirements

This application requires Python 3.8+ and the following packages:
```
streamlit>=1.22.0
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.8.0
statsmodels>=0.13.0
```

All required dependencies are listed in `requirements.txt`.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/confidence-intervals-explorer.git
cd confidence-intervals-explorer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application with Streamlit:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Live Demo
[Access the live application on Streamlit Cloud](https://visvikbharti-confidence-intervals-explorer.streamlit.app/)
## Compatibility Notes

### Plotly Version Compatibility

If you encounter errors related to Plotly's `add_hline` or `add_vline` methods with annotations, it may be due to version compatibility issues. The error typically looks like:

```
Invalid value of type 'builtins.str' received for the 'xref' property of layout.annotation
Received value: 'paper domain'
```

#### Solution Options:

1. **Update Plotly**: Upgrade to the latest version of Plotly which may resolve these issues:
   ```bash
   pip install plotly --upgrade
   ```

2. **Fix Code Implementation**: If updating is not possible, you can modify the problematic parts of the code:
   - Separate the annotation from the line addition:
     ```python
     # Instead of
     fig.add_hline(y=value, line=dict(color='red'), annotation=dict(text="Label"))
     
     # Use
     fig.add_hline(y=value, line=dict(color='red'))
     fig.add_annotation(x=0.5, y=value, text="Label", showarrow=False, xref="paper", yref="y")
     ```
   
   - Or use scatter traces for horizontal/vertical lines:
     ```python
     # For horizontal line
     fig.add_trace(go.Scatter(
         x=[x_min, x_max], y=[value, value],
         mode='lines', line=dict(color='red')
     ))
     ```

## Extending the Application

The Confidence Intervals Explorer is designed to be modular and extensible. You can add new components or enhance existing ones by following these guidelines:

### Adding a New Module

1. Create a new Python file in the `pages` directory with a name like `XX_ModuleName.py`, where `XX` is the next available number.
2. Follow the structure of existing modules, including the page configuration and CSS styles.
3. Implement your content with appropriate sections, explanations, and interactive elements.
4. Add relevant references to the existing reference module if applicable.

### Enhancing Existing Modules

1. Locate the module file you want to enhance in the `pages` directory.
2. Add new sections, visualizations, or explanations, maintaining consistency with the existing style.
3. Update any relevant references or documentation.

### Best Practices

- Maintain a clear separation between explanation and code
- Use intuitive visualizations to illustrate concepts
- Include both theoretical background and practical applications
- Provide references for advanced or specialized topics
- Test thoroughly with different input parameters

## Contributing

Contributions to the Confidence Intervals Explorer are welcome! Please feel free to submit a pull request or open an issue to discuss potential enhancements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The open-source community behind Streamlit, Plotly, NumPy, and other libraries
- Statistical researchers and educators who have developed and refined confidence interval methods
- Everyone who has contributed to making statistical concepts more accessible and understandable

## Contact

For questions, suggestions, or feedback, please open an issue on the GitHub repository or contact the maintainers directly.