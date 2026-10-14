I am a numerical analysis student, and I need to complete a coursework project comparing numerical integration methods. Please compute the symbolic and numerical solutions for the following 4 definite integrals, and generate visualizations:

1. $\int_0^{\pi} \sin(x)\,dx$
2. $\int_0^{1} e^{-x^2}\,dx$
3. $\int_1^{e} \ln(x)\,dx$
4. $\int_0^{\pi/2} \sqrt{\sin(x)}\,dx$

For each integral, apply 3 numerical methods (Trapezoidal rule N=100, Simpson's rule N=100, Gaussian quadrature) and compute the numerical approximation and error.

Output integrals.json as a JSON object with an "integrals" key containing an array of 4 objects, i.e. {"integrals": [...]}, each object containing integral_id, integrand, bounds, exact_numeric, trapezoidal, simpson, gaussian_quadrature, and the error for each method.

Generate 5 figures: integral_1.png through integral_4.png (function curve with shaded area and exact value annotation) and error_comparison.png (a grouped bar chart comparing errors of the 3 methods across all 4 integrals on a logarithmic scale). All figures should be publication quality with LaTeX labels.