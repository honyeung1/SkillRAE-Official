I am a math education content creator, and I want to produce an educational video explaining Fourier series convergence. Please analyze the Fourier series of a square wave: f(x)=1 for 0<x<\pi, f(x)=-1 for -\pi<x<0, with period 2\pi.

Compute the Fourier coefficients b_n for n=1 to n=10 (exactly 10 coefficients, do NOT include a_0). Also compute partial sums for N=1, 3, 5, and 10 terms.

Save the results to fourier_coefficients.json with:
- A 'coefficients' array containing exactly 10 objects, each with 'n' (integer 1-10) and 'value' (the b_n coefficient value)
- A 'partial_sums' object with keys 'N=1', 'N=3', 'N=5', 'N=10', where each contains an 'expression' field showing the formula using symbolic fractions with π (e.g., '(4/pi)*sin(x) + (4/(3*pi))*sin(3*x)'), not decimal approximations

Then create fourier_animation.mp4 showing the square wave and how the Fourier approximation progressively improves. The animation must show 4 clearly distinct and separate stages for N=1, 3, 5, and 10, with visible transitions between each stage. Display the mathematical formulas (using fractions with π) in the animation. Ensure axis labels use clean formatting without excessive decimal places.