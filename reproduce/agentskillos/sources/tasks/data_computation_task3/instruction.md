I am a computational biology PhD student, and I'm working on a chapter of my thesis analyzing predator-prey dynamics. Please perform symbolic stability analysis of the Lotka-Volterra model:
  \frac{dx}{dt} = \alpha x - \beta xy
  \frac{dy}{dt} = \delta xy - \gamma y
Parameters: \alpha=1.0, \beta=0.1, \delta=0.075, \gamma=1.5.

Find all equilibrium points, compute Jacobian matrices and eigenvalues at each, and classify stability type as stable/unstable node, saddle, center, or stable/unstable spiral.

Save to stability_analysis.json with an equilibria array, each entry containing: coordinates with x and y, eigenvalues as a list, and stability_type as a string.