# VI-method State Value
Value iteration method over State Value

## Taxonomy
- Model-free
- Value-based
- On-Policy

## Algorithm

1. Initialize the value of all states V<sub>i</suub> to some value
1. Let the agent randomly explore the environment for some steps.
1. For all states in the Markov decision process, perform the Bellman update:<br>
V<sub>s</sub>&larr;max<sub>a</sub>&sum;<sub>s'</sub>p<sub>a,s&rarr;s'</sub>(r<sub>s,a</sub>+&#x213D;V<sub>s'</sub>)
1. Test the agent.
1. Repeat from step 2. until convergence.

