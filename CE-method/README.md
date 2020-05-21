# CE-method
## Taxonomy
- Model-free
- Policy-based
- On-Policy

## Algorithm

1. Play N episodes using the current model.
1. Calculate the total reward for every episode and rank them by it.
1. Select the best episodes.
1. Train the model over those episodes, using the observations as inputs and the actions as target.
1. Repeat from step 1. until convergence criterion.