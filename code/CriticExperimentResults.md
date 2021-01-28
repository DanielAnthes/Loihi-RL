# Insights from Learning the Critic in a deterministic 1D environment

## Setup

- a 1D environment in which the action is fixed, so that only the critic needs to be learned
- environment is a one dimensional "path" on which the agent always moves right
- once the agent reaches the right end of the path, it receives a reward of 1, with no other rewards supplied

## Variable Settings

- simulator timestep dt: 0.001 (default for nengo simulator)
- path length = 2
- distance travelled in each timestep: 0.0002
- discount factor for learning: 0.9995
- timesteps spent at the goal state before resetting: 1000

With the above parameter settings, the agent takes 10 seconds to reach the goal and then spends an extra second in the goal state continuously receiving a reward to allow it to learn to associate this state with a reward. The discount factor is chosen such that the discounted reward is a nice gradient over the entire path.

## Findings

- the simulator step size is quite small, it seems important to adjust the discount factor and agent step size accordingly to allow for a "smooth" simulation
- the agent is able to learn a smooth value function in one shot, maybe due to the way nengo represents values (Investigate!). I think this may be because representations in nengo are smooth and correlated. In comparison to a discrete TD-learning implementation this seems to lead to much faster learning as we propagate the learned value over multiple states "for free"
- properly resetting the simulation after each episode is very important, we now force the value and delta to be zero in the reset timestep
- even though the error is delayed by at least one timestep in the current implementation the agent is still able to solve the problem (correlated representations agein?)

## Learning Rate issues

(copy from telegram)

"learning impulse when reaching the goal is MUCH larger than the minuscule difference between subsequent predictions (1 vs 0.0005) that probably either gets lost in the noise completely, or more likely: the network first learns that the reward should be close to 1, because that is where the error initially gets very large and then afterwards it goes backwards from there and learns that it should decrease its output going further away from the goal.

With the larger discount (0.9 in this case) that second step happens much quicker. We can see that in  the output above, where it sort of approaches one and then starts to differentiate between the different positions.
I think we saw this before btw, where it saturates to 1 and then learns tiny peaks, but it would have probably taken forever to get the result we get with a larger discount"
