# CART POLE BALANCING

## AIM
To implement and evaluate a Monte Carlo control algorithm for optimizing state-action values in a gym environment, using discretized states and policy updates.

## PROBLEM STATEMENT
The project focuses on developing an efficient Monte Carlo control algorithm specifically tailored for stabilizing the Cart-Pole system. With the objective of achieving robust stability, the aim is to fine-tune the algorithm to effectively balance the Cart Pole. Through rigorous development and testing, the goal is to optimize the algorithm's performance and reliability in stabilizing the Cart Pole under various conditions.

## MONTE CARLO CONTROL ALGORITHM FOR CART POLE BALANCING

**Step 1**: Environment Setup: Import gym and initialize the environment (e.g., CartPole). Set hyperparameters for bins, learning rates, and exploration rates.<br>
**Step 2**: Discretization: Divide the continuous state space into discrete bins to facilitate learning.<br>
**Step 3**: Policy Initialization: Initialize Q-values (state-action values) and define an epsilon-greedy policy for action selection.<br>
**Step 4**: Episode Generation: Generate trajectories by interacting with the environment using the policy. Each trajectory consists of state, action, reward, and next state information.<br>
**Step 5**: Return Calculation: Compute the cumulative discounted return<br>
**Step 6**: Policy Improvement: Update the policy to select actions that maximize the updated Q-values for each state.<br>
**Step 7**: Convergence: Repeat the process until Q-values stabilize or a predefined number of episodes is reached.<br>

## MONTE CARLO CONTROL FUNCTION
```python
def mc_control (env,n_bins=g_bins, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True, init_Q=None):

    nA = env.action_space.n
    discounts = np.logspace(0, max_steps,
                            num = max_steps, base = gamma,
                            endpoint = False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            0.9999, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                            0.99, n_episodes)
    pi_track = []
    global Q_track
    global Q


    if init_Q is None:
        Q = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    else:
        Q = init_Q

    n_elements = Q.size
    n_nonzero_elements = 0

    Q_track = np.zeros([n_episodes] + [n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[tuple(state)]) if np.random.random() > epsilon else np.random.randint(len(Q[tuple(state)]))

    progress_bar = tqdm(range(n_episodes), leave=False)
    steps_balanced_total = 1
    mean_steps_balanced = 0
    for e in progress_bar:
        trajectory = generate_trajectory(select_action, Q, epsilons[e],
                                    env, max_steps)

        steps_balanced_total = steps_balanced_total + len(trajectory)
        mean_steps_balanced = 0

        visited = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            #if visited[tuple(state)][action] and first_visit:
            #    continue
            visited[tuple(state)][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps]*trajectory[t:, 2])
            Q[tuple(state)][action] = Q[tuple(state)][action]+alphas[e]*(G - Q[tuple(state)][action])
        Q_track[e] = Q
        n_nonzero_elements = np.count_nonzero(Q)
        pi_track.append(np.argmax(Q, axis=env.observation_space.shape[0]))
        if e != 0:
            mean_steps_balanced = steps_balanced_total/e
        #progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], Steps=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}", NonZeroValues="{0}/{1}".format(n_nonzero_elements,n_elements))
        progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], StepsBalanced=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}")

    print("mean_steps_balanced={0},steps_balanced_total={1}".format(mean_steps_balanced,steps_balanced_total))
    V = np.max(Q, axis=env.observation_space.shape[0])
    pi = lambda s:{s:a for s, a in enumerate(np.argmax(Q, axis=env.observation_space.shape[0]))}[s]

    return Q, V, pi
```

## OUTPUT:
```python
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200)
```

![image](https://github.com/user-attachments/assets/bf62f64c-aa4b-473f-a044-5e42c6f63c08)
<br>
```python
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200,
                                    init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                                    init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                                    max_steps=500, init_Q=Q)
```
![image](https://github.com/user-attachments/assets/f7d80152-9ecd-4525-b6b3-ff1266fb4f4b)
<br>

```python
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=500,
                                    init_alpha = 0.01,min_alpha = 0.005, alpha_decay_ratio = 0.5,
                                    init_epsilon = 0.1 , min_epsilon = 0.08, epsilon_decay_ratio = 0.9,
                                    max_steps=500, init_Q=Q)
```
![image](https://github.com/user-attachments/assets/75081c4f-6381-452d-b4e7-3065c5d1e2aa)
<br>



## RESULT:
Thus, a Python program is developed to find the optimal policy for the given cart-pole environment using the Monte Carlo algorithm.
