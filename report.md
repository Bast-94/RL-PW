# Rapport du TP3 de Reinforcement learning



## Q-learning

Le Q-learning est un algorithme d'apprentissage par renforcement qui vise à déterminer la meilleure action à prendre dans un environnement pour maximiser la récompense cumulative. Il maintient une table (Q-table) pour chaque paire état-action:

### Implementation et description de l'algorithme

Dans cette section le but sera de presenter le pseudo-code de Q-learning en faisant reference aux methodes implementees. L'algorithme est implemente dans la methode `QLearningAgent.play_and_train` disponible dans [q_learning.py](q_learning.py) et fonctionne de la maniere suivante:

**Initialisation** :

$Q[s,a] \leftarrow 0 \ \forall (s,a) \in \mathcal{S} \times \mathcal{A} $

$\pi$ correespond a la politique de l'agent, initialement il s'agit de la politique $\varepsilon \text{-greedy}$. 

On fixe $\alpha$ le taux d'apprentissage `learning_rate`

**Pour chaque épisode** :
- $s \leftarrow \text{initial state}$

- $\text{terminated} \leftarrow \text{False} $

- $\text{While not terminated}$:

   - $a \leftarrow \text{get\_action}_{\pi}(s)$ // choix de l'action a l'etat $s$ seolon la politique $\pi$ voir la methode `QLearningAgent.get_action(state : State)` 

   - $s',r , \text{terminated} \leftarrow \text{execute}(a)$ // `gym.Env.step(a: Action)`

   - $ Q(s, a) \leftarrow Q(s, a) + α * [r + γ * max_{a' \in \mathcal{A}}(Q(s', a')) - Q(s, a)]$ // appel a la methode `QLearningAgent.update(state: State, action: Action, reward: t.SupportsFloat, next_state: State)`
   - $s \leftarrow s' $


#### Politique $\varepsilon \text{-greedy}$

- $\mathbb{P}(a=max_{a' \in \mathcal{A}}(Q(s', a'))\ |\ s'  ) = 1- \varepsilon$


## Q-learning avec un ordonnancement de $\varepsilon$

L'ordonnancement de $\varepsilon$ consiste a fixe une valeur de depart relativement elevee ($\varepsilon_{start}$)  reduire epsilon au fur et a mesure de l'exploration de l'agent vers ($\varepsilon_{end}$). Le but est de permettre une large exploration au debut et de se focaliser porgressivement sur les valeurs de $Q[s,a]$.

### Implementation de la mise a jour de $\varepsilon$

Voici comment donctionne l'ordonnancement de $\varepsilon$:





## Sarsa Agent

### Implementation


![q-learning-eps](img/qlearning_eps.gif)

## Comparison 

###

### Visuels

|      | Q-learning   | Q-learning eps   | SARSA   | SARSA with softmax  policy | Q-learning with softmax policy|
|-----:|:-------------|:-----------------|:--------|:---------------------|:---------------------|
|  250 | ![](img/qlearning-250-ep.gif)| ![](img/qlearning-eps-250-ep.gif) | ![](img/sarsa-250-ep.gif) | ![](img/sarsa-softmax-250-ep.gif) | ![](img/qlearning-eps-softmax-250-ep.gif) |
|  500 | ![](img/qlearning-500-ep.gif)| ![](img/qlearning-eps-500-ep.gif)  | ![](img/sarsa-500-ep.gif) | ![](img/sarsa-softmax-500-ep.gif) | ![](img/qlearning-eps-softmax-500-ep.gif) |
| 1000 | ![](img/qlearning-1000-ep.gif) | ![](img/qlearning-eps-1000-ep.gif) | ![](img/sarsa-1000-ep.gif) | ![](img/sarsa-softmax-1000-ep.gif) | ![](img/qlearning-eps-softmax-1000-ep.gif) |
