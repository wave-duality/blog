---
title: "CatanRL"
date: "2026-10-13"
excerpt: |-
    In this project, we study a two-player Catan variant based on Colonist’s 1v1 format. Our implementation makes the environment more tractable for reinforcement learning while preserving the core ruleset. We train CatanRL, a 1.32M parameter graph neural network, from randomly initialized weights through search-guided self-play. We combine these predictions with Monte Carlo tree search using Gumbel-based root action selection to explore possible continuations.

    Our interpretability work focuses on opening settlement placement, context-dependent resource valuation, and competition for road expansion. The goal is to understand whether we can learn, from scratch, useful principles in human play.
published: true
paper: true
author: "Owen Zhang"
abstract: ""
---

## Acknowledgements

Thank you to my good friend Alex Wa ([djdumpling](https://djdumpling.github.io/)) for the continued support and motivation, my roommate Andrew Liu for playing countless rounds of Colonist with me, and [Modal](https://modal.com/) for nearly $1000 in compute credits to support the training process. Congratulations on the Series D!

## Introduction

Among board games that can be picked up without having to digest an unreasonable amount of rules, Catan might be one of my favorite. I think the game teaches several important lessons about trading and strategy games in general. 

I got sucked into this rabbit hole after various friends introduced me to Colonist, the largest competitive Catan community. I ended up playing a lot (everywhere): it replaced Instagram Reels before bed, filled ferry delays in Greece, and fueled "productive" study breaks.

The primary idea is that trading can genuinely be something that is mutually beneficial. Similarly to games like Figgie, one may initially expect that trades between two players must be 0 EV, since only one player may win at the end (in particular, the sum of each players' probabilities of winning, assuming a solved game, at any state, is 1). This may be true globally, but with this set of assumptions alone, another possibility you may arrive at is that the players who are not participating in the trade are simply taking an EV loss, while the two participating create some EV between them. This turns out to be the better explanation, at least when trades are made rationally. Thus, part of the game of Catan is figuring out what trades you should be making, with who, and how to convince them to make it with you.

Another idea is that players can create value in different areas of the game, and different players are better positioned to do different things. At a very basic level, players that have a large production of brick and wood are better allocated to win “longest road” than others. This doesn't mean that you can't compete with them in this aspect, but that it is comparatively worse for you. I would say that victory point generation is best divided into three categories: development cards, cities, and settlements/roads. This creates an interesting game since there are usually four players, and reaching the objective of 10 victory points usually requires investment in at least two of these categories.

There are other things you can think about, like tradeoffs between resource production and port access, long-run vs. short-run victory optimization, and more granular things. But I think these may be less obvious conclusions that one arrives at after playing a few times.

Finally, I think the most interesting section of this article is the last one — I was motivated to finish this project mostly because I was curious what RL could teach us about approaching Catan as humans, so check out the gallery of interpretability experiments I carried out :)

## Setup

Catan, in its canonical four-player format, is difficult to model within an RL environment for two main reasons:

- There is some level of incomplete information; for instance, you do not know which development cards your opponents are holding, or which resource Opponent A has just stolen from Opponent B.

- The second is the trading aspect, which is the dynamic that makes Catan such an illustrative (and frustrating) game. Assuming that everyone's objective function is to optimize their individual probability of winning (as opposed to “throwing” the game or “kingmaking”), players that are not first may try to trade more often with each other, as an example.

It is not completely unreasonable to try to model Catan with these restrictions, but Catan is already computationally demanding without these social elements. We therefore use a two-player environment with several modifications:

- **Two players.**
- **15 victory points to win.**
- **A higher discard threshold.** Players discard on a seven only when holding **10 or more resource cards**.
- **Friendly robber protection.** Players with **two or fewer public victory points** are protected from robber placement and theft, subject to the exceptional placement case described below.

The following specifies the environment used throughout the project. Standard Catan building and connectivity rules otherwise apply.

<details>
<summary><strong>Board composition, number tokens, and ports</strong></summary>

We use the standard board geometry: **19 hexes, 54 intersections, and 72 road locations**.

| Terrain | Number of hexes |
|---|---:|
| Wood | 4 |
| Brick | 3 |
| Sheep | 4 |
| Wheat | 4 |
| Ore | 3 |
| Desert | 1 |
| **Total** | **19** |

The 18 producing hexes receive the following number tokens:

| Number | Copies of each number | Production probability per roll |
|---|---:|---:|
| 2 and 12 | 1 each | 1/36 each |
| 3 and 11 | 2 each | 2/36 each |
| 4 and 10 | 2 each | 3/36 each |
| 5 and 9 | 2 each | 4/36 each |
| 6 and 8 | 2 each | 5/36 each |

The desert has no number, and no hex carries a seven. A tile’s **pip count** is its production probability multiplied by 36.

For generated training boards:

- Terrain and number tokens are shuffled independently.
- Boards are rejected if two red numbers—any combination of **6 and 8**—share an edge.
- Boards are also rejected if two **identical numbers** share an edge.
- There are **nine fixed coastal port locations**, with randomly shuffled port types: **four generic 3:1 ports** and **one 2:1 port for each resource**.

Dice follow the ordinary distribution of two independent fair six-sided dice. There is one roll per regular turn, with no mechanism that balances results across previous rolls.

</details>

<details>
<summary><strong>Opening placement: A1 → B1 → B2 → A2</strong></summary>

Players begin with empty resource hands and place their starting pieces in this order:

1. **A1:** Player A places a settlement and its adjoining road.
2. **B1:** Player B places a settlement and its adjoining road.
3. **B2:** Player B places their second settlement and its adjoining road.
4. **A2:** Player A places their second settlement and its adjoining road.

Each road is chosen immediately after its settlement, before the next settlement is placed.

- Each player finishes setup with **two settlements, two roads, and two public victory points**.
- Only the **second settlement** supplies starting resources: **one card per adjacent non-desert hex**, regardless of that hex’s number or pip count.
- No dice are rolled during setup.
- **Player A takes the first regular turn.**

</details>

<details>
<summary><strong>Piece limits, resource supply, costs, and scoring</strong></summary>

Each player has the following pieces. Starting pieces count toward these limits.

| Piece | Supply per player | Resource cost | Victory points |
|---|---:|---|---:|
| Road | 15 | 1 wood + 1 brick | 0 |
| Settlement | 5 | 1 wood + 1 brick + 1 sheep + 1 wheat | 1 |
| City | 4 | 2 wheat + 3 ore | 2 |

Cities replace settlements and return the settlement piece to the player’s supply. A city is worth **two points total**, increasing the previous settlement’s score by one.

The shared bank starts with **19 cards of each resource**, or **95 resource cards total**. Resource supply is finite; spent and discarded resources return to the bank.

Additional scoring:

| Source | Points | Requirement |
|---|---:|---|
| Longest Road | 2 | A qualifying road of at least 5 segments |
| Largest Army | 2 | At least 3 played knights |
| Victory-point development card | 1 each | Held privately |

The usual bonus ownership rules apply: surpassing the holder transfers the bonus; matching them does not.

A player wins upon reaching **at least 15 total points on their own turn**, including hidden victory-point cards.

**Buildings alone provide at most 13 points:** five settlements plus four cities. Winning therefore requires Longest Road, Largest Army, victory-point cards, or some combination of them.

</details>

<details>
<summary><strong>Development deck and information available to players</strong></summary>

There is one shared **25-card development deck**, drawn without replacement.

| Card | Count | Effect |
|---|---:|---|
| Knight | 14 | Move the robber; eligible theft; counts toward Largest Army |
| Victory point | 5 | Add one hidden victory point |
| Road Building | 2 | Place up to two free roads |
| Year of Plenty | 2 | Take up to two available resource cards from the bank |
| Monopoly | 2 | Take all of one chosen resource from the opponent |
| **Total** | **25** | |

A development card costs **1 sheep + 1 wheat + 1 ore**.

Information is divided as follows:

- **Public:** board contents, pieces, ports, robber location, both resource-hand compositions, bank inventory, public scores, played development cards, development-card hand sizes, and how many cards were bought this turn.
- **Private:** the identities of each player’s unplayed development cards.
- **Unknown to either player:** the exact composition of the remaining development deck, except where it can be deduced from known cards.

Each player sees their own development cards. Hidden victory-point cards count toward winning immediately, including when purchased, but do **not** contribute to the public score used for friendly robber protection.

</details>

<details>
<summary><strong>Sevens, discarding, and the friendly robber</strong></summary>

On a seven, players holding **10 or more resource cards** discard half, rounded down. If both players must discard, the current player goes first.

**Friendly robber protection** applies to players with **two or fewer public victory points**. For both sevens and knights, the robber cannot move onto a hex adjoining either protected player's buildings, including the moving player's own buildings.

If protection leaves no destination other than the robber's current hex, the placement restriction is relaxed. Stealing from a protected opponent remains forbidden.

</details>

<details>
<summary><strong>Trading and other implementation-specific details</strong></summary>

- **Trading:** player-to-player trades are unavailable. Bank trades are **4:1**, generic ports **3:1**, and resource-specific ports **2:1**. The requested resource must be in stock.
- **Development-card timing:** only a knight can be played before rolling. Road Building, Year of Plenty, and Monopoly require the roll first.
- **Production shortages:** if the bank cannot cover a resource's entire payout for a roll, nobody receives that resource. Other resources are resolved independently.
- **Partial development-card effects:** Road Building places as many roads as legal locations and pieces allow. Year of Plenty takes as many cards as the bank can supply. Either card is consumed even if it provides no benefit. Monopoly may name a resource the opponent does not hold.

</details>

The examples below use **Generation 37, the final generation we trained in our training run**.

There are now two interesting claims that we can make. <b>These have pretty important implications for the training.</b>

<div class="claim">

**Claim 1: Player-to-player trading cannot strictly improve both players’ expected results under a common evaluation.**

</div>

Notably, player-to-player trading is something we completely exclude from our environment. The motivation is that two-player Catan is a zero-sum game. There is a finite amount of states in Catan (even with stochasticity), and thus there exists some optimal move for each player at every state. Under this solved paradigm, since the probability of win for both players must sum to 1, there can not exist a trade that improves both player's probability of winning.

A trade may help both players build more efficiently in the short-run, but it can not possibly establish a mutual improvement in their expected result. This follows directly from the definition of [zero-sum utility](https://www.cs.cmu.edu/~15281/coursenotes/gametheory/index.html).

So it is true that players can disagree about a trade’s value because of private information (myopically, a trade may seem good for both players). However, under optimal play, either player can reject every trade and retain their original guaranteed outcome. If $v$ is Player A’s minimax value without trading, A’s ability to refuse trades guarantees $V_{\text{trade}}\geq v,$ while B’s ability to refuse guarantees $V_{\text{trade}}\leq v$. Thus,

$$
v\leq V_{\text{trade}}\leq v
\quad\Rightarrow\quad
V_{\text{trade}}=v.
$$

**Removing trading is therefore a reasonable modeling choice that preserves the game’s equilibrium value.**

<div class="claim">

**Claim 2: The resources that players hold is exactly deducible by both players under our environment’s information rules.**

</div>

Our environment chooses to exposes the bank’s remaining supply of each resource. Because there are exactly **19 cards of each resource**, distributed between the bank and the two players, either player can reconstruct the opponent’s holdings.

For example, if the bank contains 11 wheat and I hold 3, my opponent must hold the remaining 5. 

To justify that this is a modeling choice that remains true to the spirit of 1v1 Catan, we can compare this to Colonist's implementation, in which every possible game event preserves public resource counts (to the best of my knowledge):

- **Discarding after a seven:**  the cards that are discarded are publicly published in the action log.
- **Year of Plenty:** the chosen resources are publicly published.
- **Monopoly:** resources move directly between the two players. Both participants know which resource was taken and how many cards changed hands.
- **Robber theft, including a knight’s effect:** the stolen card moves directly between the two players. 

The difference is that, in Colonist, the action log exposes every change of resources instead of providing a public bank count (which, in my opinion, is a bit silly, since memorizing doesn't seem like an interesting part of Catan strategy).

Thus, we directly supply the resources counts in both hands to the CatanRL network. These naturally seem like important parameters so we feature engineer this to (hopefully) accelerate the model's learning. However, development-card identities remain private, so the game is still a hidden-information game.

## Representation & Feature Engineering

We'll follow one position through the encoder and network. We're 65 player turns in, and A leads by 1 public victory point and 2 including hidden points.

<iframe src="/figures/catan-tile-vector.html" title="Explore the full Catan board and all four model input groups"></iframe>

Select a tile, intersection, road location, or global feature to inspect its vector.

We chose these features using our human knowledge of Catan, and what I thought might be important for a network to know. Resources, ports and building types use **one-hot encoding**. Dice numbers appear both as exact categories and as **pips**, which express how often they roll. A 6 and an 8 are equally likely, but having two 6 tiles is not the same gameplay-wise as having a 6 and a 8 (the two 6 tiles always produce together). Keeping both encodings preserves production rate and correlation (useful for reducing variance or matching an opponent's production).

We also supply production totals by resource, nearby expansion routes, build affordability, and thresholds such as “at least three ore.” Counts are scaled by fixed divisors, and ownership is relative to the acting player.

The visible board, pieces and resource hands can be recovered from the encoding using the fixed board geometry.

Our encoding loses some information about the game (i.e. the map is not invertible). For example, it includes the opponent’s development-card count and holding-time counters, but not the full history of when cards were acquired or played. Different histories can therefore produce the same model input, even when they would support different estimates of the opponent’s hidden victory points.

Post-mortem I do think it'd be worth making this encoding fully complete. There's a lot of granular decisions that are important especially as both players approach 15 VPs, so it seems important for the bot to have the most complete representation possible.

<details>
<summary>Information outside the 4,752 values</summary>

The turn counter is capped at 200, and development-card history counters at 40.

Generation 37 also omits the flag recording whether a development card has already been played this turn. The rules engine uses that flag when constructing the legal-action mask, but the value head never receives it. This can matter when evaluating what a player could build using Year of Plenty before ending their turn.

The legal-action mask and fixed board connections are supplied separately from the 4,752 features.

</details>

## Model

The **graph neural network (GNN)** processes the 4,752 inputs as 19 tile vectors, 54 intersection vectors, 72 road-location vectors, and one global vector. Connected locations exchange information over four message-passing rounds.

Generation 37 has **1,320,375 learned parameters** and a hidden width of **128**. It uses mean aggregation and predicts a policy and a scalar value.

<iframe src="/figures/catan-gen37-model.html" title="Generation 37 architecture: tensors, weight matrices, four graph blocks, and output heads"></iframe>

Each round outputs $(19+54+72+1)\times128=18{,}688$ activations across the four groups.

### 1. Turn each input row into 128 learned features

We apply a linear layer followed by ReLU to each input group.

| Input | Shared transformation | Output shape |
|---|---|---|
| 19 tiles, 32 features each | $32\rightarrow128$, then ReLU | $H^{(0)}\in\mathbb{R}^{19\times128}$ |
| 54 intersections, 48 features each | $48\rightarrow128$, then ReLU | $V^{(0)}\in\mathbb{R}^{54\times128}$ |
| 72 road locations, 16 features each | $16\rightarrow128$, then ReLU | $P^{(0)}\in\mathbb{R}^{72\times128}$ |
| 400 global features | $400\rightarrow128$, then ReLU | $g^{(0)}\in\mathbb{R}^{128}$ |

The same $128\times48$ weight matrix processes all 54 intersections, sharing learned patterns across corners. Their features and connections distinguish them.

### 2. Exchange information along the board

Consider the case where a promising settlement location is close to an opponent’s road network. This could be a good or bad thing, but it certainly should be something our model has enough expressiveness to handle. Message passing lets its representation incorporate that surrounding context.

Each graph round constructs the following updates:

| Representation updated | Information concatenated before the update | Update network |
|---|---|---|
| Each intersection | Itself; mean of adjacent tiles; mean of incident road locations; mean of neighboring intersections; global vector | $640\rightarrow128\rightarrow128$ |
| Each tile | Itself; mean of its six intersections; global vector | $384\rightarrow128\rightarrow128$ |
| Each road location | Itself; mean of its two endpoint intersections; global vector | $384\rightarrow128\rightarrow128$ |
| Global vector | Itself; mean of all tiles; mean of all intersections; mean of all road locations | $512\rightarrow128\rightarrow128$ |

Each intersection concatenates five 128-dimensional vectors. A $640\rightarrow128\rightarrow128$ MLP with ReLU produces a residual update:

$$
v_i^{(\ell+1)}
=
v_i^{(\ell)}
+
f_V^{(\ell)}\!\left(
\left[v_i^{(\ell)},\ \overline{h}_{N_H(i)}^{(\ell)},\ \overline{p}_{N_P(i)}^{(\ell)},\ \overline{v}_{N_V(i)}^{(\ell)},\ g^{(\ell)}\right]
\right).
$$

Brackets denote concatenation and bars denote neighbor averages. Missing neighbors at coastal intersections are masked out.

All four updates in a round read the previous round’s representations. We repeat this process four times, using **different learned weights in each round**, while preserving all tensor shapes. There is no extra ReLU after the residual addition, so later representations can contain negative values.

The global vector collects three board averages in one round and broadcasts the result in the next. This gives distant locations a two-round communication path through a shared summary.

<details>
<summary>The complete update equations</summary>

For round $\ell$, let $\operatorname{mean}_{N}(X)$ average the rows of $X$ in a location’s fixed neighborhood. The updates are:

$$
\begin{aligned}
V' &= V + f_V([V,\operatorname{mean}_{V\leftarrow H}(H),\operatorname{mean}_{V\leftarrow P}(P),\operatorname{mean}_{V\leftarrow V}(V),g]),\\
H' &= H + f_H([H,\operatorname{mean}_{H\leftarrow V}(V),g]),\\
P' &= P + f_P([P,\operatorname{mean}_{P\leftarrow V}(V),g]),\\
g' &= g + f_g([g,\operatorname{mean}(H),\operatorname{mean}(V),\operatorname{mean}(P)]).
\end{aligned}
$$

In a local update, $g$ is broadcast to every row. Each $f$ has the form $W_2\operatorname{ReLU}(W_1x+b_1)+b_2$, with a separate parameter set for each location type and round. Neighbor averaging itself has no learned parameters. The adjacency tables are fixed geometry loaded from `adjacency.json`.

</details>

### 3. Read out a policy and a value

After round four, linear policy heads produce **241 logits**:

| Head | Reads | Output slots |
|---|---|---:|
| Build a road | Each final road-location vector | 72 |
| Build a settlement | Each final intersection vector | 54 |
| Upgrade to a city | Each final intersection vector | 54 |
| Move the robber | Each final tile vector | 19 |
| Other actions | Final global vector | 42 |
| **Total policy output** | | **241** |

The remaining **42 action slots** are:

- **Buy a development card (1).**
- **Play a development card (4):** Knight, Road Building, Year of Plenty, or Monopoly.
- **Choose a resource (5):** wood, brick, sheep, wheat, or ore, for Year of Plenty or Monopoly.
- **Discard one card (5):** one slot per resource type.
- **Bank/port trade (25):** each give/receive resource pair ($5\times5$), using the player's applicable 4:1, 3:1, or 2:1 rate.
- **End the turn (1).**
- **Roll the dice (1).**

Illegal actions are masked out, including the five trades that give and receive the same resource.

The settlement logit at location $i$ is:

$$
\ell_{\mathrm{settle},i}=w_{\mathrm{settle}}^\top v_i^{(4)}+b_{\mathrm{settle}}.
$$

The same 128 weights and one bias score all 54 intersections.

The rules engine supplies the legal-action set $\mathcal{A}(s)$. We mask illegal actions and apply a softmax over the rest:

$$
p_\theta(a\mid s)=
\begin{cases}
\displaystyle\frac{\exp(\ell_a)}{\sum_{b\in\mathcal{A}(s)}\exp(\ell_b)}, & a\in\mathcal{A}(s),\\
0, & \text{otherwise}.
\end{cases}
$$

A search-free bot chooses the highest-probability legal action.

A separate head reads the final global vector and predicts one value:

$$
v_\theta(s)=\tanh(w_v^\top g^{(4)}+b_v).
$$

The value lies between $-1$ and $1$, with positive values favoring the acting player. Like a chess engine’s evaluation, it estimates who is better off in a position. We can express this as an implied win probability for the acting player, $p=(v_\theta(s)+1)/2$, with $1-p$ for the opponent.

Learning to evaluate positions is one part of training: the target combines the eventual game result with a search estimate. Search uses this head to evaluate possible continuations, and we later compare its implied winning probabilities with actual outcomes to see how well it predicts who wins.

### Worked example: why the forward pass selects road 27

In the position above, Generation 37’s highest-scoring legal action is **building a road at edge 27**, connecting intersections **18 and 23**. The recorded game also played that move.

1. **Encode the edge.** Road location 27 starts as a 16-number input row. The shared road-location encoder maps it to a 128-number vector, $P_{27}^{(0)}$.
2. **Read its surroundings.** In each of four rounds, it combines its previous vector, the mean of intersections 18 and 23, and the previous global vector. Those $3\times128=384$ values pass through the round’s $384\rightarrow128\rightarrow128$ update network. Adding that update produces the next $P_{27}$.
3. **Score the move.** The road head takes a weighted sum of the final 128 activations and adds its bias. For this edge, the activation–weight products sum to $2.685312$, and the bias is $-0.122642$, giving a logit of **2.562670**.
4. **Compare legal actions.** The other heads score the alternatives. The legal-action mask removes 211 of the 241 slots, and a softmax over the remaining 30 gives road 27 **30.60%** of the probability mass. It has the highest probability, so a greedy network-only decision selects it. We could also sample an action from these probabilities.
5. **Translate the action.** Road actions occupy indices 0–71, so index 27 decodes directly to `BuildRoad(27)`. Applying it places A’s road on that edge and spends one wood and one brick: A’s holdings change from 4 wood / 2 brick to 3 wood / 1 brick.

| Legal action | Raw logit | Network policy probability |
|---|---:|---:|
| **Road at edge 27** | **2.5627** | **30.60%** |
| City at intersection 22 | 2.2104 | 21.52% |
| Buy a development card | 1.6179 | 11.90% |
| Road at edge 49 | 1.3241 | 8.87% |
| City at intersection 23 | 1.2657 | 8.37% |

At road 27, channel 125 has a final activation of $-8.982087$ and a head weight of $-0.157480$, contributing **+1.414500** to the logit.

The value head outputs $0.8309$ for A.


<details>
<summary>Where the 1,320,375 parameters are</summary>

| Component | Parameters |
|---|---:|
| Tile embedding, $32\rightarrow128$ | 4,224 |
| Intersection embedding, $48\rightarrow128$ | 6,272 |
| Road-location embedding, $16\rightarrow128$ | 2,176 |
| Global embedding, $400\rightarrow128$ | 51,328 |
| Four graph rounds, $4\times312{,}320$ | 1,249,280 |
| Four location-specific policy head types, $4\times129$ | 516 |
| Other-action head, $128\rightarrow42$ | 5,418 |
| Value head, $128\rightarrow1$ | 129 |
| Auxiliary intersection-ownership head, $128\rightarrow5$ | 645 |
| Auxiliary road-ownership head, $128\rightarrow3$ | 387 |
| **Total** | **1,320,375** |

Within each graph round, the intersection update has 98,560 parameters, tile and road updates have 65,792 each, and the global update has 82,176. The two auxiliary heads are trained but not evaluated for ordinary move selection. Excluding their 1,032 parameters leaves **1,319,343 parameters used by the policy/value forward pass**.

</details>

### Why this architecture?

Weight sharing makes the same local pattern recognizable across the board. Message passing gives each location access to its particular neighbors before information is pooled. The global vector supplies resources, scores, and board-wide context. Location-specific heads keep a settlement score attached to the intersection being considered.

The model is small enough to evaluate repeatedly inside search. Mean aggregation is inexpensive, though it collapses neighborhoods that produce the same average.

If you’re observant enough, you might notice that the policy chooses one action at a time, although in Catan you can do many things in a turn. In particular, many actions do not depend on each other (if I want to upgrade to a city and build a road, either order can lead to the same position). However, our search treats these sequences separately, and the model can have a logit preference for which to do first. One thing I realized after this training run is that we should probably treat equivalent action sequences as identical. This could improve sample efficiency and cut out some redundant search.

## Learning Loop

The project combines four operations with the last one being the only one that actually updates weights:

| Operation | Input → output | What changes? |
|---|---|---|
| **Apply rules** | State + action → next state | The board and game state |
| **Evaluate** | Observation → policy and value | Temporary GNN activations |
| **Search** | Current position → move and learning targets | A tree of visits and value estimates |
| **Train** | Recorded positions + targets → updated GNN | Neural weights |

<iframe src="/figures/catan-gen37-training.html" title="How frozen Generation 36 generates search targets and games that train Generation 37"></iframe>

### 1. Apply the rules

The **rules engine** knows which moves are legal, what they cost, how pieces affect the board, and when somebody has won. Given “build road 27,” it places the road and deducts one wood and one brick. It also supplies probabilities for dice, development-card draws, and thefts. This is just a nice layer of abstraction so that we don't have to worry about any of this.

### 2. Evaluate a position

The **same GNN** supplies both search priors and position evaluations through its policy and value heads.

We can call this frozen network once to choose a move cheaply, or repeatedly inside search to evaluate hypothetical positions.

### 3. Search

**Monte Carlo tree search (MCTS)** uses the engine to construct continuations from the current position (the root) and the frozen GNN to evaluate them.

One simulation does four things:

1. Select a root action to investigate.
2. Follow actions and chance outcomes through a copied game state.
3. At a new decision position, obtain a policy and value from the same GNN. At a terminal position, use the actual result instead.
4. Add that evaluation to the visited tree edges' statistics.

Each edge stores a **visit count $N(s,a)$** and a **value estimate $Q(s,a)$**, the mean backed-up evaluation from the choosing player's perspective. Simulations usually stop at a network-evaluated leaf.

#### Gumbel at the root \& PUCT inside the tree

At the root, **Gumbel selection** shortlists up to eight legal actions using the network's log probabilities plus random noise. The noise gives less-favored moves opportunities. Search investigates the candidates, ranks them using their original noisy scores and accumulated Q estimates, and progressively removes roughly half:

$$
8\ \text{candidates}\;\longrightarrow\;4\;\longrightarrow\;2\;\longrightarrow\;1.
$$

This is **sequential halving**. The Gumbel noise is drawn once per action and held fixed through these rounds.

Below the root, **PUCT** chooses actions using their current Q estimate plus an exploration bonus. The bonus favors actions with a high network prior and relatively few visits. At an opponent's decision, Q is oriented to that opponent, so search models them trying to improve their own result. The sign follows the acting player, not tree depth: one Catan turn can contain several consecutive actions by the same player.

<details>
<summary>The selection formulas and chance handling</summary>

For root action $a$, draw $u_a\sim\operatorname{Uniform}(0,1)$ and set

$$
g_a=-\log(-\log u_a),\qquad
r_a=\log\max(p_\theta(a\mid s),10^{-9})+g_a.
$$

The shortlist contains the eight largest $r_a$ values, or all legal actions if fewer exist. Halving ranks survivors by $r_a+\sigma Q(s,a)$, with $\sigma=50+\max_b N(s,b)$ under the Generation 37 run's settings. Remaining simulations after elimination investigate the finalist.

At interior decision nodes, PUCT selects the action maximizing

$$
Q(s,a)+c_{\mathrm{puct}}p_\theta(a\mid s)
\frac{\sqrt{\max(1,N(s))}}{1+N(s,a)}.
$$

The policy prior stays fixed after expansion while Q and visit counts are updated. Ties are broken randomly.

For chance events, search schedules the outcome furthest behind its engine-supplied probability share, with random tie-breaking. Forced transitions can be advanced without a new policy choice.

</details>

**A note on luck.** Search uses the theoretical dice probabilities (for example, a 6 has probability $5/36$) to distribute simulations across possible outcomes. Actual self-play games still sample dice rolls, development-card draws and robber thefts, so good decisions can lose through bad luck. Generation 37's value target combines 35% of the actual game result with 65% of the search estimate. Learning from many games reduces this sampling noise, but a finite training run can still get unlucky.

#### Hidden cards and search outputs

For Generation 37's training data, each searched decision received **1,024 total simulations**. When the opponent held hidden development cards, search normally sampled four possible allocations consistent with the acting player's information and ran a separate 256-simulation tree for each. Otherwise, one tree received the budget. Their final policies are averaged equally.

Search returns **a move, a policy target $\pi$, and a value estimate**. The move advances the game; the other outputs become learning targets. The policy combines the prior and search Q estimates, rather than simply normalizing visit counts.

<details>
<summary>The exact target, why it can disagree with the played move, and the hidden-information approximation</summary>

Within one world, the target is a **completed-Q policy**:

$$
\pi(a\mid s)=\operatorname{softmax}_{a\in\mathcal A(s)}
\left[\log\max(p_\theta(a\mid s),10^{-9})+\sigma\widehat Q(s,a)\right],
\qquad \sigma=50+\max_a N(s,a).
$$

Visited actions use their Q estimates. Unvisited actions use a common completion value that blends the root's network value with prior-weighted Q estimates of visited actions. Q normalization and adaptive sigma scaling were disabled. With several hidden-card worlds, the final target averages their policies, not their logits. This formula describes the distribution before recording; the recording format omits probabilities at or below $10^{-6}$.

In a single-world search, sequential halving chooses the finalist before leftover simulations investigate it. The final policy can therefore favor a different action from that finalist. With multiple worlds, the returned move instead maximizes the averaged policy. Self-play may then sample a move from that policy during its early exploration window, described below.

The hidden-world sampler uses only the acting player's information, but deeper decisions can behave as if the sampled allocation were known. This limits the search's treatment of imperfect information.

</details>

#### Self-play: collecting games

**Self-play** repeats this process until the game ends. We save learner positions with their search policy and value, then attach the final result and piece ownership.

To produce **Generation 37**, the frozen evaluator was **Generation 36**. Opponents were sampled with weights of 75% mirror play and 25% a frozen `exploiter33` opponent. The run completed **1,940 games**, yielding **289,034 learner positions**. A copy of Generation 36 was then trained on those examples and saved as Generation 37.

During the first 60 engine-loop steps (including chance events and forced actions), learner moves were sampled from the search policy. Later they used search's returned move. Forced moves require no policy-learning example, and the frozen opponent's decisions are not learner targets.

A **generation** is one data-generation/training iteration.

### 4. Train

We train the GNN on the recorded observations and targets using AdamW.

Each recorded position supplies three learning signals:

- **Policy:** match search's action distribution $\pi$ using cross-entropy.
- **Value:** predict $y_v=0.35z+0.65v_{\mathrm{search}}$, where $z$ is the eventual result ($+1$ for a win, $-1$ for a loss, $0$ for a draw), and both terms use the acting player's perspective.
- **Future ownership:** predict the eventual owner/building type at each intersection and road location. These auxiliary heads provide local supervision for the shared representations.

We chose the Generation 37 objective to be

$$
\mathcal L=
\underbrace{-\sum_{a\in\mathcal A(s)}\pi(a\mid s)\log p_\theta(a\mid s)}_{\text{policy}}
+\underbrace{(v_\theta(s)-y_v)^2}_{\text{value}}
+0.15\mathcal L_{\mathrm{ownership}}.
$$

The policy and value terms teach the model what to play and how promising a position is. Ownership prediction adds a more detailed learning signal about where each player eventually builds. Increasing the 0.15 weight gives this auxiliary task more influence over the shared GNN, while lowering it puts relatively more emphasis on policy and value.

Because the value target includes the previous evaluator's search estimates, it can inherit their errors. Generation 37 applied no time discount to outcomes.

<details>
<summary>Training configuration and auxiliary losses</summary>

The auxiliary intersection classes are empty, my settlement, my city, their settlement, and their city. Road classes are empty, mine, and theirs. The ownership loss sums the mean retained intersection and road losses, excludes existing cities and built roads, and weights nonempty classes four times as strongly as empty ones.

| Setting | Recorded configuration |
|---|---|
| Initialization | Generation 36 weights; fresh AdamW optimizer |
| Requested / completed games | 2,000 / 1,940 |
| Recorded learner positions | 289,034 |
| Opponent sampling | 75% mirror, 25% frozen `exploiter33` |
| Search budget | 1,024 total simulations; up to four sampled worlds |
| Root exploration | Gumbel noise on; up to eight candidates per world |
| Training data | Only this run's new games |
| Epochs / batch size | 1 / 512 |
| Learning rate / weight decay | $5\times10^{-4}$ / $10^{-4}$ |
| Policy-target temperature | 1; no additional softening |
| Value target | 35% outcome, 65% search value; no time discount |
| Auxiliary weight | 0.15; nonempty-class weight 4 |
| Gradient norm limit | 1.0 |
| Resignation | Disabled |

The learner's seat is randomized against the frozen opponent. Training applies one of six valid board relabellings to each example's features and targets.

Games capped at **500 player turns or 50,000 engine steps** are scored by total victory points, including hidden cards. Tied scores produce a draw.

</details>

<details>
<summary>Training history</summary>

Training began with randomly initialized Generation 0 weights.

Earlier generations trained with search that could access referee-hidden information. Generations 36 and 37 used corrected search while retaining weights inherited from those earlier runs.

</details>

## Model Behavior and Interpretability Experiments

As said before, the part of this project I care about most is how we can understand these results; theoretically if we had a GTO (Game theory optimal) model for Catan I'd want to use it to figure out how we should think about different decisions. One big family of these decisions is opening theory. When I look at a board, I am usually thinking about a possible route to winning (win condition, if you will): cities and development cards, fast expansion, or a resource-heavy opening that becomes useful with a port. I want to know whether the bot has learned anything resembling those ideas, and where its preferences disagree with mine.

There are a few different questions hiding inside that goal. We can count who wins, check whether the model recognizes good positions, and describe the openings it prefers. Explaining *which internal computations produce those preferences* takes another step: changing a representation and testing whether the decision changes. The experiments below move from recorded behavior to paired changes in legal positions, continuation games, and tests of the internal representations.

For the opening study, A is the first player and B is the second. Placements follow

$$
\mathrm{A1}\;\longrightarrow\;\mathrm{B1}\;\longrightarrow\;\mathrm{B2}\;\longrightarrow\;\mathrm{A2}.
$$

Experiments 1–2 use recorded games. Experiments 3–7 analyze **100,000 random boards** with Generation 37’s weights fixed and search disabled. Experiments 8–13 add controlled comparisons and **6,144 complete continuation games**, also with the weights fixed and search disabled. Experiment 14 adds exact opening-production calculations, external opening trees, and controlled changes to distant tiles. Experiment 15 compares learned openings with simple production rules on 10,000 fresh boards, measuring who actually wins. Experiment 16 reuses their self-play controls to ask which board features favor A.

### Experiment 1: Is it better to go first or second?

This is the single experiment that motivated me to go down this rabbithole, since often me and my friends would complain about getting "unlucky" (whether that meant having to go first or second). A gets first choice of any corner on the board, but then has to wait while B places twice. B can choose a pair of settlements together; A has to leave something for later and hope it's still available. I wanted to know which advantage matters more in games played by the bot.

The most direct starting point is to count completed games where the same model played both seats. With no draws, the observed first-player win rate is

$$
\widehat p_A=\frac{W_A}{W_A+W_B}.
$$

Here $W_A$ and $W_B$ count wins by the first and second player. Games against a different frozen opponent are excluded, since a difference in playing strength could otherwise look like a seat advantage.

| Self-play policy | Completed games | A wins | A win rate (95% interval) |
|:---|---:|---:|---:|
| Generation 36 | 1,469 | 798 | **54.3%** (51.8–56.9%) |
| Adapted Generation 37 continuation | 574 | 311 | **54.2%** (50.1–58.2%) |
| Generation 37 (network only) | 10,000 | 5,475 | **54.8%** (53.8–55.7%) |

*The first two cohorts used 1,024-simulation search, training exploration, and no resignation. The continuation used a Generation 37 copy adapted to the later discard encoding, with newly activated input weights set to zero; collection stopped at 97% completion per shard. The third cohort uses the original Generation 37 network without search or exploration: all 10,000 unique control games from Experiment 15 finished. Intervals are 95% Wilson intervals.*

All three cohorts favor A by roughly the same amount, including the larger network-only sample. That is evidence of a modest first-player advantage under these policies. It still leaves an interesting question open: is the advantage coming from the opening placements, or from getting the first turn afterward? These games combine both effects.

On **those same 10,000 boards**, the value head predicts an average **50.06%** chance of A winning before A1, compared with **54.75%** observed. It underestimates A's advantage by **4.69 percentage points** (95% interval: 3.71–5.68). Experiment 2 checks its predictions during play, and Experiment 16 asks whether particular board features explain the seat advantage.

<details>
<summary>Opening predictions matched to game outcomes</summary>

Before A1, Generation 37's mean raw value is **+0.0012** and its median is **+0.0036**. The middle 90% of implied winning probabilities fall between **47.24% and 52.39%**.

![Opening forecasts on the same 10,000 boards used for self-play: mean implied A win probability 50.06%, versus 54.75% actual A wins.](/figures/catan-value-calibration-10k/opening-predictions.svg)

The forecasts also only weakly distinguish boards where A eventually wins from those where B wins: **AUC = 0.516**. The average prediction–outcome gap uses 10,000 bootstrap resamples of whole boards.

[Matched opening statistics](/figures/catan-value-calibration-10k/opening-summary.json)

</details>

### Experiment 2: Is the evaluator correctly calibrated?

When the evaluator assigns a player an 80% chance of winning, does that player actually win about 80% of the time?

The value head outputs $v=\tanh(u)$. If a win is $+1$ and a loss is $-1$, then

$$
\mathbb{E}[z]=p_{\mathrm{win}}-(1-p_{\mathrm{win}})=2p_{\mathrm{win}}-1.
$$

That gives the conversion

$$
p_{\mathrm{implied}}=\frac{v+1}{2}=\sigma(2u).
$$

A value of $0.6$ maps to 80%. Generation 37 was trained on the blended target $0.35z+0.65v_{\mathrm{search}}$, so calibration needs to be checked empirically.

We checked **160,000 positions from 10,000 games**, sampling 16 post-roll decisions throughout each game. We compare predictions with **actual wins under frozen, greedy Generation 37 play on both sides**. Changing either player's policy can change the winning probability.

Each point below groups predictions into a ten-percentage-point band: the horizontal coordinate is the mean prediction, and the vertical coordinate is the observed win rate. Accurate probabilities would lie near the diagonal.

![Generation 37 implied winning probabilities versus observed outcomes across 160,000 positions from 10,000 games, with whole-game 95% intervals.](/figures/catan-value-calibration-10k/post-roll-calibration.svg)

Across the ten bands, the weighted mean absolute prediction–outcome gap is **1.82 percentage points**. The evaluator distinguishes favorable from unfavorable positions: its Brier score—the mean squared probability error—is **0.1486**, compared with **0.2498** for always predicting the sample's overall win rate.

Equivalently, the outcome $R^2$ is **0.405**: a 40.5% reduction in squared error relative to that constant prediction. The broad agreement during play coexists with the opening bias in Experiment 1.

<details>
<summary>Sampling and prediction metrics</summary>

We reused the 10,000 unique network-versus-network games from Experiment 15. For each turn, we kept the first regular post-roll decision with more than one legal action, then selected 16 positions evenly spaced through that game's eligible-turn list. Every game supplied all 16. These positions span early through late play, with no requirement that the opponent hold a development card.

Predictions use the acting player's observation and the original Generation 37 checkpoint, without search or recalibration. Labels record whether that player eventually won. This tests forecasts under greedy continuation, while the model's training targets came from searched self-play.

Each game has total weight one, shared equally among its sampled positions. For outcome $y_i$ and implied probability $p_i$, the Brier score is $\sum_i w_i(p_i-y_i)^2/\sum_i w_i$, and outcome $R^2$ is

$$
1-\frac{\sum_i w_i(p_i-y_i)^2}{\sum_i w_i(y_i-\bar y)^2}.
$$

The reference $\bar y$ is the weighted sample win rate. Intervals use 5,000 bootstrap resamples of whole games, preserving dependence between positions from the same game. The outcome $R^2$ has a 95% interval of **0.396–0.415**. The ten-band gap weights each band's absolute mean prediction–outcome difference by its share of the sampled positions.

[Full statistics](/figures/catan-value-calibration-10k/post-roll-summary.json) · [Plotted data](/figures/catan-value-calibration-10k/calibration-bands.csv) · [Protocol](/figures/catan-value-calibration-10k/post-roll-protocol.json) · [Data and analysis code](/figures/catan-value-calibration-10k/data-and-methods.zip)

</details>

<details>
<summary>Five positions to compare with your own intuition</summary>

These five positions span a substantial deficit through a win available on the next move. Each shows the same value head’s output; $p=(v+1)/2$ gives its implied winning probability. The positions are illustrations selected for their range of evaluations, rather than a representative accuracy sample.

<iframe src="/figures/catan-gen37-value-examples.html" title="Five real positions evaluated by the Generation 37 value head"></iframe>

Equal public scores need not imply an even position: the 7–7 example receives $v=+0.900$, or 95% on the implied-probability scale. The $+0.998$ example has a legal city upgrade that immediately wins.

</details>

### Experiment 3: Does the bot just care about total pip count?

It seems like a pretty naïve thing to just go "I'll place my settlements on whichever corners have the most net production." But maybe naïve is good: we should see what CatanRL thinks. Before looking for complicated internal explanations, I want to know how far this familiar idea gets us.

Start with three decisions from the opening gallery. In each, the alternative maximizes total pips among legal corners.

<iframe src="/figures/catan-opening-comparisons.html" title="Three Generation 37 opening choices compared with legal maximum-pip alternatives"></iframe>

On each of the 100,000 boards, the network chooses all four settlements and roads using its highest-scoring legal move. We save the scores of **every legal settlement candidate**, giving 400,000 decisions and about 19.26 million candidate evaluations. B2 and A2 are evaluated after the earlier placements have been made; occupied corners and corners excluded by the distance rule are absent.

We fit a separate linear approximation to the policy logits at each placement stage. If $b$ identifies a board, $c$ a legal corner, and $x_{bcr}$ a feature such as ore pips, the approximation is

$$
\widehat{\widetilde z}_{bc}=\sum_r\beta_r\widetilde x_{bcr}.
$$

A tilde means we subtract the average over legal corners at that decision. This keeps the comparison within the board: a corner’s five ore pips matter relative to what the other available corners offer. The coefficients are fitted on **80,000 boards** and evaluated on **20,000 held-out boards**.

We summarize the fit with

$$
R^2=1-\frac{\mathrm{MSE}_{\mathrm{fit}}}{\mathrm{MSE}_{\mathrm{equal}}}.
$$

Both errors average over legal corners first and then equally over test boards. The reference predicts the same score for every legal corner. Thus $R^2=0.86$ means 86% less squared error than that reference.

*Held-out score fit, separated by placement stage.*

| Features | A1 | B1 | B2 | A2 |
|:---|---:|---:|---:|---:|
| Total pips | 0.856 | 0.860 | 0.682 | 0.676 |
| Five resource pip counts | 0.869 | 0.874 | 0.724 | 0.685 |
| + Coverage, concentration, adjacent tiles | 0.874 | 0.880 | 0.834 | 0.838 |
| All regression features | 0.927 | 0.924 | 0.869 | 0.878 |

Total production explains much of the variation in policy scores, especially on first placements. On second placements, adding coverage and concentration helps substantially: whether a corner adds resources you lack becomes part of the question. The full regression adds ports, dice-number patterns, expansion/proximity features, resource-specific interactions, and starting-card counts (defined below).

Even then, exact top-choice agreement is only **58.3%, 41.5%, 55.4%, and 57.6%** for A1, B1, B2, and A2. Those close choices are promising examples to inspect, because a simple production rule has already explained much of the separation between strong and weak locations.

<details>
<summary>Are the R² gains significant? Paired t-statistics</summary>

Each cell gives **$\Delta R^2$ (paired $t$-statistic)** on the same **20,000 held-out boards**.

| Feature change | A1 | B1 | B2 | A2 |
|:---|---:|---:|---:|---:|
| Total pips → five resource weights | +0.0122 (+116.7) | +0.0142 (+112.4) | +0.0419 (+90.9) | +0.0092 (+35.3) |
| + Coverage, concentration, adjacent tiles | +0.0056 (+96.5) | +0.0058 (+93.7) | +0.1103 (+231.2) | +0.1533 (+241.8) |
| + Ports | +0.0130 (+196.2) | +0.0230 (+253.6) | +0.0088 (+133.8) | +0.0108 (+128.1) |
| + Dice-number patterns | +0.0371 (+423.8) | +0.0108 (+169.7) | +0.0169 (+150.8) | +0.0135 (+130.1) |
| + Expansion and settlement proximity | -0.000022 (-2.27) | +0.0058 (+80.4) | +0.0043 (+60.5) | +0.0103 (+92.5) |
| + Resource-specific context | +0.0029 (+73.2) | +0.0040 (+91.3) | +0.0049 (+64.4) | +0.0048 (+53.1) |
| Remaining context combined¹ | +0.0531 (+502.6) | +0.0437 (+328.4) | +0.0349 (+221.4) | +0.0394 (+205.4) |

¹ The last row compares the coverage model with all regression features, combining the four subsequent feature groups.

**Every positive change has Holm-adjusted $p<0.001$.** The small decrease from adding expansion/proximity at A1 has adjusted $p=0.0232$. For comparison, the A2 coverage gain is **0.1533**, with a pointwise 95% interval of **0.1522–0.1544**.

For regression $m$, let $e_{m,b}$ be its mean squared logit error over board $b$’s legal corners. The [paired test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) uses

$$
\begin{aligned}
d_b &= e_{\mathrm{reduced},b}-e_{\mathrm{full},b},\\
t &= \frac{\bar d}{s_d/\sqrt{20{,}000}},
\qquad
\Delta R^2=\frac{\bar d}{\bar q}.
\end{aligned}
$$

Here $s_d$ is the sample standard deviation of $d_b$, and $q_b$ is the board’s mean squared centered logit. Two-sided tests use 19,999 degrees of freedom and Holm correction across all 28 comparisons. Pointwise $\Delta R^2$ intervals use the paired ratio delta method.

[Comparisons and intervals (CSV)](/figures/catan-opening-significance/comparisons.csv), [results (JSON)](/figures/catan-opening-significance/results.json), and [paired board errors and reproduction code (ZIP)](/figures/catan-opening-significance/reproduction.zip).

</details>

<details>
<summary>Full regression results</summary>

| Features (held-out $R^2$) | A1 | B1 | B2 | A2 |
|:---|---:|---:|---:|---:|
| Total pips | 0.856 | 0.860 | 0.682 | 0.676 |
| Five resource pip counts | 0.869 | 0.874 | 0.724 | 0.685 |
| + Coverage, concentration, adjacent tiles | 0.874 | 0.880 | 0.834 | 0.838 |
| + Ports | 0.887 | 0.903 | 0.843 | 0.849 |
| + Dice-number patterns | 0.924 | 0.914 | 0.860 | 0.863 |
| + Expansion and settlement proximity | 0.924 | 0.920 | 0.864 | 0.873 |
| + Resource-specific context | 0.927 | 0.924 | 0.869 | 0.878 |

Rows add feature groups cumulatively after the five-resource model. Total pips is a separate model with one shared resource weight. Production variance depends nonlinearly on total production, so the dice-number gain includes more than diversification.

**Each resource alone.** Each cell is the held-out $R^2$ of a one-feature regression using that resource’s local pips.

| Placement | Wood | Brick | Sheep | Wheat | Ore |
|:---|---:|---:|---:|---:|---:|
| A1 | 0.065 | 0.102 | 0.062 | 0.108 | 0.104 |
| B1 | 0.053 | 0.071 | 0.097 | 0.108 | 0.097 |
| B2 | 0.090 | 0.141 | 0.019 | 0.050 | 0.065 |
| A2 | 0.069 | 0.068 | 0.063 | 0.058 | 0.048 |

**Removing one resource.** Each cell is the loss in held-out $R^2$ when that resource’s pip count is removed from the five-resource model. Total pips and derived coverage features are excluded so they cannot reconstruct it.

| Placement | Wood | Brick | Sheep | Wheat | Ore |
|:---|---:|---:|---:|---:|---:|
| A1 | 0.284 | 0.318 | 0.280 | 0.351 | 0.321 |
| B1 | 0.273 | 0.286 | 0.343 | 0.371 | 0.332 |
| B2 | 0.295 | 0.354 | 0.176 | 0.243 | 0.253 |
| A2 | 0.287 | 0.271 | 0.264 | 0.276 | 0.237 |

**Fitting and uncertainty.** Each training board has equal weight. Features are scaled by their training-set weighted root-mean-square values and fitted with ridge penalty $0.001\sum_r\gamma_r^2$ in scaled units. Constant and exactly redundant columns are removed using training data. All four decisions from a board stay in the same split.

Coefficient intervals use a board-cluster sandwich calculation that includes uncertainty in feature scales. Test $R^2$ intervals use the whole-board delta method, conditional on the fitted regression.

</details>

<details>
<summary>What is included in each context feature group?</summary>

**Coverage, concentration, and adjacent tiles (six added features)**

- **Adjacent tiles:** number touching the candidate, including desert. This distinguishes coastal and interior geometry.
- **Local resource types:** number of distinct producing resources at the candidate corner.
- **New resource types:** number of resources the new settlement adds that the player’s existing settlement does not produce.
- **New-resource pips:** total candidate pips in those previously missing resources.
- **All five covered:** 1 if the candidate plus the player’s existing settlement covers every resource; otherwise 0.
- **Combined concentration:** $\sum_r s_r^2$, where $s_r$ is resource $r$’s share of the player’s combined pips. It is 1 for production entirely in one resource and 0.2 for an equal five-way split.


**Port features (six added features)**

- **Direct port:** 1 if the candidate touches any port.
- **Generic port:** an additional 1 if it is a 3:1 port. A generic port activates both this feature and direct port.
- **Matching production at a direct port:** the player’s combined pips in that port’s resource, or 0 unless the candidate touches a resource-specific 2:1 port.
- **Nearest port distance:** the smallest number of board edges from the candidate to any port corner.
- **Nearest generic-port distance:** the same distance restricted to 3:1 ports.
- **Matching-port potential:** the maximum, over resources, of combined resource pips divided by $(1+\text{distance to that resource’s port})$. A port the player already owns uses distance zero. This is a hand-designed approximation, not an estimate of trade profit.

The distances ignore road ownership, blocking, direction, building costs, and whether the opponent reaches the port first.


**Dice-number features (four added features)**

- **Distinct local numbers:** number of different dice outcomes producing resources at this corner.
- **Distinct combined numbers:** number of different producing outcomes across the player’s existing settlement and the candidate.
- **Overlapping-number pips:** candidate pips on numbers the player already collects. Two new tiles with the same number count their production separately.
- **Production variance:** variance of the total number of cards produced per dice roll across the player’s combined settlements. The calculation uses the dice probabilities and accounts for tiles producing together; it ignores robber blocking and bank shortages.

For example, wood on 6 and brick on 6 arrive together, while wood on 6 and brick on 8 arrive on different rolls. Their expected production is equal, but the distribution of total cards per roll differs. These features describe that distinction; they do not by themselves identify what risk preference the network has learned.


**Expansion and settlement proximity (three added features)**

- **Best production two edges away:** the largest pip total of a corner exactly two board edges from the candidate that currently satisfies the settlement-distance rule relative to existing pieces. It does not verify a buildable route, affordability, road direction, or who gets there first.
- **Opponent proximity:** $1/(1+d)$, where $d$ is the edge distance to the nearest opponent settlement; 0 if none exists yet.
- **Own proximity:** the same formula for the player’s existing settlement; 0 on their first placement.

**Resource-specific context** adds separate terms for each resource: candidate pips multiplied by existing production, by an indicator that the resource is missing, and by ownership of its matching 2:1 port; resource-specific direct-port indicators; candidate pips weighted by geometric distance to that resource’s port; and starting-resource-card counts. These let a resource’s fitted weight depend on the player’s current position. The exact columns and every coefficient are in the [analysis report](/figures/catan-opening-study-100k/report.html).

Some contexts remain rare even with 100,000 boards. For example, only two B2 positions already own an ore port. Coefficients for such cases do not support general strategic conclusions.

</details>

<details>
<summary>Browse opening examples and internal score contributions</summary>

Browse **1,000 boards from the earlier study**, filtering for concentrated production near a matching port, missing resources, close decisions, or picks that give up pips. Open a board to compare the chosen corner with its runner-up and inspect contributions to their logit difference.

<iframe src="/figures/catan-gen37-opening-gallery.html" title="Separate gallery of 1,000 Generation 37 openings with A1 B1 B2 A2 choices and internal score contributions"></iframe>

[Gallery data](/figures/catan-gen37-opening-study.zip). The regression tables in this article use the newer 100,000-board sample.

</details>

### Experiment 4: How much is an ore pip worth?

This might sound like a weird question to have, but one thing you will quickly pick up if you play 1v1 Colonist is that certain resources are just worth a little more... for example ore (for cities, dev card control. etc.). My starting guess was that ore and wheat should be especially valuable, with sheep closer to the bottom. But a ranking is not quite the question a player faces. Would I give up four sheep pips to gain three ore pip? Does the answer change between the first and second settlement?

The five-resource regression gives us one way to put a number on that tradeoff. **The unit is a pip of production, not a resource card in hand.** Ignoring robber blocking and bank shortages, one pip contributes $1/36$ of an expected card per dice roll from a settlement.

For two candidate corners, the fitted score difference is

$$
\widehat{\Delta z}=\sum_r\beta_r\Delta P_r,
$$

where $\Delta P_r$ is their difference in pips for resource $r$. Suppose we gain ore production and give up sheep production, leaving the other pip counts unchanged. To keep the fitted score equal,

$$
\beta_{\mathrm{ore}}\Delta P_{\mathrm{ore}}
+\beta_{\mathrm{sheep}}\Delta P_{\mathrm{sheep}}=0.
$$

Solving for the amount of sheep production gives

$$
-\frac{\Delta P_{\mathrm{sheep}}}{\Delta P_{\mathrm{ore}}}
=\frac{\beta_{\mathrm{ore}}}{\beta_{\mathrm{sheep}}}
=\frac{2.119}{1.763}\approx1.20\qquad(\mathrm{A1}).
$$

In this approximation, **one additional ore pip contributes about as much to the A1 preference score as 1.20 sheep pips**. Dividing each resource’s coefficient by the sheep coefficient gives the same comparison for all five:

*A1 production-pip equivalents in the five-resource approximation.*

| Resource | Fitted score per pip | Equivalent sheep pips |
|:---|---:|---:|
| Wood | 1.775 | 1.01 |
| Brick | 2.111 | 1.20 |
| Sheep | 1.763 | 1.00 |
| Wheat | 1.977 | 1.12 |
| Ore | 2.119 | 1.20 |

Ore and brick are almost tied here. Wheat is somewhat lower, while wood and sheep are close. That is already a useful complication of my initial ranking. It is also a different question from how much ore appears in the bot’s chosen settlements, which we measure next.

These are **fitted score equivalents**, not trading prices or equal chances of winning. The ratio describes a tradeoff inside a simplified equation; it does not establish that a legal board edit can exchange those pips while leaving ports, number patterns, and every other strategic property unchanged. The five-resource model’s A1 $R^2$ is 0.869, so the approximation is useful but incomplete.

The coefficients also change with placement stage: the ore/sheep ratio is **1.17 at B1, 1.43 at B2, and 1.18 at A2**. A single exchange rate for “the start of the game” would hide that variation. The later experiments examine two reasons to expect more variation within a stage: existing resource access and scarcity.

<details>
<summary>Resource equivalents across all four placements</summary>

Each cell is $\beta_r/\beta_{\mathrm{sheep}}$, in equivalent sheep pips.

| One pip of… | A1 | B1 | B2 | A2 |
|:---|---:|---:|---:|---:|
| Wood | 1.007 | 0.905 | 1.295 | 1.097 |
| Brick | 1.197 | 1.061 | 1.622 | 1.241 |
| Sheep | 1.000 | 1.000 | 1.000 | 1.000 |
| Wheat | 1.121 | 1.080 | 1.227 | 1.119 |
| Ore | 1.202 | 1.169 | 1.434 | 1.182 |

Pointwise 95% intervals are **1.200–1.204** for A1 ore/sheep and **1.430–1.438** for B2. The underlying coefficients are in logit units per pip:

| Placement | Wood | Brick | Sheep | Wheat | Ore |
|:---|---:|---:|---:|---:|---:|
| A1 | 1.775 | 2.111 | 1.763 | 1.977 | 2.119 |
| B1 | 1.509 | 1.769 | 1.667 | 1.801 | 1.949 |
| B2 | 1.923 | 2.408 | 1.485 | 1.822 | 2.130 |
| A2 | 1.976 | 2.236 | 1.801 | 2.015 | 2.130 |

Replacing exactly one sheep pip with one ore pip keeps total production fixed and changes the fitted A1 score by

$$
\Delta\widehat z=\beta_{\mathrm{ore}}-\beta_{\mathrm{sheep}}
\approx2.119-1.763=0.356.
$$

Ratio intervals use the log-ratio delta method and the five-coefficient covariance, reconstructed from coefficient standard errors and pairwise-difference intervals. [Derived ratios and uncertainty](/figures/catan-opening-study-100k/resource-ratios.json) and [calculation](/figures/catan-opening-study-100k/methods/derive_resource_ratios.py).

</details>

### Experiment 5: Which resources does it favor among comparable corners?

Counting the resources the bot picks is tempting, but the board does not offer equal amounts of everything. Ore and brick each occupy three tiles; the other resources occupy four. Number tokens change their production, and previous settlements remove some of the best corners from consideration.

We therefore compare each chosen corner with the legal alternatives that have **the same total pips and the same number of adjacent tiles**. This asks what mix of resources the bot takes once we match those two properties. The pool includes the chosen corner and must contain at least three candidates.

For resource $r$, let $s^{\mathrm{chosen}}_{br}$ be its share of the chosen corner’s pips. Let $s^{\mathrm{matched}}_{br}$ be the average share across that board’s matching alternatives. We report

$$
E_r=\frac{\operatorname{mean}_b\,s^{\mathrm{chosen}}_{br}}
{\operatorname{mean}_b\,s^{\mathrm{matched}}_{br}}.
$$

An $E_r$ of 1 means equal average shares. A value of 1.45 means the resource’s share is 45% larger in the chosen settlements. This is a description of the selected resource mix, separate from the pip tradeoff in Experiment 4.

*Chosen resource share divided by matched-alternative share.*

| Resource | A1 | B1 | B2 | A2 |
|:---|---:|---:|---:|---:|
| Wood | 0.80× | 0.65× | 1.09× | 0.97× |
| Brick | 1.19× | 0.88× | 1.38× | 1.20× |
| Sheep | 0.54× | 0.92× | 0.45× | 0.71× |
| Wheat | 1.13× | 1.22× | 1.04× | 1.11× |
| Ore | 1.45× | 1.41× | 1.11× | 1.16× |

*Eligible boards: A1 48,159; B1 48,513; B2 54,503; A2 49,554. Each board has equal weight. The [full report](/figures/catan-opening-study-100k/report.html) includes paired intervals and alternative baselines.*

Ore is the most overrepresented resource at A1 and B1; brick leads at B2 and A2. Wheat is above the matched baseline throughout. Sheep is below it, particularly at A1 and B2.

One possible story is that the bot secures production for cities and development cards early, then looks for expansion resources on its second settlement (since you get one resource per hex of the second settlement). That would be interesting opening theory, but the table alone does not demonstrate it. Starting cards, resource coverage, and the corners left by the opponent all change at the same time. The next experiment groups positions by one of those factors: what the player already produces.

### Experiment 6: Does the second settlement fill gaps or double down?

If my first settlement already has strong wood and brick, another wood–brick corner may be less useful than its production suggests. On the other hand, concentrated production can be a plan in its own right, especially with a port. I want to know whether the bot tends to fill gaps or commit more heavily to what it already has.

We split the matched comparison from Experiment 5 according to whether the player’s first settlement produces the resource. Each group gets its own chosen-share and matched-share averages, then the same ratio $E_r$. For A2, the results are:

*Resource-share ratios at A2, conditioned on production from A1.*

| Resource | Missing at A1 | Already produced at A1 |
|:---|---:|---:|
| Wood | 1.20× | 0.72× |
| Brick | 1.61× | 0.67× |
| Sheep | 0.92× | 0.32× |
| Wheat | 1.49× | 0.82× |
| Ore | 1.57× | 0.76× |

The contrast is large. Brick changes from **1.61×** the matched baseline when missing to **0.67×** when already produced; ore changes from **1.57× to 0.76×**. The same direction appears for all five resources at B2 as well.

Sheep is an interesting exception to a simple “collect all five” rule. Its share remains below baseline even when missing: **0.73× at B2 and 0.92× at A2**. I would like to inspect the openings where the bot accepts that gap. Does it expect to trade for sheep, or does another constraint dominate the choice?

This is evidence of context-dependent preferences and also aligns well with human intuition (in most cases you would want at least one hex of each resource if possible), but not an isolated effect of changing the first settlement. The groups have different boards and opening histories. A stronger test would hold the candidate corners fixed, vary the first settlement through legal histories, and then test which internal representations explain the ranking change.

<details>
<summary>Both second placements, with sample sizes and intervals</summary>

| Stage | Resource | Missing: n | Ratio [95% CI] | Produced: n | Ratio [95% CI] |
|:---|---:|---:|---:|---:|---:|
| B2 | Wood | 31,933 | 1.181 [1.174, 1.188] | 22,570 | 0.933 [0.923, 0.943] |
| B2 | Brick | 36,161 | 1.508 [1.499, 1.517] | 18,342 | 1.054 [1.039, 1.069] |
| B2 | Sheep | 18,759 | 0.727 [0.718, 0.737] | 35,744 | 0.267 [0.261, 0.273] |
| B2 | Wheat | 21,310 | 1.314 [1.304, 1.324] | 33,193 | 0.814 [0.805, 0.823] |
| B2 | Ore | 25,877 | 1.397 [1.384, 1.410] | 28,626 | 0.739 [0.726, 0.752] |
| A2 | Wood | 23,415 | 1.198 [1.188, 1.208] | 26,139 | 0.715 [0.705, 0.726] |
| A2 | Brick | 23,202 | 1.615 [1.602, 1.628] | 26,352 | 0.673 [0.660, 0.686] |
| A2 | Sheep | 28,784 | 0.923 [0.915, 0.931] | 20,770 | 0.323 [0.315, 0.332] |
| A2 | Wheat | 16,632 | 1.492 [1.481, 1.504] | 32,922 | 0.816 [0.806, 0.826] |
| A2 | Ore | 18,063 | 1.574 [1.559, 1.589] | 31,491 | 0.763 [0.751, 0.777] |

“Missing” means the existing settlement has zero production pips for that resource. Each resource partitions the eligible boards separately, with the same matching rule applied within each group.

</details>

### Experiment 7: Does scarcity change what a pip is worth?

Five ore pips might be less special on a board full of productive ore than on a board where most ore is weak. A local pip count cannot distinguish those situations. The network can see the whole board, so it is worth asking whether its preference reflects the supply elsewhere.

We refit the context model with an additional interaction for each resource. Let $S_r$ be its total production pips across the board and $S_r^{\mathrm{ref}}$ a reference level. The added term is

$$
\gamma_r P_r\bigl(S_r^{\mathrm{ref}}-S_r\bigr).
$$

When comparing corners on the same board, $S_r$ is fixed. The scarcity term therefore adds the following adjustment to the fitted weight per pip:

$$
\delta_r(\mathrm{board})=\gamma_r(S_r^{\mathrm{ref}}-S_r).
$$

A positive $\gamma_r$ means that lower board supply is associated with a larger fitted weight per pip.

*Held-out score fit before and after adding board-supply interactions.*

| Placement | Context $R^2$ | + Scarcity $R^2$ | Gain |
|:---|---:|---:|---:|
| A1 | 0.9274 | 0.9369 | +0.0096 |
| B1 | 0.9238 | 0.9306 | +0.0068 |
| B2 | 0.8691 | 0.8727 | +0.0036 |
| A2 | 0.8778 | 0.8795 | +0.0016 |

Ore has the largest positive scarcity coefficient at every placement stage, with wheat next. At A1, adding scarcity raises exact-choice agreement from **58.3% to 62.0%**. The effect on fit is smaller for the second settlements.

The result is more specific than a general preference for scarce resources. Wood has small negative fitted scarcity coefficients at A1, B1, and A2, and a positive one at B2. Scarcity helps describe this policy, but the table does not show that making ore scarce *causes* the network to value it more. Valid swaps of distant number tokens, followed by tracing and patching the changed representations, would test that explanation more directly.

<details>
<summary>Reference supply and legal availability</summary>

There are 58 pips across 18 producing tiles. For a resource occupying $n_r$ tiles, the reference supply is

$$
S_r^{\mathrm{ref}}=\frac{58n_r}{18}.
$$

That gives 9.67 pips for brick or ore and 12.89 for wood, sheep, or wheat.

![Board-scarcity coefficients by resource and placement stage, with individual and multiplicity-adjusted intervals.](/figures/catan-opening-study-100k/board-scarcity-slopes.svg)

Replacing board supply with mean resource pips among legal corners gives held-out $R^2$ values of **0.937, 0.931, 0.871, and 0.878** for A1, B1, B2, and A2. At A1 every corner is legal, making the two supply measures proportional and their fits identical.

The [full report](/figures/catan-opening-study-100k/report.html) includes paired $R^2$-change intervals and individual and Bonferroni-adjusted intervals across the 40 scarcity coefficients.

</details>

[Full report](/figures/catan-opening-study-100k/report.html), [results](/figures/catan-opening-study-100k/results.json), [coefficients](/figures/catan-opening-study-100k/coefficients.csv), [model comparisons](/figures/catan-opening-study-100k/models.csv), and [resource-share comparisons](/figures/catan-opening-study-100k/enrichment.csv).

The next experiments test some of those explanations directly. We screen the same 100,000 boards, then sample eligible paired positions for fresh evaluations. Every changed game position is replayed through the real rules engine and encoder. For the internal tests, we replace selected activations between two positions and measure the output change.

Each experiment reports its own sample size. Intervals treat whole boards as the independent units and are exploratory, pointwise intervals. They measure sampling uncertainty for this checkpoint and these comparisons, rather than uncertainty across independently trained models.

### Experiment 8: Does the opening plan end up being the win condition?

When I start on ore and wheat, I am usually imagining cities or development cards. A wood–brick opening suggests expansion. Do those different starts actually lead the bot down different paths?

We found **20,469 boards** with two legal A2 choices that offer a useful comparison. Both choices have the same total pips, sheep pips, number of adjacent tiles, local resource diversity and combined resource coverage with A1. Neither directly owns a port. One shifts at least three pips from wood/brick into ore/wheat. A1, B1 and B2 remain fixed.

We sampled **2,048 board pairs**, forced each alternative, and let the frozen network choose every subsequent move for both players. All **4,096 games** finished. By global turn 40 (about 20 turns per player), the two openings had produced visibly different behavior:

*Mean additional building or buying actions by A through turn 40; setup pieces are excluded.*

![Wood and brick openings produce more roads and settlements; ore and wheat openings produce more cities and development cards. Mean actions by player A through turn 40 across 2,048 paired boards.](/figures/catan-opening-building-paths.svg)

The direction agrees with the familiar human distinction: the ore–wheat side builds upward and buys cards; the wood–brick side expands. For example, **75.0%** of the ore–wheat openings build at least one city by turn 40, compared with **61.3%** of the wood–brick openings.

A also won **37.01%** of the ore–wheat games and **34.08%** of the wood–brick games: a paired difference of **+2.93 percentage points** [0.36, 5.50] for these selected, forced alternatives.

This establishes a relationship between opening composition and later play. It does not yet show that the network commits to an abstract plan. Different resources change which actions become affordable, and a policy that responds sensibly to each current hand could produce this pattern without representing a long-term strategy explicitly.

<details>
<summary>Building counts, paired games and the global-representation test</summary>

| Action | More wood/brick | More ore/wheat | Paired difference (95% interval) |
|:--|--:|--:|--:|
| Roads | 2.70 | 2.02 | −0.68 [−0.76, −0.61] |
| Settlements | 1.35 | 1.05 | −0.31 [−0.35, −0.26] |
| Cities | 0.89 | 1.23 | +0.34 [0.29, 0.38] |
| Development cards | 0.89 | 1.59 | +0.69 [0.63, 0.76] |

Each pair starts with the same random-number state, though later draws and thefts can desynchronize the streams. Both players use greedy Generation 37 without search or resignation. Roads include free roads from development cards. The alternatives can differ in location, number identities, starting cards and A2 road, so the comparison changes the complete opening.

We saved each game's first A decision in ordinary play with multiple legal actions. In **331 pairs**, the snapshots occur on the same turn and share at least two legal actions. We transplant the ore–wheat position's global representation into the wood–brick position to test whether it carries opening-dependent information into the later decision.

Let $\Delta z_b$ be the donor-minus-target score change, centered over common legal actions, and $\Delta z_b^{\mathrm{patch}}$ the change from the transplant. Their pooled directional projection is

$$
T=\frac{\sum_b\langle\Delta z_b^{\mathrm{patch}},\Delta z_b\rangle}
{\sum_b\|\Delta z_b\|^2}.
$$

After the first graph round, $T=0.146$ [0.108, 0.184], with similar values at the other tested boundaries (0.137–0.145). The global representation carries some of the score difference, including hand, production and other context. This projection can fall outside [0, 1].

[Results](/figures/catan-opening-followups/plans/results.json) and [data and reproduction scripts](/figures/catan-opening-followups/data-and-methods.zip).

</details>

### Experiment 9: Does a matching port make specialization more attractive?

A wood-heavy opening feels different when it comes with a wood port. I expected the model to notice that combination: if more of my income is wood, turning two wood into anything should become more useful. The result was less straightforward than that.

I screened all 100,000 opening boards, then sampled **2,000 eligible B2 positions and 2,000 eligible A2 positions**. At each position, I kept the proposed second settlement and its local tiles fixed. A same-number resource swap increased the share of one resource in the player's combined production, from about **39% to 63%** on average. A separate port-label swap changed the port at that settlement. These were valid boards with the same inventories, legal placements, and starting hands.

<iframe src="/figures/catan-port-patch.html" title="A controlled port comparison and the target-vertex embedding patch"></iframe>

For a target corner, let $s$ be its logit minus the average logit of unaffected legal comparison corners. The interaction is:

$$
\begin{aligned}
B(p)&=s(p,\text{matching})-s(p,\text{comparison}),\\
I&=B(\text{concentrated})-B(\text{less concentrated}).
\end{aligned}
$$

A positive result means concentrating production strengthens the matching port's bonus. I compared the matching 2:1 port against both a generic 3:1 port and a 2:1 port for a resource the player would not produce at either settlement.

*Change in the matching port's relative score when production becomes more concentrated; 95% intervals resample boards.*

| Comparison port | B2 interaction | A2 interaction |
| :-- | --: | --: |
| Generic 3:1 | +0.086 [0.073, 0.099] | +0.099 [0.083, 0.115] |
| 2:1 for an unproduced resource | −0.040 [−0.054, −0.028] | −0.041 [−0.057, −0.024] |

**The model responds to the combination, but it does not give a consistent specialization bonus.** Concentration makes a matching 2:1 port a little more attractive relative to a generic port, yet the generic port still scores substantially higher: even in the concentrated condition, changing 3:1 to matching 2:1 lowers the target's relative score by **1.057 logits at B2 and 0.874 at A2**. Against an equally priced 2:1 port for an unproduced resource, the matching port does better—but its advantage *shrinks* as production becomes more concentrated.

On **128 random positions per stage**, restoring the target-port vertex embeddings to their comparison-port values removed almost all of the interaction averaged across stages: **+0.112 to +0.004 logits** for the generic-port comparison, and **−0.073 to −0.004** for the same-rate comparison. Restoring only the reciprocal remote port left most of the effect, locating the main input pathway at the target port.

This test concerns **immediate port ownership**. Reaching a nearby port through later expansion remains untested.

<details>
<summary>Controlled resource and port swaps</summary>

The resource swap exchanges same-number hexes: one touches only the first settlement, and the other touches no placed settlement. Neither touches the target or comparison corners. This preserves aggregate resource pips, number patterns, total portfolio pips, starting hands and owned ports.

Port labels are exchanged at unoccupied locations, preserving the inventory. Comparison corners are legal non-port corners with unchanged local tiles. Subtracting their mean logit measures the target's relative preference.

The same-rate control includes **1,954 B2 and 1,957 A2 positions**. Its comparator resource is absent from both production portfolios. This diagnostic was added after inspecting the planned generic-port comparison. Intervals use 2,000 board-bootstrap replicates.

[Full results](/figures/catan-opening-followups/ports_numbers/results.json) and [data and reproduction scripts](/figures/catan-opening-followups/data-and-methods.zip).

</details>

### Experiment 10: How much does the starting hand matter?

An opening can have good production and still take a while to get moving. I wanted to separate the cards that arrive immediately from the cards we hope to roll later.

There is a useful legal counterfactual here: **swap A1 and A2, carrying each settlement's road with it**. The final pieces and future production stay exactly the same, but A receives starting resources from the other settlement. We sampled 2,000 boards where this changes the resource mix without changing the number of starting cards. Both histories were replayed through the actual game engine.

After a shared first roll, sampled from the usual dice frequencies conditional on not rolling seven, the bot chose a different action **10.3% of the time**. Different cards make different actions affordable. When we restrict both policies to the moves legal in *both* positions, the preferred action changes in only **0.4%** of pairs.

| Measurement | Result (95% interval) |
|:--|--:|
| Different first post-roll action | 10.3% (9.0–11.7%) |
| Different favorite among common legal actions | 0.4% (0.2–0.8%) |
| Mean absolute change in pre-roll value | 0.0699 (0.0675–0.0722) |

*2,000 independent boards; paired histories have identical pieces and equal-sized starting hands.*

The immediate effect is mostly that a hand opens or closes building options. The value head still distinguishes the hands even before any dice are rolled, when the only legal action is to roll. On average, it rates the original order 0.0168 raw value units above the reversed order. That is a preference of this trained evaluator, not a measured improvement in win rate.

The hand difference enters through global features. To test whether that signal remains in the global pathway after message passing, we transplanted the global representation after the first graph round in **512 pairs**. Its directional projection onto the full value change was **1.105 [1.054, 1.156]**, slightly overshooting it.

[Full results](/figures/catan-opening-followups/hands_roads/results.json) and [data and reproduction scripts](/figures/catan-opening-followups/data-and-methods.zip).

### Experiment 11: Does the bot prefer steady income or useful bursts?

Two settlements can have identical expected income and still play differently. Collecting wood and brick together might let me build a road immediately. Collecting on more numbers might make my income steadier. Does the model systematically prefer either pattern?

I tested **2,000 eligible B2 positions and 2,000 eligible A2 positions**, screened from the same 100,000-board corpus. I swapped a number next to the player's first settlement with a remote number on the **same resource and with the same pip count**—for example, 6 and 8. Every swap preserved the number inventory and the rules against adjacent equal or red numbers. The candidate second settlements' own tiles did not change, and neither did any resource's expected production or either player's starting hand.

The swap makes some second-settlement candidates stop sharing a number with the first settlement, while making others start sharing it. I measured the difference between those groups' changes in policy score:

$$
D=\operatorname{mean}_{c\in C_{\text{becomes spread}}}(z'_c-z_c)
-\operatorname{mean}_{c\in C_{\text{becomes synchronized}}}(z'_c-z_c).
$$

Positive $D$ would indicate a shift toward spreading income across rolls; negative $D$ would indicate a shift toward synchronized income.

*Mean relative logit change; 95% intervals resample boards.*

| Second settlement | Mean $D$ | 95% interval | Boards with $D>0$ |
| :-- | --: | --: | --: |
| B2 | −0.0004 | [−0.0029, 0.0021] | 50.5% |
| A2 | −0.0018 | [−0.0046, 0.0008] | 49.9% |

**There is no clear average preference for spreading or synchronizing income in this test.** Individual positions move in both directions, with median absolute responses of **0.030 logits at B2 and 0.036 at A2**.

<details>
<summary>Production variance and controls</summary>

Each comparison corner has exactly one tile on an exchanged number, and the first settlement has no other tile on either number. If that number has $w$ pips, changing the candidate from synchronized to spread production reduces total-card variance per roll by

$$
\Delta\operatorname{Var}(\text{cards per roll})=-\frac{2w}{36},
$$

with expected production unchanged for every resource. This calculation excludes robber blockage and bank shortages. Opponent production and the global input vector stay fixed.

The swap also changes number identities and the remote hex's expansion context. Earlier placements remain fixed throughout the comparison.

[Full results](/figures/catan-opening-followups/ports_numbers/results.json) and [data and reproduction scripts](/figures/catan-opening-followups/data-and-methods.zip).

</details>

### Experiment 12: Does it account for the sites it denies the opponent?

A settlement can be useful because of what it produces, or because of what it takes away. B2 is a particularly clean place to look for the second effect: A1 and B1 are already fixed, and A has exactly one opening settlement left to place.

For each legal B2 corner, we measured how it changes A2's remaining options. One simple measure is the loss in A's best available production:

$$
D(c)=\max_{d\in L}P(d)-\max_{d\in L\setminus N[c]}P(d),
$$

where $L$ is the set of currently legal corners, $P(d)$ is a corner's total pips, and $N[c]$ contains $c$ and its neighboring intersections.

Across the 100,000 boards, adding five measures of the sites denied improved held-out prediction of B2's policy scores:

| Approximation of B2's scores | Held-out $R^2$ | Exact choice agreement |
| :--- | ---: | ---: |
| Existing resource, portfolio, port, number and scarcity features | 0.8727 | 55.62% |
| Those features + effects on A2's available sites | 0.8891 | 58.66% |

*The added features explain another 0.0164 of the score variance (95% interval: 0.0162–0.0166). Fits use 80,000 boards; evaluation uses the other 20,000.*

There is also a more direct test: **does choosing the corner that denies more production actually help B win?** We found 1,315 held-out boards with a useful comparison and sampled 1,024 of them. Within each pair, the two B2 choices had exactly the same production pips for every resource, the same starting resource cards, the same number of adjacent hexes, and the same direct-port type. They differed by at least one pip in how much they reduced A's best remaining production option.

We forced each choice, let the network finish the opening, and played both games to completion with the frozen policy on both seats. All 2,048 games finished.

| Forced B2 choice | B wins |
| :--- | ---: |
| Denies more of A2's best production | 31.35% |
| Denies less | 29.39% |

The difference is **+1.95 percentage points**, with a paired 95% interval of **−1.46 to +5.37**. This does not establish a winning advantage. These are selected, forced alternatives, so their absolute win rates do not describe normal self-play. Geometry, number patterns and future road opportunities still differ between the corners.

The evidence is stronger that the policy responds to opposing pieces. In a separate test on 2,000 boards, we moved A1 and its road while keeping the board and B1 fixed. A1's total pips and coastal/interior class stayed the same. Even when compared only over corners legal in both positions, B's preferred B2 changed **20.95%** of the time. Patching the input embeddings at the moved opponent settlement and road locations transferred part of that score change.

<details>
<summary>Denial features and opponent-relocation controls</summary>

The five added features measure:

- The number and total pips of other corners removed by the placement.
- The drop in A's best remaining pip total.
- The drop in the best resource coverage achievable by A1+A2.
- The number of upper-quartile production sites removed.

The baseline already includes the candidate's production, portfolio, starting cards, ports, number patterns, expansion context and resource scarcity.

On **7,482 boards** with another legal corner matching the chosen B2's five-resource pip vector, the chosen corner denied **0.058 more pips** than a uniform matched choice [0.051, 0.064]. The matching pool includes the chosen corner.

In the relocation test, **171 of 2,000 pairs** also matched A1's exact resource vector. B2's favorite changed in **13.45%** of this subset [8.32, 18.58%]. Other pairs change A's resource mix as well as its position.

Patching post-ReLU embeddings at the moved settlement and road locations gave a directional projection of **0.143 [0.136, 0.150]**. We use Experiment 8’s formula with each board’s numerator and denominator divided by its number of common legal actions. These embeddings carry local features as well as ownership.

[Full results](/figures/catan-opening-followups/blocking_fallback/results.json), [paired continuation results](/figures/catan-opening-followups/blocking_fallback/continuations/results.json), and [opponent-relocation intervention](/figures/catan-opening-followups/blocking_fallback/opponent-relocation.json).

</details>

### Experiment 13: When is a contested road worth racing for?

By “wall gambling,” I mean two settlements **two edges apart**, with their opening roads pointing into the same empty junction. B has placed its inward road already. Should A point inward too, betting on being the first to extend out of that junction?

The junction has three edges. B occupies one; A would occupy the second. The race is for the **third branch**, leading away from both settlements. Whoever builds there first leaves the other inward road with no extension through that junction.

This is especially interesting at **A2**. B has placed both settlements and roads, so A can see the threat before choosing. A also takes the first turn after setup. If A already holds wood and brick, it can claim the exit on that turn without needing a favorable roll. Otherwise, pointing inward is a gamble on getting the missing resources before B does. The first roll produces for both players; A's advantage is the first opportunity to **spend** those resources.

<iframe src="/figures/catan-two-edge-wall-gambling.html" title="Two-edge wall gambling: the opening roads meet at a shared junction, then compete for its third branch"></iframe>

To test this geometry, we screened the same 100,000 boards and found **30,849 eligible positions**. We sampled **2,000 boards**, kept the settlements and starting cards fixed, and rotated B2's road inward versus outward before asking Generation 37 to choose A2's free road. Its legal choices were identical in both versions.

The average response hides an interesting split. These are A’s mean policy probabilities for the inward opening road; the change is paired within each board (95% intervals).

| A’s hand | Boards | B outward | B inward | Change (pp) |
| :--- | ---: | ---: | ---: | ---: |
| Wood + brick | 584 | 44.5% | 54.8% | **+10.3** [8.9, 11.7] |
| Missing either | 1,416 | 42.1% | 29.1% | **−13.0** [−14.1, −12.0] |

The greedy choice follows the same pattern: with wood and brick, the share choosing inward rises from **45.0% to 55.0%**; without that pair, it falls from **41.6% to 27.5%**.

The bot does not simply avoid an opponent's inward road. When A can already pay for the next road, it becomes *more* interested in meeting B at the junction. One possible explanation is that it recognizes an opportunity to trap B's road before B gets a turn. When A lacks those resources, the same geometry is usually a deterrent. These groups contain different positions, so their contrast does not by itself isolate the effect of giving A wood and brick.

A separate test changes **B's starting hand on the same board** by swapping B1 and B2, preserving B's final pieces and production. We crossed the two hands with both road orientations on **1,000 paired boards**, keeping A's cards fixed. The orientation effect is the change in A's inward-road probability when B points inward rather than away:

| A’s hand | Boards | B missing either | B has both |
| :--- | ---: | ---: | ---: |
| Wood + brick | 364 | +10.76 pp | +8.79 pp |
| Missing either | 636 | −3.75 pp | −20.68 pp |

Swapping to B's road-ready hand shifts the orientation effect by **−11.48 percentage points** on average (95% interval **−12.58 to −10.38**). The shift is largest when A cannot already build. The swap changes B's whole resource mix and associated bank counts.

This is evidence that the policy responds to the interaction between roads and starting cards. We have not yet measured who actually claims the exit, whether entering improves the chance of winning, or how the decision differs when the challenger is B. Those require continuation games and a separate second-player comparison.

<details>
<summary>Road geometry and paired comparisons</summary>

The geometry is $A2-u-B2$, with an empty third edge $(u,w)$. Both $u$ and $w$ are empty, and no earlier road touches them. B's outward alternative increases its distance from A2 while preserving its connection status with B1's road. All global inputs match across orientation pairs.

Within each board, we measure

$$
\Delta p=p(\text{A points inward}\mid\text{B points inward})
-p(\text{A points inward}\mid\text{B points away}).
$$

Across all 2,000 boards, the mean is **−6.21 percentage points** [−7.19, −5.24]. A's highest-scoring road changes on **15.5%** of boards. The split by A's starting cards compares different boards. Evaluations use the frozen policy without search, with paired intervals calculated across boards.

[Results](/figures/catan-two-edge-wall-gambling/results.json), [methods](/figures/catan-two-edge-wall-gambling/METHODS.md), and [data and scripts](/figures/catan-two-edge-wall-gambling/data-and-methods.zip).

</details>

### Experiment 14: Does the first pick leave a fallback?

When I choose A1, I am usually imagining a second settlement: “If I take the ore and wheat now, I can get wood and brick over there.” The problem is that B gets two picks before I come back. Does the network account for that, or does it mostly take the best-looking corner and work out A2 later?

Covering all five resources is one part of this question. The more interesting question is whether A1 sets up a strong **pair** of settlements against the replies B might actually make. A good plan could rely on one exceptional complement, several interchangeable backups, a port, or denying B a particularly strong opening.

The opening engine makes A1 with one forward pass. It does not run a tree search at decision time. Training with search could still teach it to anticipate consequences; the tests below ask what its choices are consistent with, rather than assuming it has learned a particular planning algorithm.

#### How much of the second pick survives?

We first made the fallback calculation exact on all **100,000 boards**. For each possible A1, we found its best A2 complement before B moved, then the best complement left under every legal B1/B2 pair. We scored the pair using either total pips or resource-weighted pips. The resource weights come from the earlier A1 regression; they are associations with policy scores, not exchange rates between cards.

$$
\begin{aligned}
O(a)&=\max_{d\in L(a)}U(a,d),\\
W(a)&=\min_{(b_1,b_2)\text{ legal after }a}
       \max_{d\in L(a,b_1,b_2)}U(a,d).
\end{aligned}
$$

Here $a$ is A1, $d$ is A2, and $L$ is the set of legal second settlements. $O$ assumes the ideal complement survives; $W$ asks what A can still obtain when B removes the most valuable options under the chosen production score $U$.

Across the recorded openings, a maximum-pip A2 complement survived B's actual placements only **42.35%** of the time. The best surviving corner had an average original rank of **2.17** (one plus the number of initially available corners with strictly more pips). Actual B placements removed **0.84 pips** from A's best available complement on average; the most damaging legal pair removed **1.51 pips**. So the difference between “my ideal second pick” and “what comes back to me” is substantial.

But agreement with a minimax calculation can be deceptive. On the 20,000 held-out boards:

| A1 selection rule | Maximizes worst-case pair pips | Maximizes worst-case resource-weighted pair pips |
| :--- | ---: | ---: |
| Generation 37 | 64.18% | 49.10% |
| Local-feature regression | 80.75% | 72.41% |
| Highest immediate pip count | 90.24% | 55.63% |
| Highest immediate resource-weighted production | 83.29% | 84.79% |

The local rules do not examine B's replies, yet agree with these narrow minimax objectives more often than the network does. This does **not** mean they play better Catan. It means that a high agreement rate, by itself, cannot establish planning—and production minimax is not a sufficient explanation of the network.

<details>
<summary>Do fallback features explain anything beyond local production?</summary>

They add a little. We predicted the network's within-board-centered A1 logits using the original 80,000/20,000 train/test split. Added features describe resource-weighted A2 production alone, excluding the A1 production already in the baseline.

| Regression features | Held-out $R^2$ |
| :--- | ---: |
| Local production, variety, concentration and adjacency | 0.874308 |
| Local features + optimistic weighted A2 production | 0.877942 |
| Local features + worst-case weighted A2 production | 0.879869 |
| Local features + both | 0.882880 |
| Existing context and board-scarcity features | 0.936937 |
| Those features + optimistic weighted A2 production | 0.936983 |
| Those features + both | 0.937471 |

Beyond the existing features and optimistic complement, the worst-case complement adds $\Delta R^2=0.000487$ [0.000436, 0.000539], with paired $t=18.65$ on 19,999 degrees of freedom. Adding both summaries changes exact top-choice agreement from **62.03% to 62.67%**. The tests compare per-board prediction errors with fitted coefficients held fixed.

The production objective lets B minimize A's production regardless of the cost to B. It excludes road strategy and later play. Ties count as agreement whenever the chosen A1 is a maximizer.

The resource weights, in wood/brick/sheep/wheat/ore order, are **1.77472, 2.11092, 1.76318, 1.97658 and 2.11934**.

[Results and methods](/figures/catan-opening-anticipation/portfolios-README.md) · [Regression comparisons](/figures/catan-opening-anticipation/portfolios-secondary.json)

</details>

#### Comparing complete openings

A production score misses ports, starting hands, expansion routes and the strength of B's own position. We therefore built a small opening tree externally and asked the value head to evaluate the resulting positions.

On **128 randomly selected held-out boards**, we considered the network's top six A1 choices, adding the strongest local production alternatives when absent. For each A1, B could choose its four highest-scoring B1 placements and then its four highest-scoring B2 placements. We evaluated every legal A2 response, with roads chosen greedily by the policy. This produced **527,556 completed opening evaluations**, each just before A's first roll.

For each opposing reply pair, let

$$
F(a,b_1,b_2)=\max_{d\text{ legal}}v_\theta\!\left(s_{\mathrm{after}\ A1,B1,B2,A2}\right).
$$

We then compared three summaries of $F$: the most favorable B reply examined, an average weighted by B's policy preferences, and the least favorable B reply examined. Each assumes A selects A2 by the resulting value. Higher values favor A.

| How an A1 candidate is scored | Rank correlation with the A1 policy |
| :--- | ---: |
| Best examined B reply for A | 0.501 [0.441, 0.561] |
| Policy-weighted B replies | 0.608 [0.548, 0.662] |
| Worst examined B reply for A | 0.614 [0.550, 0.675] |
| Immediate resource-weighted production | 0.586 [0.519, 0.651] |

*Mean within-board Spearman rank correlations, with 95% board-bootstrap intervals.*

Accounting for opposing replies agrees with the policy better than assuming a favorable continuation. However, the weighted and worst-reply scores are almost indistinguishable here: their paired correlation difference is **0.006 [−0.035, 0.043]**. Neither clearly exceeds the local resource-weighted baseline. This test cannot tell us that A1 is specifically a minimax choice.

The first sampled board is a useful example. The policy selects **corner 44**. Corner 40 has a much stronger best-case endpoint (+0.558 versus +0.201), but corner 44 is better under policy-weighted replies (+0.028 versus −0.050) and under the worst replies examined (−0.107 versus −0.146). Change A1 and B's response in the viewer to see which A2 remains.

<iframe src="/figures/catan-opening-anticipation.html" title="Compare A1 choices and B1/B2 replies across 128 opening boards"></iframe>

[Open the viewer separately](/figures/catan-opening-anticipation.html) · [Download its boards and continuations](/figures/catan-opening-anticipation-examples.json)

There is no single fixed A2 plan across B's responses. For a given A1, the 16 reply pairs produce an average of **4.80 different value-maximizing A2 corners**. The greedy A2 policy selects the value-maximizing corner on 64.2% of the policy-weighted reply branches, averaged across A1 candidates and boards.

<details>
<summary>Branch coverage, reply weights and sensitivity</summary>

At each B decision, we renormalize the legal-policy softmax over the four examined choices. A reply pair's weight is the product of the two conditional weights. These branches retain **85.5%** of the original joint softmax mass on average. All legal A2 choices are evaluated, with all four opening roads chosen greedily.

Relative to favorable replies, the paired rank-correlation gains are **0.107 [0.042, 0.174]** for weighted replies and **0.113 [0.044, 0.185]** for worst replies. Intervals use 2,000 whole-board bootstrap samples. The policy, evaluator and branch shortlist share the same network, making this a consistency check.

Expanding B's choices from four to eight on the first eight sampled boards added **133,595 evaluations**. The best A1 under weighted and worst-reply scores stayed unchanged on all eight boards. The best A1 under optimistic scoring changed on five.

The mean raw-value gap between value-best A2 and greedy-policy A2 is **0.0230 [0.0209, 0.0252]**, using the same reply weights.

[Complete-opening results](/figures/catan-opening-anticipation/lookahead-results.json) · [Wider-branch control](/figures/catan-opening-anticipation/lookahead-width-sensitivity.json)

</details>

#### Can distant alternatives change A1?

The comparisons above are observational. To test whether local production is the whole explanation, we changed the board while holding the two leading A1 corners fixed.

On **500 held-out boards**, we swapped different resources between equal-number tiles touching neither candidate. We also required each candidate's **best possible resource-weighted A1+A2 production to stay unchanged**. What changed was how much of that production B could remove: the two candidates' relative worst-case scores.

The diagnostic sample is balanced: **250 swaps favor the original favorite's worst-case production and 250 favor its runner-up's**. This selection tests responses to changed fallback options rather than their frequency on random boards.

The network reversed its ranking of the two unchanged A1 corners on **14.2%** of these boards (95% interval: 11.1–17.3%). Its preference therefore depends on context beyond the candidates' immediate production—and beyond just the score of their ideal complement.

But the change favored the candidate with improved worst-case production only **45.8%** of the time (41.5–50.1%). There is no clear positive alignment with that particular minimax proxy. Distant resources also change B's opportunities, expansion prospects and other interactions, so the intervention does not isolate a single reason for the response.

This gives us a more specific conclusion than “the bot plans ahead.” **A1 is sensitive to the rest of the board, but we have not shown that it chooses the best minimax opening or explicitly predicts which A2 will survive.** The complete-opening examples suggest useful hypotheses; identifying the computation inside the GNN would require more targeted interventions and tests of competing explanations.

[Intervention results](/figures/catan-opening-anticipation/remote-results.json) · [Data and reproduction scripts](/figures/catan-opening-anticipation/data-and-methods.zip)

### Experiment 15: Does the bot's opening theory actually beat our simple heuristics?

Pip counts explain most of the opening scores. Do the details they miss actually help the bot win?

We tested two shortcuts on **10,000 fresh boards**: choose the legal corner with the most pips, or weight those pips by resource using the A1 regression from Experiment 4. Each shortcut replaced one player's two settlement choices. The opponent's settlements, every opening road, and all later moves used the same frozen Generation 37 policy without search. Each board was played with the shortcut in both seats.

![Network opening win rates against two simple opening rules; intervals resample whole boards.](/figures/catan-opening-baseline/win-rates.svg)

The network openings won **67.5% against maximum pips** and **64.4% against resource-weighted pips**. Retaining the network's settlement choices instead of replacing them improved the tested player's win rate by **17.5 percentage points** (95% CI 16.9–18.2) and **14.4 percentage points** (95% CI 13.8–15.1), respectively.

This connects the earlier score analyses to actual play: the network's preferences beyond a fixed production score improve its openings under this continuation policy.

The learned openings give up some production for coverage: their two settlements average **22.0 pips**, versus **23.2** for maximum pips, but cover all five resources **50.8%** of the time versus **10.1%**. That gives us a concrete difference to investigate next.

<details>
<summary>Comparison details</summary>

The weighted rule uses wood **1.775**, brick **2.111**, sheep **1.763**, wheat **1.977**, and ore **2.119** per pip at both settlements. These coefficients were frozen before this experiment. Both shortcuts break exact ties by corner index; neither adds a coverage or port rule.

We ran 60,000 rollouts: four heuristic matchups and two identical network-versus-network controls per board (50,000 distinct experimental conditions). The controls agree exactly and average 50% across seats. Within each board, conditions start from the same chance RNG state, although later actions can consume different random events. The 95% intervals use 10,000 bootstrap samples of whole boards, keeping both seats together.

The network's win rates in each seat were:

| Network seat | vs. maximum pips | vs. resource weights |
| :-- | --: | --: |
| A | 70.1% | 67.0% |
| B | 65.0% | 61.9% |

In the **10,000 unique network-versus-network games**, A won **5,475** and B won **4,525**: **54.8% for A** (95% Wilson CI 53.8%–55.7%). The duplicated control replays are excluded from these counts.

All 60,000 rollouts reached a winner; none hit the pre-set 500-turn cap. There were no draws or resignations. This is an independent cohort; the earlier 1,024-board study is preserved separately.

[Results](/figures/catan-opening-baseline/results.json) · [Protocol](/figures/catan-opening-baseline/protocol.json) · [Outcomes](/figures/catan-opening-baseline/outcomes.csv)

</details>

### Experiment 16: Which boards favor the first player?

In Experiment 1 we concluded that the first player usually has a slight advantage, but this is not board context-aware. If one opening corner is exceptional and the next few are mediocre, first pick seems valuable. If several strong corners remain, B's consecutive placements might matter more. A central desert could also change how many productive sites the players compete over.

We tested these ideas against the **10,000 unique self-play outcomes** from Experiment 15. First, we grouped boards by desert location. A won **55.2%** with a central desert, **54.0%** with an inner-ring desert, and **55.2%** with a coastal desert. The center–coast difference was **+0.06 percentage points**, with a 95% interval of **−3.97 to +4.07**.

For production scarcity, simply sorting corners can overstate B's options: two high-pip corners might be adjacent. Instead, take a maximum-pip A1 corner $a$ and find B's strongest *mutually legal* pair after that placement. Define the first-pick premium as

$$
D(a)=P(a)-\frac12\max_{(b_1,b_2)\in\mathcal B(a)}\bigl[P(b_1)+P(b_2)\bigr],
$$

where $P$ counts pips and $\mathcal B(a)$ contains B1/B2 pairs legal after $a$. We average over tied best A1 corners. For example, a 13-pip A1 followed by a best legal B pair totaling 24 pips gives a **1-pip premium**. This measures scarcity among the first three picks; A2 is omitted.

![A win rates by desert location and compatible first-pick pip premium. Confidence intervals overlap substantially across groups.](/figures/catan-board-seat-advantage/board-types.svg)

*Bars show 95% Wilson intervals. The dashed line is A's overall 54.75% win rate.*

Larger premiums show a small descriptive rise in A wins, but the adjusted analysis finds no clear effect. A logistic model combining desert location, this premium, maximum corner pips and the spread of corner pip counts also fails to improve prediction on **2,000 held-out boards**: Brier score **0.2470**, versus **0.2469** for the constant baseline.

We found no useful predictive signal from these simple board descriptors under this policy. Resource balance, ports and the A2 options left after B's pair remain useful hypotheses for a follow-up.

<details>
<summary>Board features and validation</summary>

The analysis uses one completed game per board, excluding the duplicated control replay. Desert rings are graph distance from the central hex: one center tile, six inner tiles and twelve coastal tiles. Every corner is legal before A1. For B's pair, we enforce both the distance from A1 and mutual settlement distance.

The additive logistic regression uses five predictors: two desert indicators, the compatible first-pick premium, maximum corner pips, and the standard deviation of pips across all 54 corners. Its adjusted odds ratio per extra premium pip is **1.03** (95% interval: **0.93–1.14**). None of the five coefficients is significant after Holm correction.

A deterministic seed split assigns 8,000 boards to fitting and 2,000 to testing, with scaling learned on the training portion. The held-out Brier improvement over the training-mean baseline is **−0.00012** (95% paired-bootstrap interval: **−0.00070 to +0.00046**). The center–coast interval resamples boards within each group.

This is an exploratory analysis of existing outcomes. One game per board supports comparisons across board types; repeated continuations would be needed to estimate individual boards' win rates.

[Results](/figures/catan-board-seat-advantage/results.json) · [Board features and outcomes](/figures/catan-board-seat-advantage/board-features.csv) · [Data and analysis code](/figures/catan-board-seat-advantage/data-and-methods.zip)

</details>

