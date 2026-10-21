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

Thank you to Alex Wa ([djdumpling](https://djdumpling.github.io/)) for the continued support and motivation, and [Modal](https://modal.com/) for nearly $1000 in compute credits to support the training process. Congratulations on the Series D!

(This article is currently a DRAFT.)

## Introduction

Among board games that can be picked up without having to digest an unreasonable amount of rules, Catan might be one of my favorite. I think the game teaches several important lessons about trading and strategy games in general. (Feel free to skip this section since it is not prerequisite to our training setup.)

I got sucked into this rabbit hole after various friends introduced me to Colonist, the largest competitive Catan community. I ended up playing a lot (everywhere): it replaced Instagram Reels before bed, filled ferry delays in Greece, and fueled "productive" study breaks.

The primary idea is that trading can genuinely be something that is mutually beneficial. Similarly to games like Figgie, one may initially expect that trades between two players must be 0 EV, since only one player may win at the end (in particular, the sum of each players' probabilities of winning, assuming a solved game, at any state, is 1). This may be true globally, but with this set of assumptions alone, another possibility you may arrive at is that the players who are not participating in the trade are simply taking an EV loss, while the two participating create some EV between them. This turns out to be the better explanation, at least when trades are made rationally. Thus, part of the game of Catan is figuring out what trades you should be making, with who, and how to convince them to make it with you.

Another idea is that players can create value in different areas of the game, and different players are better positioned to do different things. At a very basic level, players that have a large production of brick and wood are better allocated to win “longest road” than others. This doesn't mean that you can't compete with them in this aspect, but that it is comparatively worse for you. I would say that victory point generation is best divided into three categories: development cards, cities, and settlements/roads. This creates an interesting game since there are usually four players, and reaching the objective of 10 victory points usually requires investment in at least two of these categories.

There are other things you can think about, like tradeoffs between resource production and port access, long-run vs. short-run victory optimization, and more granular things. But I think these may be less obvious conclusions that one arrives at after playing a few times.

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

When a seven is rolled:

- A player with **9 or fewer resource cards** discards nothing.
- A player with **10 or more** chooses and discards **half their resource cards, rounded down**.
- For example, hands of 10 and 11 both discard 5; a hand of 12 discards 6.
- Development cards do not count toward the threshold.
- If both players must discard, the current player discards first.

The friendly robber uses **public victory points**: points from buildings, Longest Road, and Largest Army.

- A player with **two or fewer public points** is protected.
- The robber normally cannot move onto a hex adjoining **either protected player’s buildings**, including the moving player’s own buildings.
- Hidden victory-point cards do not remove protection.
- Protection does not prevent discarding on a seven.
- The same protection applies whether the robber moves because of a seven or a knight.

**Exceptional case:** if protection would leave no legal destination other than the robber’s current hex, the placement restriction is relaxed so the robber can move. Stealing from a protected opponent remains forbidden.

Otherwise, an eligible theft selects one resource card uniformly from the opponent’s hand. This samples individual cards, so resources held in greater quantities are more likely to be stolen.

</details>

<details>
<summary><strong>Trading and other implementation-specific details</strong></summary>

**Bank and port trading**

Player-to-player trades are unavailable. Bank trades exchange identical resources for one different resource:

- **4:1** without a relevant port.
- **3:1** with a generic port.
- **2:1** when giving the resource associated with an owned resource-specific port.

The requested resource must be available in the bank.

**Development-card timing**

Only a **knight** can be played before rolling. Road Building, Year of Plenty, and Monopoly must wait until after the roll.

**Bank shortages during production**

For each resource, the bank must cover the **entire payout owed to both players**. If it cannot, neither player receives that resource on that roll.

This also applies when only one player is owed that resource: they do not receive a partial payout. Other resources are resolved independently.

**Partially resolvable development cards**

Road Building places as many of its two roads as legal locations and remaining pieces allow. Year of Plenty takes as many of its two cards as the bank can supply. The card is consumed even if no benefit can be obtained. Monopoly may legally name a resource the opponent does not hold.

</details>

<details>
<summary><strong>Simulation limits used during training</strong></summary>

These are computational safeguards rather than strategic rules:

- Gen37 self-play stops an unfinished game when its turn counter reaches **500**, or after **50,000 engine steps**.
- A turn means **one player’s turn**, not a complete round. Engine steps also include individual actions and chance resolutions.
- If stopped at these limits, the player with more **total victory points, including hidden cards**, receives the win; equal scores produce a draw.
- Early resignation was disabled for Gen37’s final training update.
- Separately, the rules engine has an emergency draw limit for games continuing past **5,000 player turns**.

</details>

There are now two interesting claims that we can make. <b>These have extremely important implications for the training.</b>

<div class="claim">

**Claim 1: Player-to-player trading cannot strictly improve both players’ expected results under a common evaluation.**

</div>

Notably, we exclude player-to-player trading from our environment. The motivation is that two-player Catan is a zero-sum game. There is a finite amount of states in Catan (even with stochasticity), and thus there exists some optimal move for each player at every state. Under this solved paradigm, since the probability of win for both players must sum to 1, there can not exist a trade that improves both player's probability of winning.

A trade may help both players build more efficiently in the short-run, but it can not possibly establish a mutual improvement in their expected result. This follows directly from the definition of [zero-sum utility](https://www.cs.cmu.edu/~15281/coursenotes/gametheory/index.html).

So it is true that players can disagree about a trade’s value because of private information (myopically, a trade may seem good for both players). However, under optimal play, either player can reject every trade and retain their original guaranteed outcome. If $v$ is Player A’s minimax value without trading, A’s ability to refuse trades guarantees $V_{\text{trade}}\geq v$, while B’s ability to refuse guarantees $V_{\text{trade}}\leq v$. Thus,

$$
v\leq V_{\text{trade}}\leq v
\quad\Rightarrow\quad
V_{\text{trade}}=v.
$$

**Removing trading is therefore a reasonable modeling choice that preserves the game’s equilibrium value.**

<div class="claim">

**Claim 2: The resources that players hold is exactly deducible by both players under our environment’s information rules.**

</div>

Our environment exposes the bank’s remaining supply of each resource. Because there are exactly **19 cards of each resource**, distributed between the bank and the two players, either player can reconstruct the opponent’s holdings.

For example, if the bank contains 11 wheat and I hold 3, my opponent must hold the remaining 5. 

To justify that this is a modeling choice that remains true to the spirit of 1v1 Catan, we can compare this to Colonist's implementation, in which every possible game event preserves public resource counts (to the best of my knowledge):

- **Discarding after a seven:**  the cards that are discarded are publicly published in the action log.
- **Year of Plenty:** the chosen resources are publicly published.
- **Monopoly:** resources move directly between the two players. Both participants know which resource was taken and how many cards changed hands.
- **Robber theft, including a knight’s effect:** the stolen card moves directly between the two players. 

The difference is that, in Colonist, the action log exposes every change of resources instead of providing a public bank count (which, in my opinion, is a bit silly, since memorizing doesn't seem like an interesting part of Catan strategy).

Thus we supply the resources counts in both hands directly to the network, which naturally seems like a good feature engineering choice.

**Development-card identities remain private**, so the game is still a hidden-information game.

Okay, that's it for the rules. It's not particularly imperative to understand all of these to follow the following part of this post, except for those two claims above.

## Representation 

Now that we have decided what information the agent can access, we need to figure out how to represent the game state rigorously. For the length of this section, we will build around the following game state as an example, and visualize what the bot sees from this position. We're 65 turns in (where a turn measures one player's move and not both), and A is publicly leading by 1 victory point and privately by 2.

<iframe src="/figures/catan-tile-vector.html" title="Explore the full Catan board and all four model input groups"></iframe>

Select a tile, intersection, road location, or global feature to inspect its vector. Click a value to see what it represents.

We chose these features using our knowledge of Catan. Resources, ports and building types use **one-hot encoding**: one entry is 1 and the others are 0, so categories do not acquire an arbitrary numerical order. Dice numbers appear both as exact categories and as **pips**, which express how often they roll. A 6 and an 8 are equally likely, but two tiles on 6 produce together; keeping both encodings preserves that distinction.

We also supply production totals by resource, nearby expansion routes, build affordability, and thresholds such as “at least three ore.” Counts are scaled by fixed divisors, and ownership is relative to the acting player. These choices make useful relationships easier to learn without specifying which resources or placements the bot should prefer.

The encoding is **deterministic**: the same board and state produce the same input values. The visible board, pieces and resource hands can be recovered using the fixed board geometry. The complete game state cannot: opponent development-card identities remain hidden, history is summarized, and some public rule information is handled separately when determining legal actions. A fixed network produces a deterministic evaluation of its input; future dice rolls, card draws and thefts remain random.

<details>
<summary>Information outside the 4,752 values</summary>

The network receives public development-card counts and probability estimates instead of the opponent's card identities or the deck's exact remaining composition. It has selected history counters, not the complete sequence of past actions. The turn counter stops distinguishing turns after 200, and development-card history counters stop distinguishing counts after 40.

Gen37 also omits the flag recording whether a development card has already been played this turn. The rules engine still uses it to construct the legal-action mask, which is applied to policy logits after the forward pass. That mask is separate from the 4,752 features and is not an input to the value head. Fixed board connections are also supplied separately to the GNN. These vectors therefore should not be described as a lossless save file or a proven complete representation of everything relevant to a decision.

**These omissions can matter strategically.** Being able to play Year of Plenty now can change what is achievable before ending the turn; masking illegal moves does not give the value head that timing information. History can also help infer hidden cards: an opponent repeatedly leaving one of their valuable production tiles blocked can provide different evidence about holding a knight than simply having no reason to use one. Summarizing history loses some of that context. We have not measured how much either limitation affects Gen37's playing strength.

</details>

## The model: from features to decisions

The representation above gives us 4,752 numbers. We keep their structure: 19 tile vectors, 54 intersection vectors, 72 road-location vectors, and one global vector. The model processes these with a **graph neural network (GNN)**, which repeatedly lets connected locations exchange information before choosing an action.

This section describes **Gen37**: a network with **1,320,375 learned parameters**, a hidden width of **128**, and **four message-passing rounds**. These counts come from the saved checkpoint. Gen37 uses neighbor averages and a scalar value prediction; the attention and distributional-value variants in the repository are separate experiments.

<iframe src="/figures/catan-gen37-model.html" title="Gen37 architecture: tensors, weight matrices, four graph blocks, and output heads"></iframe>

Read this as a computation diagram: each box is an actual operation or tensor, and each arrow carries its output to the next operation. The shapes describe one position. A forward pass computes scores from that position; it does not advance the game or update the weights.

The distinction between **parameters** and **activations** matters here. Parameters are the model’s stored weights and biases. Activations are the temporary numbers those parameters produce for a particular position. After embedding, this network holds $(19+54+72+1)\times128=18{,}688$ activations across its four groups; these are recomputed at every graph block.

### 1. Turn each input row into 128 learned features

We first apply a linear layer followed by a ReLU to each input group. A linear layer computes $Wx+b$; ReLU replaces negative entries with zero.

| Input | Shared transformation | Output shape |
|---|---|---|
| 19 tiles, 32 features each | $32\rightarrow128$, then ReLU | $H^{(0)}\in\mathbb{R}^{19\times128}$ |
| 54 intersections, 48 features each | $48\rightarrow128$, then ReLU | $V^{(0)}\in\mathbb{R}^{54\times128}$ |
| 72 road locations, 16 features each | $16\rightarrow128$, then ReLU | $P^{(0)}\in\mathbb{R}^{72\times128}$ |
| 400 global features | $400\rightarrow128$, then ReLU | $g^{(0)}\in\mathbb{R}^{128}$ |

Shapes here describe one position. During training, each tensor also has a leading batch dimension.

For example, the same $128\times48$ weight matrix processes every intersection. We learn **one intersection encoder**, applied 54 times. This lets a pattern learned at one corner be used at another, while keeping the parameter count small. The model distinguishes corners through their features and connections, rather than a learned corner-ID embedding.

The original input entries have meanings we assigned, such as “my city” or “wheat pips.” The 128 new entries are **learned activations**: their meanings are not specified in advance. Calling one an “ore neuron” would require evidence. A hidden channel is also reused across locations; “128 features” does not mean that the entire board has only 128 activations.

### 2. Exchange information along the board

An intersection’s own features cannot describe every strategically relevant relationship around it. A promising settlement location may also be close to an opponent’s road network. To evaluate that interaction, the model needs information tied to particular neighboring locations.

Each graph round constructs the following updates:

| Representation updated | Information concatenated before the update | Update network |
|---|---|---|
| Each intersection | Itself; mean of adjacent tiles; mean of incident road locations; mean of neighboring intersections; global vector | $640\rightarrow128\rightarrow128$ |
| Each tile | Itself; mean of its six intersections; global vector | $384\rightarrow128\rightarrow128$ |
| Each road location | Itself; mean of its two endpoint intersections; global vector | $384\rightarrow128\rightarrow128$ |
| Global vector | Itself; mean of all tiles; mean of all intersections; mean of all road locations | $512\rightarrow128\rightarrow128$ |

Every item being concatenated is 128-dimensional. An intersection therefore collects five vectors, giving $5\times128=640$ inputs to its update network. That network has a ReLU between its two linear layers. Its output is added to the existing intersection vector:

$$
v_i^{(\ell+1)}
=
v_i^{(\ell)}
+
f_V^{(\ell)}\!\left(
\left[v_i^{(\ell)},\ \overline{h}_{N_H(i)}^{(\ell)},\ \overline{p}_{N_P(i)}^{(\ell)},\ \overline{v}_{N_V(i)}^{(\ell)},\ g^{(\ell)}\right]
\right).
$$

Here, brackets mean concatenation, and a bar means averaging over the specified neighbors. The addition is a **residual connection**: a round learns a change to the existing representation. Coastal intersections average only their real neighbors; missing neighbors are masked out.

All four updates in a round read the previous round’s representations. We repeat this process four times, using **different learned weights in each round**, while preserving all tensor shapes. There is no extra ReLU after the residual addition, so later representations can contain negative values.

The global vector provides a short route for board-wide information. It collects three separate board averages in one round and broadcasts its updated information in the next. Consequently, the network is not restricted to a four-hop local neighborhood, although distant information passing through this route is compressed into a shared summary.

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

After round four, small linear layers called **heads** read the representations. The policy heads produce **241 logits**, one score per action slot:

| Head | Reads | Output slots |
|---|---|---:|
| Build a road | Each final road-location vector | 72 |
| Build a settlement | Each final intersection vector | 54 |
| Upgrade to a city | Each final intersection vector | 54 |
| Move the robber | Each final tile vector | 19 |
| Other actions | Final global vector | 42 |
| **Total policy output** | | **241** |

The 42 other slots cover buying a development card, four playable development-card types, five resource choices, five discard choices, 25 bank/port trade pairs, ending the turn, and rolling the dice. Some slots are illegal in a particular state; for example, trading a resource for itself is masked out.

For settlement location $i$, the entire final readout is a dot product:

$$
\ell_{\mathrm{settle},i}=w_{\mathrm{settle}}^\top v_i^{(4)}+b_{\mathrm{settle}}.
$$

The same 128 weights and one bias score all 54 intersections. The four graph rounds have already mixed local and global information into each $v_i^{(4)}$.

The rules engine supplies the legal-action set $\mathcal{A}(s)$. We mask illegal actions and apply a softmax over the rest:

$$
p_\theta(a\mid s)=
\begin{cases}
\displaystyle\frac{\exp(\ell_a)}{\sum_{b\in\mathcal{A}(s)}\exp(\ell_b)}, & a\in\mathcal{A}(s),\\
0, & \text{otherwise}.
\end{cases}
$$

These are **policy probabilities**, expressing the network’s relative preference among legal actions. They are not the probability of winning after each move. A search-free bot can simply choose the highest-probability legal action.

A separate head reads the final global vector and predicts one value:

$$
v_\theta(s)=\tanh(w_v^\top g^{(4)}+b_v).
$$

This lies between $-1$ and $1$, from the acting player’s perspective. Its training target combines the eventual game result with a search estimate, as described below. Positive means favorable to the actor; it is not automatically a calibrated win probability.

### Worked example: why the forward pass selects road 27

For the same mid-game position above, Gen37’s largest legal logit belongs to **building a road at edge 27**, connecting intersections **18 and 23**. The recorded game also played that move next. We can follow the network computation that leads to this preference, without running search again.

1. **Encode the edge.** Road location 27 starts as a 16-number input row. The shared road-location encoder maps it to a 128-number vector, $P_{27}^{(0)}$.
2. **Read its surroundings.** In each of four rounds, it combines its previous vector, the mean of intersections 18 and 23, and the previous global vector. Those $3\times128=384$ values pass through the round’s $384\rightarrow128\rightarrow128$ update network. Adding that update produces the next $P_{27}$. Its endpoints have meanwhile been receiving their own messages from adjacent tiles, edges, and corners.
3. **Score the move.** The road head takes a weighted sum of the final 128 activations and adds its bias. For this edge, the activation–weight products sum to $2.685312$, and the bias is $-0.122642$, giving a logit of **2.562670**.
4. **Compare legal actions.** The other heads score the alternatives. The legal-action mask removes 211 of the 241 slots, and a softmax over the remaining 30 gives road 27 **30.60%** of the probability mass. It has the highest probability, so a greedy network-only decision selects it.
5. **Translate the action.** Road actions occupy indices 0–71, so index 27 decodes directly to `BuildRoad(27)`. Applying it places A’s road on that edge and spends one wood and one brick: A’s holdings change from 4 wood / 2 brick to 3 wood / 1 brick. The network would then be called again for the next decision.

| Legal action | Raw logit | Network policy probability |
|---|---:|---:|
| **Road at edge 27** | **2.5627** | **30.60%** |
| City at intersection 22 | 2.2104 | 21.52% |
| Buy a development card | 1.6179 | 11.90% |
| Road at edge 49 | 1.3241 | 8.87% |
| City at intersection 23 | 1.2657 | 8.37% |

For a concrete hidden-channel example, the final activation in channel 125 at road location 27 is approximately $-8.982087$. The road head’s weight on that channel is $-0.157480$, so their product adds approximately **+1.414500** to the road’s logit. The other 127 channel contributions and the bias complete the logit; some contributions push it down.

That calculation explains how the score is assembled. It does not yet explain what channel 125 represents strategically, or which original features caused its activation. Those are interpretability questions requiring further experiments. Likewise, the recorded game agreeing with the network’s top choice does not show that search used the same reasoning. This walkthrough is an actual CPU float32 forward pass through the saved Gen37 checkpoint, not an attribution of the search process.

The separate value output is $0.8309$ for A. The move probabilities above describe preferences among legal actions, not win probabilities or a complete plan for the rest of the turn.


<details>
<summary>Where the 1,320,375 parameters are</summary>

A linear layer from $m$ inputs to $n$ outputs has $mn+n$ parameters, including its biases. Shared layers are counted once, regardless of how many locations use them.

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

Within each graph round, the intersection update has 98,560 parameters, tile and road updates have 65,792 each, and the global update has 82,176. The two auxiliary heads are trained but not evaluated for ordinary move selection. Excluding their 1,032 parameters leaves **1,319,343 parameters used by the policy/value forward pass**. Fixed adjacency indices and masks are buffers, not learned parameters.

</details>

### Why this architecture?

Weight sharing makes the same local pattern recognizable across the board. Message passing gives each location access to its particular neighbors before information is pooled. The global vector supplies resources, scores, and board-wide context; residual updates let successive rounds refine existing representations. Location-specific heads keep a settlement score attached to the intersection being considered.

These choices encode useful structure in Catan and keep the model small enough to evaluate repeatedly inside search. Mean aggregation is simple and inexpensive, but can lose distinctions between neighborhoods that produce the same average. The global summary is also a compression. **Width 128 and four rounds are the configuration we used, not a demonstrated optimal architecture.** Explaining their role does not establish that each choice independently improved playing strength.

## Four operations in the learning loop

The project combines four operations. Only the last one learns new weights:

| Operation | Input → output | What changes? |
|---|---|---|
| **Apply rules** | State + action → next state | The board and game state |
| **Evaluate** | Observation → policy and value | Temporary GNN activations |
| **Search** | Current position → move and learning targets | A tree of visits and value estimates |
| **Train** | Recorded positions + targets → updated GNN | Neural weights |

<iframe src="/figures/catan-gen37-training.html" title="How frozen Gen36 generates search targets and games that train Gen37"></iframe>

### 1. Apply the rules

The **rules engine** knows which moves are legal, what they cost, how pieces affect the board, and when somebody has won. Given “build road 27,” it places the road and deducts one wood and one brick. It also supplies probabilities for dice, development-card draws, and thefts. This is ordinary game code: we do not ask a neural network to learn the rules or predict state transitions.

### 2. Evaluate a position

The engine encodes a position, and the **same GNN described above** produces a policy and a value. Its policy suggests promising actions; its value estimates how favorable the position is for the acting player. There is no separate search network. Both predictions share the encoders and four graph blocks, with different output heads.

Changing the input changes the activations and predictions. It does not change the weights. We can call this frozen network once to choose a move cheaply, or repeatedly inside search to evaluate hypothetical positions.

## Search: investigate possible continuations

The third operation is **Monte Carlo tree search (MCTS)**. Starting from the current position—the **root**—it uses the engine to construct possible continuations and the frozen GNN to evaluate them. Search can therefore change the move played now, without changing the network.

One simulation does four things:

1. Select a root action to investigate.
2. Follow actions and chance outcomes through a copied game state.
3. At a new decision position, obtain a policy and value from the same GNN. At a terminal position, use the actual result instead.
4. Add that evaluation to the visited tree edges' statistics.

An edge's **visit count $N(s,a)$** records how often it was investigated; its **value estimate $Q(s,a)$** averages backed-up evaluations, oriented to the player choosing at $s$. These statistics change between simulations. The weights do not. A simulation usually stops at a network-evaluated leaf, so 1,024 simulations do not mean 1,024 complete games.

### Gumbel at the root; PUCT inside the tree

At the root, **Gumbel selection** shortlists up to eight legal actions using the network's log probabilities plus random noise. The noise gives less-favored moves opportunities. Search investigates the candidates, ranks them using their original noisy scores and accumulated Q estimates, and progressively removes roughly half:

$$
8\ \text{candidates}\;\longrightarrow\;4\;\longrightarrow\;2\;\longrightarrow\;1.
$$

This is **sequential halving**. The Gumbel noise is drawn once per action and held fixed through these rounds; it is not noise added to neural weights.

Below the root, **PUCT** chooses actions using their current Q estimate plus an exploration bonus. The bonus favors actions with a high network prior and relatively few visits. At an opponent's decision, Q is oriented to that opponent, so search models them trying to improve their own result. The sign follows the acting player, not tree depth: one Catan turn can contain several consecutive actions by the same player.

<details>
<summary>The selection formulas and chance handling</summary>

For root action $a$, draw $u_a\sim\operatorname{Uniform}(0,1)$ and set

$$
g_a=-\log(-\log u_a),\qquad
r_a=\log\max(p_\theta(a\mid s),10^{-9})+g_a.
$$

The shortlist contains the eight largest $r_a$ values, or all legal actions if fewer exist. Halving ranks survivors by $r_a+\sigma Q(s,a)$, with $\sigma=50+\max_b N(s,b)$ under the Gen37 run's settings. Remaining simulations after elimination investigate the finalist.

At interior decision nodes, PUCT selects the action maximizing

$$
Q(s,a)+c_{\mathrm{puct}}p_\theta(a\mid s)
\frac{\sqrt{\max(1,N(s))}}{1+N(s,a)}.
$$

The policy prior stays fixed after expansion; Q and visit counts change. Ties are broken randomly.

Chance outcomes are not choices available to the player. The engine supplies their probabilities, and this search schedules the outcome furthest behind its allotted visit share, with random tie-breaking. Forced transitions can be advanced without a new policy choice.

</details>

### Hidden cards and search outputs

For Gen37's training data, each searched decision received **1,024 total simulations**. When the opponent held hidden development cards, search normally sampled four possible allocations consistent with the acting player's information and ran a separate 256-simulation tree for each. Otherwise, one tree received the budget. These are different possible worlds evaluated by the same network. Their final policies are averaged equally.

Search returns **a move, a policy target $\pi$, and a value estimate**. The move advances the game; the other outputs become learning targets. The policy combines the prior and search Q estimates, rather than simply normalizing visit counts. A sharply peaked policy is not a calibrated statement that an action is certainly best.

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

The hidden-world sampler uses only the acting player's information. However, deeper decisions in a sampled tree can behave as if that sampled allocation were known. This is an approximation to imperfect-information play, not a certificate of optimal play.

</details>

## Self-play: generate the training examples

**Self-play** organizes the first three operations into complete games. The engine presents a position, search calls the frozen GNN to select an action, and the engine applies it. Repeat until the game ends. We save learner positions with their search policy and value, then attach the final result and piece ownership.

To produce **Gen37**, the frozen evaluator was **Gen36**. Opponents were sampled with weights of 75% mirror play and 25% a frozen `exploiter33` opponent. The run completed **1,940 games**, yielding **289,034 learner positions**. A copy of Gen36 was then trained on those examples and saved as Gen37.

During the first 60 engine-loop steps, learner moves were sampled from the search policy; later they used search's returned move. That window includes chance events and forced actions, not 60 full turns. Forced moves require no policy-learning example, and the frozen opponent's decisions are not learner targets.

A **generation** is one data-generation/training iteration. It is unrelated to the four graph blocks, and it does not mean weights change after each move.

## Training: update the weights

The fourth operation is the only one that changes the GNN. Once games are collected, the trainer runs their recorded observations through the network in batches, compares predictions with targets, and backpropagates through the output heads, graph blocks, and encoders. **Search is not part of this gradient calculation.** AdamW uses the gradients to update the weights.

Each recorded position supplies three learning signals:

- **Policy:** match search's action distribution $\pi$ using cross-entropy.
- **Value:** predict $y_v=0.35z+0.65v_{\mathrm{search}}$, where $z$ is the eventual result ($+1$ for a win, $-1$ for a loss, $0$ for a draw), and both terms use the acting player's perspective.
- **Future ownership:** predict the eventual owner/building type at each intersection and road location. These auxiliary heads provide local supervision for the shared representations.

The Gen37 objective is

$$
\mathcal L=
\underbrace{-\sum_{a\in\mathcal A(s)}\pi(a\mid s)\log p_\theta(a\mid s)}_{\text{policy}}
+\underbrace{(v_\theta(s)-y_v)^2}_{\text{value}}
+0.15\mathcal L_{\mathrm{ownership}}.
$$

If search assigns a move more probability than the network does, the policy loss pushes toward raising its score. This is how expensive search can teach a cheaper forward pass. But the shared weights affect many predictions, and search itself can be wrong: neither an optimizer step nor a higher generation number guarantees better play.

The value target also explains why we should measure the evaluator rather than assume it is calibrated. It mixes actual outcomes with the previous evaluator's search estimates, so it can inherit their errors. Gen37 applied no time discount to outcomes.

<details>
<summary>Loss details and the verified Gen36 → Gen37 recipe</summary>

For a normalized policy target, the cross-entropy gradient with respect to a legal action's logit is $p_\theta(a\mid s)-\pi(a\mid s)$. A target larger than the current probability therefore gives a negative derivative; gradient descent pushes the logit upward locally.

The auxiliary intersection classes are empty, my settlement, my city, their settlement, and their city; road classes are empty, mine, and theirs. Its loss excludes existing cities and built roads, whose ownership cannot change, and weights nonempty classes four times as strongly as empty ones. It adds the mean retained intersection loss and mean retained road loss. These are prediction targets, not extra game rewards.

| Setting | Recorded configuration |
|---|---|
| Initialization | Gen36 weights; fresh AdamW optimizer |
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

The learner's seat is randomized against the frozen opponent. Training randomly applies one of six valid board relabellings to each example's features and targets. It does not explicitly duplicate each batch six times; the final incomplete batch is dropped.

Ten shards requested 200 games each, with up to five self-play workers running concurrently. A 97% completion cutoff avoided waiting for the last few games. The manifest records 1,940 completed games and zero draws. Games stopped at the 500-turn or 50,000-step limits were scored by total victory points, including hidden cards. Thus the recorded outcome can reflect this cutoff convention rather than the normal victory threshold.

</details>

<details>
<summary>Where learning began, and a training-history caveat</summary>

Gen0's weights were random. The supplied rules engine already knew legal moves, exact transitions, and terminal results. Game results provide an external learning signal even while initial predictions are poor. Better evaluations can make search more useful, and new searches supply targets for the next update. “From scratch” describes the learned weights; the rules, observation design, graph geometry, action mask, and search algorithm were provided.

Gen37 continued Gen36 rather than restarting randomly. Earlier generations used a search path that could access referee-hidden information. Search was later corrected to use the acting player's information and sampled hidden-card allocations. Gen36 and Gen37 trained on new games from that corrected search, while retaining inherited weights. Gen37 therefore does not have an entirely clean information-boundary training history.

The architecture and loss were checked against the source archive saved with `honest-gen37-20260911`, and the parameter count was recomputed from its checkpoint (SHA-256 beginning `fbc5209b17a14a1d`). More recent defaults and separate experiments are not settings used by this checkpoint.

</details>


## What the value head has learned

The **value head belongs to the same Gen37 GNN**. It reads the final 128-number global vector, computes a weighted sum $u=w_v^\top g^{(4)}+b_v$, and returns $v=\tanh(u)$ for the acting player. Here, **raw value** means the output before probability calibration; it has already passed through `tanh`.

### From a value score to winning probability

There is a natural conversion from the $-1$ to $+1$ scale to an **implied winning probability**:

$$
p_{\mathrm{implied}}=\frac{v+1}{2}=\sigma(2u),
\qquad \sigma(x)=\frac{1}{1+e^{-x}}.
$$

Rescaling the `tanh` output is therefore exactly equivalent to applying a sigmoid to twice the weighted sum.

| Value $v$ | Implied winning probability |
|---:|---:|
| $-0.6$ | 20% |
| $0$ | 50% |
| $+0.6$ | 80% |
| $+0.9$ | 95% |

Why this conversion? If wins are labeled $+1$ and losses $-1$, then, with no draws, the expected outcome is $p_{\mathrm{win}}-(1-p_{\mathrm{win}})=2p_{\mathrm{win}}-1$. A model that accurately predicts that expectation can be converted back to a winning probability using $(v+1)/2$.

**Gen37 is trained on a blend of the game result and search's evaluation:**

$$
y_v=0.35z+0.65v_{\mathrm{search}},
\qquad \mathcal L_{\mathrm{value}}=(v-y_v)^2.
$$

Here $z$ is the eventual result: $+1$ for a win, $-1$ for a loss, or $0$ for a draw. Both terms use the acting player's perspective. Backpropagation updates the value head and the shared GNN to reduce the squared error, alongside the policy and ownership losses.

For example, a position whose player eventually wins, but whose recorded search value was $0.2$, receives target $0.35(1)+0.65(0.2)=0.48$. That is one sample's teaching target. Across many examples, the network learns to predict these blended targets from the position.

The conversion gives an implied probability; **calibration checks whether it matches observed winning frequencies**. Among positions assigned 80%, does the acting player actually win about 80% of the time? Accurate search estimates can support calibration, while search errors can carry into the learned value. The examples below show $v$ directly; their rescaled probabilities have not been separately calibrated.

<details>
<summary>Draws and how we could calibrate the probabilities</summary>

With draws, an accurate expected-outcome prediction rescales to $P(\mathrm{win})+\tfrac12P(\mathrm{draw})$, rather than winning probability alone. Gen37 applied no time discount to its outcome targets.

One possible calibration step is to freeze Gen37 and fit $p=\sigma(au+b)$ to actual wins and losses from new games. The two fitted numbers $a$ and $b$ adjust the scale and offset; this does not retrain the GNN. We would fit and evaluate on separate whole games, so positions from the same game cannot appear in both sets. These probabilities describe the continuation strategy and opponents used to collect the data.

Opening calibration needs its own check: performance on midgame positions does not establish reliable opening probabilities. Calibration can correct systematic bias or scale, but cannot add board information the score failed to capture. No calibration function has been fitted for the results shown here.

</details>

### Five positions to compare with intuition

The examples below range from a substantial deficit to a verified win available on the next move. All use **A’s perspective**, with A in blue and B in red. Each is a real nonterminal position evaluated by Gen37 without search.

<iframe src="/figures/catan-gen37-value-examples.html" title="Five real Catan positions with Gen37 values, visible resources, scores, and later outcomes"></iframe>

The 7–7 position is particularly interesting: equal public scores receive a value of **+0.900 for A**. A’s cities, resources, and opportunities differ from B’s, but this example alone cannot tell us which features explain the prediction. At the other extreme, the **+0.998** position has a legal city upgrade that immediately wins. These are illustrations selected for a range of scores; they are not an accuracy test.

### Do the scores track subsequent results?

On **12,230 recorded positions from 776 games that postdate Gen37 training**, the player favored by the sign of the value eventually won **85.6%** of the time (95% game-bootstrap interval: **84.1–87.0%**). A simple baseline that picks the public-VP leader scores **75.8%**, with tied scores receiving half credit. Each game has equal total weight.

![Acting-player observed win rates increase across five raw Gen37 value ranges: 3.2%, 26.5%, 51.4%, 77.1%, and 97.1%. Error bars resample whole games.](/figures/catan-gen37-value-outcomes.svg)

Higher scores track better outcomes in this sample. This supports useful outcome prediction here; it does not establish calibrated probabilities, optimal-play values, or performance against arbitrary opponents.

<details>
<summary>Evaluation sample, weighting, and limitations</summary>

The observations come from an existing corpus of the recovered Gen38-generation games, generated using a neutral-prepared Gen37 parent and a frozen-opponent mixture. They were not part of original Gen37 training. The corpus contains up to 16 evenly spaced eligible turns per game, restricted to post-roll action positions where the opponent holds a hidden development card. It is a selected convenience sample, not all possible game situations or a preregistered benchmark.

We ran the original frozen Gen37 checkpoint on all 12,230 inputs, without fitting a calibration function or selecting positions by outcome. The model sees actor-available observations; outcomes are read afterward from the records. The five displayed boards are selected illustrations from this corpus, separate from the aggregate calculation.

Each game contributes total weight one, divided equally among its sampled positions. The intervals use 2,000 bootstrap resamples of whole games, preserving dependence between positions in the same game. Chart bins are left-closed/right-open, with +1 included in the last bin. No sampled games were draws. The public-VP baseline uses only the two visible scores; it gets half credit on ties. The network exceeds this baseline by **9.8 percentage points** (paired game-bootstrap interval: **8.4–11.0 points**).

Later positions can already be close to decided, and this opponent mixture, development-card filter, and interrupted-run recovery limit generalization. Game results reflect the recorded continuation policies, not optimal play. The value head was trained on $0.35z+0.65v_{\mathrm{search}}$, so its raw score is not a direct prediction trained exclusively on eventual wins.

</details>

## Does the first player have an opening advantage?

**Gen37 sees the empty opening as almost even, with a very small average preference for going first.** Across **10,000 fresh random boards**, evaluated before A1 with A as the first player, its mean raw value is **+0.0015** and its median is **+0.0045**.

![Distribution of first-player values before A1 across 10,000 random boards, clustered near zero with mean +0.0015. Negative values favor second player B; positive values favor first player A.](/figures/catan-gen37-opening-values.svg)

The model favors A on **55.48% of boards**, but the scores are small: the middle 90% fall between **−0.0548 and +0.0489**. The fraction of boards with a positive score is **not** a 55.48% predicted game win rate. On the convenient $(v+1)/2$ scale, the mean is only **50.076%**, and that rescaling has not been calibrated as winning odds.

This is evidence about the learned evaluator’s preference over our board generator. Establishing a playing advantage would require matches on the same boards with bot identities swapped between seats, and further work to assess optimal play. Evaluating more boards reduces sampling error; it does not remove systematic error in the value head.

<details>
<summary>Opening experiment and reproducibility</summary>

- **Checkpoint:** original Gen37, SHA-256 beginning `fbc5209b17a14a1d`.
- **Boards:** `BoardMode::ColonistLike`, seeds 1,310,000–1,319,999; the actual engine’s resource, number-token, and port generator.
- **Position:** the legal initial state before any settlement or road, with first seat A to act. No placements, resource hands, or development cards have been supplied.
- **Computation:** one CPU float32 forward pass per board, batched for efficiency; no search, fitting, or games played.
- **Mean raw value:** +0.001527; board-sampling standard error 0.000318; approximate 95% interval [+0.000904, +0.002151]. This interval covers board-sampling uncertainty, not model bias or calibration error.
- **Checks:** fresh-state invariants on all boards; exact agreement with the archived Gen37 forward implementation; all six valid geometric relabellings agree within $1.5\times10^{-7}$ on a 32-board check.

The second player’s opposite score refers to the same legal initial state. We do not manufacture a second empty state where B acts first: that would change the opening-order question. Geometric rotations and reflections preserve first/second roles and are not counted as additional independent boards.

</details>


<!-- Upcoming sections: remove the surrounding comment when you start writing them.

## Experiments and results
Add verified results with the checkpoint, opponents, settings, and sample sizes.

## What I learned
Discuss interesting behavior, limitations, and next steps.

-->
