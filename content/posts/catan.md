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

## Introduction

Among board games that can be picked up without having to comprehend an unreasonable amount of rules, Catan might be one of the best [citation needed]. I think the game teaches several important lessons about trading and strategy games in general. (Feel free to skip this section since it is not prerequisite to our training setup.)

The primary idea is that trading can genuinely be something that is mutually beneficial. Similarly to games like Figgie, one may initially expect that trades between two players must be 0 EV, since only one player may win at the end (in particular, the sum of each players' probabilities of winning, assuming a solved game, at any state, is 1). This may be true globally, but with this set of assumptions alone, another possibility you may arrive at is that the players who are not participating in the trade are simply taking an EV loss, while the two participating (assuming rational players) create some EV between them. This turns out to be the better explanation, at least when trades are made rationally. Thus, part of the game of Catan is figuring out what trades you should be making, with who, and how to convince them to make it with you.

A second idea is that players can create value in different areas of the game, and different players are better positioned to do different things. At a very basic level, players that have a large production of brick and wood are better allocated to win “longest road” than others. This doesn't mean that you can't compete with them in this aspect, but that it is comparatively worse for you. I would say that victory point generation is best divided into three categories: development cards, cities, and settlements/roads. This creates an interesting game since there are usually four players, and reaching the objective of 10 victory points usually requires investment in at least two of these categories.

Of course, there are other things you can think about, like tradeoffs between resource production and port access, long-run vs. short-run victory optimization, and more granular things. But I think these may be less obvious conclusions that one arrives at after playing a few times.

## Setup

Catan, in its canonical four-player format, is difficult to model within an RL environment for two main reasons:

- There is some level of incomplete information; for instance, you do not know which development cards your opponents are holding, or which resource Opponent A has just stolen from Opponent B.

- The second is the trading aspect, which is the dynamic that makes Catan such an illustrative (and frustrating) game. Assuming that everyone's objective function is to optimize their individual probability of winning (as opposed to “throwing” the game or “kingmaking”), players that are not first may try to trade more often with each other, as an example.

It is not completely unreasonable to try to model Catan with these restrictions, but because Catan remains computationally demanding even after simplifying its social elements. We therefore use a two-player environment with several modifications:

Catan remains computationally demanding even after simplifying its social elements. We therefore use a two-player environment with several modifications:

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

**Claim 1: Player-to-player trading cannot strictly improve both players’ expected results under a common evaluation**

We exclude player-to-player trading from our environment. The motivation is that two-player Catan, scored as **+1 for a win, 0 for a draw, and −1 for a loss**, is a zero-sum game. There is a finite amount of states in Catan (even with stochasticity), and thus there exists some optimal move for each player at every state. Under this solved paradigm, since the probability of win for both players must sum to 1, there can not exist a trade that improves both player's probability of winning.

A trade may help both players build more efficiently in the short-run, but it can not possibly establish a mutual improvement in their expected result. This follows directly from the definition of [zero-sum utility](https://www.cs.cmu.edu/~15281/coursenotes/gametheory/index.html).

This argument has limits. Players can disagree about a trade’s value because of private information, different beliefs, or imperfect evaluation; myopically, a trade may seem good for both players, even if they are perfectly rational. **Removing trading is therefore a reasonable modeling choice.**

**Claim 2: The resources that players hold is exactly deducible by both players under our environment’s information rules.**

Our environment exposes the bank’s remaining supply of each resource. Because there are exactly **19 cards of each resource**, distributed between the bank and the two players, either player can reconstruct the opponent’s holdings:

$$
\text{Opponent’s cards of resource }r
=
19-\text{Bank’s cards of }r-\text{Own cards of }r.
$$

For example, if the bank contains 11 wheat and I hold 3, my opponent must hold the remaining 5. Building costs, discards, trades, and development-card effects preserve this accounting: resource cards move between the bank and players rather than leaving circulation.

Resource hands remain deducible after every development-card effect and robber event:

- **Discarding after a seven:** discarded cards return to the bank. Changes in the bank’s resource counts reveal what was discarded. Our engine resolves players’ discards sequentially.
- **Year of Plenty:** the chosen resources leave the bank and enter the player’s hand, so changes in bank inventory reveal the selection.
- **Monopoly:** resources move directly between the two players. Both participants know which resource was taken and how many cards changed hands.
- **Robber theft, including a knight’s effect:** the stolen card moves directly between the two players. With only two players, both are participants in every theft and know the transferred resource.
- **Buying a development card:** its resource cost is fixed and paid into the bank. The purchased card’s identity remains private, but the resource expenditure is known.
- **Road Building and victory-point cards:** neither changes resource holdings.

Consequently, after each resolved event, the same conservation equation still applies: **opponent’s holdings = 19 − bank holdings − own holdings**, separately for each resource. Unplayed development cards remain hidden, but their resource effects become observable when resolved.
We supply these resource counts directly to the network. This claim depends on the information our implementation exposes; two-player play alone does not guarantee it. 

**Development-card identities remain private**, so the game still contains hidden information.

Okay, that's it for the rules. It's not particularly imperative to understand all of these to follow the actually interesting learning part of this post, except for those two claims above.

## Representation 



<!-- Upcoming sections: remove the surrounding comment when you start writing them.

## Building the agent
Explain the representation, model, training process, and search where applicable.

## Experiments and results
Add verified results with the checkpoint, opponents, settings, and sample sizes.

## What I learned
Discuss interesting behavior, limitations, and next steps.

-->
