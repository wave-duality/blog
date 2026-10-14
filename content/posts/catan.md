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

- **Two players, no player-to-player trading.** Trading with the bank and ports remains available.
- **15 victory points to win.**
- **Public resource hands.** Both players can see exactly which resources the opponent holds. Development-card identities remain private.
- **A higher discard threshold.** Players discard on a seven only when holding **10 or more resource cards**.
- **Friendly robber protection.** Players with **two or fewer public victory points** are protected from robber placement and theft, subject to the exceptional placement case described below.

The following specifies the environment used throughout our project. Standard Catan building and connectivity rules otherwise apply.

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


## Representation 



<!-- Upcoming sections: remove the surrounding comment when you start writing them.

## Building the agent
Explain the representation, model, training process, and search where applicable.

## Experiments and results
Add verified results with the checkpoint, opponents, settings, and sample sizes.

## What I learned
Discuss interesting behavior, limitations, and next steps.

-->
