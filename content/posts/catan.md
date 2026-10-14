---
title: "CatanRL"
date: "October 2026"
excerpt: ""
published: true
paper: true
author: "Owen Zhang"
abstract: ""
---

## Introduction

Among board games that can be picked up without having to comprehend an unreasonable amount of rules, Catan might be one of the best [citation needed]. I think the game teaches several important lessons about trading and strategy games in general; feel free to skip this section since it is not prerequisite to our training setup.

The primary idea is that trading can genuinely be something that is mutually beneficial. Similarly to games like Figgie, one may initially expect that trades between two players must be 0 EV, since only one player may win at the end (in particular, the sum of each players' probabilities of winning, assuming a solved game, at any state, is 1). This may be true globally, but with this set of assumptions alone, another possibility you may arrive at is that the players who are not participating in the trade are simply taking an EV loss, while the two participating (assuming rational players) create some EV between them. This turns out to be the better explanation, at least when trades are made rationally. Thus, part of the game of Catan is figuring out what trades you should be making, with who, and how to convince them to make it with you.

A second idea is that players can create value in different areas of the game, and different players are better positioned to do different things. At a very basic level, players that have a large production of brick and wood are better allocated to win ``longest road" than others. This doesn't mean that you can't compete with them in this aspect, but that it is comparatively worse for you. I would say that victory point generation is best divided into three categories: development cards, cities, and settlements/roads. This creates an interesting game since there are usually four players, and reaching the objective of 10 victory points usually requires investment in at least two of these categories. 

Of course, there are other things you can think about, like tradeoffs between resource production and port access, long-run vs. short-run victory optimization, and more granular things. But I think these may be less obvious conclusions that one arrives at after playing a few times.

## The game and the problem

Catan, in its canonical four-player format, is difficult to model within an RL environment for two main reasons. The first is that there is some level of incomplete information; for instance, you do not know which development cards your opponents are holding, or which resource Opponent A has just stolen from Opponent B. The second is the trading aspect, which is the dynamic that makes Catan such an illustrative (and frustrating) game. Assuming that everyone's objective function is to optimize their individual probability of winning (as opposed to ``throwing" the game or ``kingmaking"), players that are not first may try to trade more often 

## Building the agent

<!-- Explain the representation, model, training process, and search where applicable. -->

## Experiments and results

<!-- Add verified results with the checkpoint, opponents, settings, and sample sizes. -->

## What I learned

<!-- Discuss interesting behavior, limitations, and next steps. -->
