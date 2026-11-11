## A value-based opening tree

A complete opening matters for more than resource coverage. The first settlement determines which second settlements survive, what B can build, the starting cards both players receive, and the directions in which their roads point. We can compare these complete positions with Generation 37's value head instead of choosing a hand-written objective such as total pips or resource count.

For a candidate A1 location `a`, an opposing pair `(b1,b2)`, and an A2 response `d`, define

$$
U(a,b_1,b_2,d)=v_\theta(s_{\mathrm{opening}}(a,b_1,b_2,d)).
$$

Every settlement is followed by the frozen policy's greedy road. The endpoint is A's first decision to roll, before the dice are drawn. It is therefore always A's turn, and higher raw values always favor A. These are predictions from the same value head, not measured win rates or calibrated win probabilities.

For each opposing pair, A's best available reply under that evaluator is

$$
F(a,b_1,b_2)=\max_{d\in\mathcal L_{A2}} U(a,b_1,b_2,d).
$$

The three summaries answer different questions:

$$
S_{\mathrm{optimistic}}(a)=\max_{(b_1,b_2)\in\mathcal B_a} F(a,b_1,b_2),\qquad
S_{\mathrm{worst}}(a)=\min_{(b_1,b_2)\in\mathcal B_a} F(a,b_1,b_2),
$$

$$
S_{\mathrm{weighted}}(a)=\sum_{(b_1,b_2)\in\mathcal B_a}
\widetilde\pi(b_1\mid a)\,\widetilde\pi(b_2\mid a,b_1)\,F(a,b_1,b_2).
$$

The optimistic score asks how good the opening could become if B leaves a favorable response. The worst score asks how much remains if B chooses the least favorable reply pair examined. The weighted score averages replies using B's policy preferences. Its top-k softmax weights are renormalized separately at B1 and B2; they are a hypothetical response distribution, not the bot's actual deterministic choice frequencies.

The computation uses 128 uniformly sampled boards from the original 20,000-board held-out split, chosen before looking at the new results. Each A1 shortlist contains the six highest policy scores plus the strongest total-pip and resource-weighted local choices when absent. Resource weights come from the existing resource-only A1 regression fit on the disjoint 80,000-board training split. Lowest corner ID breaks ties; reported agreement credits tied maxima. At each A1, B1 and B2 each branch over their four highest policy scores, while A2 branches over every legal settlement. All roads are greedy, and all states are constructed by the native game engine. A fixed first-eight-board sensitivity check expands the opponent width from four to eight.

We compare the original A1 policy ranking with each external tree score. This is a consistency diagnostic: a feed-forward policy can learn to anticipate consequences without explicitly running this tree. Agreement cannot establish an internal search algorithm. Disagreement can also come from value-head error or policy/value mismatch. The worst-case calculation is limited to the examined opponent replies and greedy road directions, so it is not exact Catan minimax. We also retain the actual greedy B1/B2/A2 endpoint and measure the difference between the policy's A2 and the value-maximizing A2.
