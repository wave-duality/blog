# Two-edge opening road gamble

Generation 37 evaluates A's last free opening-road decision. A and B's selected settlements are two edges apart, so their inward roads meet at an empty degree-three junction. The third edge leaving this junction is unoccupied. Whichever player later builds that edge occupies the only remaining exit through this junction. This is a different geometry from the previous three-edge experiment.

## Sample and intervention

The source is 100,000 random boards (seeds 4100000–4199999) with the original greedy Generation 37 opening histories. There are 30,849 eligible boards. On each, the scanner selects one geometric motif, using the destination's pip total and then deterministic edge IDs to break ties; it never examines intervention scores. It excludes an occupied branch destination and any earlier road incident to the junction or that destination. Native game replay validates both road orientations and both completed inward-road histories. One uniform seeded sample of 2,000 eligible boards supplies the main experiment.

At A's final free-road decision, compare two positions differing only in B's opening-road orientation: away from A, or toward the shared junction. A's legal roads, all settlements, resources, ports, hands and 400 global features are identical. The alternative orientation preserves whether B's altered road directly connects to B's earlier road. The contrast still includes all spatial consequences of moving a road; it is not an isolated abstract threat flag. These positions are conditioned on the model's original settlement choices and do not sample all possible legal placements.

## Readiness and quantities

A is called ready when its existing starting hand contains at least one wood and one brick, before the first roll. This is an observational stratification of the sampled boards. A gets the first normal turn; an already affordable road can therefore claim the branch before B acts if A chooses to build it.

For each position, compute the frozen model's logits and softmax over the native legal roads. The reported entry preference is the probability assigned to A's inward opening road. It is not a win probability. The paired orientation effect is entry preference with B inward minus entry preference with B away. The greedy entry rate is the fraction of boards where A's inward road is the highest-logit legal choice. No tree search or gameplay rollout is performed.

The separate 1,000-board factorial is sampled from 17,751 eligible boards. It swaps B1/B2 settlement-road pairs to change which settlement supplies B's starting hand, requiring equal total card counts and wood-plus-brick affordability in exactly one order. It crosses this with B's road orientation. Within each orientation, all spatial input values match. The interaction is the orientation effect with B ready minus the orientation effect with B unready. The hand swap also changes the specific resource mix and bank counts, so the effect is not solely attributable to the binary affordability flag.

Uncertainty uses whole-board paired observations: normal 95% intervals for means, and Wilson intervals for frequencies. These are descriptive intervals for these sampled positions, not evidence that the road choice is optimal. Main sampling seed is 20261102132; factorial sampling seed is 2026110814. There are 8,000 model evaluations total.

## Files

- `results.json`: aggregate estimates, intervals, exclusions and validation metadata.
- `roads-cases.json`: all 2,000 paired cases, board, exact motif, legal actions and both policies.
- `roads-opponent-readiness-cases.json`: all 1,000 four-condition cases.
- `illustrative-cases.json`: selected examples exhibiting contrasting behavior; deliberately selected illustrations, not a random sample.
- `selected-inputs.json`: score-independent main sample before inference.
- `eligible-screen.json`: complete eligible board seed list.
- `methods/`: exact experiment script and reused evaluation helpers.
- `artifact-sha256.json`: hashes of result JSON artifacts.

Run `python opening_two_edge_gambling.py --n 2000 --readiness-n 1000` from the CatanRL repository using its existing Python environment. The script uses one Torch CPU thread and the original checkpoint. Source shard files and checkpoint are dependencies, not bundled here.
