# AdamW vs Muon vs Aurora

## Why I'm Doing This

I've been thinking a lot about optimizers lately. Not in an abstract "theory is beautiful" way, but in a very practical "why does my model train like garbage sometimes" way. 

AdamW is the default. Everyone uses it. It works. But there's this nagging feeling that we've been using the same optimizer since 2017 and maybe, just maybe, there's something better out there that we haven't bothered to try because AdamW is "good enough."

Then I stumbled across Muon. Keller Jordan's post about it genuinely made me stop and think. The idea that you can orthogonalize your updates — actually preserve the geometry of your weight space instead of just doing gradient descent with some adaptive scaling — felt like one of those "why didn't I think of that" moments. But Muon isn't magic. It has weirdness. Row anisotropy. Tall matrix problems. Dead neurons. The kind of stuff that shows up when you actually look under the hood instead of just staring at the final loss curve.

And then Aurora came along, claiming to fix exactly those tall-matrix issues. Tilde's work on making updates more uniform across different matrix shapes... I need to know if that's real or just another optimizer that looks good on paper.

**So here's the real reason I'm doing this project:** I want to see it myself. Not read about it. Not trust someone else's benchmarks. Actually train small transformers with these three optimizers, measure everything I can, and understand *why* one might be better than another. Not just "Muon converges faster" but "Muon converges faster because X, and here's the cost."

The dead neuron thing especially bugs me. I keep hearing that adaptive optimizers create dead neurons in MLPs, but I've never actually measured it systematically. This project is my excuse to do that. To look at row norm distributions. To compute Gini coefficients. To be annoying about statistics instead of just looking at a smoothed loss curve and calling it a day.

## What This Actually Is

A controlled experiment. Three optimizers, same model, same data, same everything except what we're allowed to change (learning rates, momentum, the optimizer-specific stuff). I'm training tiny GPTs — 10M to 125M parameters — because I don't have an A100 cluster and because if an optimizer can't show its value at small scale, why would I trust it at large scale?

The real test isn't just "who has the lowest validation loss." It's:
- Does Muon actually speed things up, or does it just look good in carefully tuned speedruns?
- Does Aurora actually fix Muon's tall-matrix problems, or is it just Muon with extra steps?
- How many dead neurons does each optimizer create, and where?
- Is the row norm distribution actually more uniform with Aurora, or is that just a claim?

## What I Expect

Honestly? I have no idea who wins. AdamW might still be king at this scale. Muon might be faster but unstable. Aurora might be the best of both worlds, or it might be over-engineered. The whole point is to find out.

Even if AdamW wins, this project is worth it. Because then I'll *know* why I'm still using AdamW instead of just assuming it's the best because everyone else uses it.

## The Vibe

This is a learning project. I'm going to document the math. I'm going to explain Newton-Schulz iterations until I understand them intuitively, not just symbolically. I'm going to look at histograms of row norms and actually think about what they mean. I'm going to make plots that no paper would publish because they're too messy, but that tell me something real.

If you're reading this, you're either future-me trying to remember what I did, or someone who cares about optimizers as much as I suddenly do. Either way: welcome. Let's figure out if any of these fancy new optimizers are actually worth the hype.

---

*Started because I got annoyed that I didn't understand my optimizer. Continuing because the rabbit hole goes deeper than expected.*
