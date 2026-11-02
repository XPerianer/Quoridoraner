# Overnight optimized transformer: 12-hour learning curve

Session: `qtransformer-night`, isolated L4. Protected ResNet session: `qfinal`; immutable saved opponent files copied read-only, no training changes or evaluations there.

Architecture: pre-LN transformer, 81 board tokens, width256,4blocks,4heads,MLP expansion2 (2,185,864parameters). Fresh weights, not a continuation of the older192×6experiment. Effective training batch1024, gradient microbatch128; inference microbatch64. Four-epoch supervised pretraining on recreated100k-game seed1 corpus, then12hours cumulative self-play+optimization (evaluations/pretraining excluded). Report total hardware time separately.

RL settings match the comparison baseline: pure self-play,2048games periteration, randomized40/400playout caps,AdamW3e-4constant,600kreplay,reuse4,seed1. Seed a quarter of replay from the identical existing30k-game `mix-data.npz`.

Stages at1,2,4,8,12training hours, with full optimizer/replay/RNG/tree state resumed between stages. Hourly model+fullstate snapshots and20-minute state-latest snapshots. Each stage gets400games (200paired openings,seed731,200simulations,100maxplies) versus matching-hour ResNet and frozen13hResNet. Stagesafter1h also face frozen1hResNet. GPU training/evaluation is sequential. No tuning based on stage results within this run.

This measures equal RL time approximately (iteration overshoot recorded), not equal lifetime training compute. Inference speed differs, so equal simulations is not equal thinking time. One seed and12hours cannot establish an asymptotic ceiling; a sustained closing gap would support a longer trial.

Drive mount on the new runtime requires fresh authorization; do not claim Drive durability. Remote snapshots stay in `/content/transformer-night/snapshots`; maintain local copies of snapshots/full resumable state and logs during monitoring. Local large artifacts are ignored by git.

Budget: roughly26–30extra Colab compute units including generation/pretraining/evaluation, at1.94units/hour. Original run continues separately. Do not terminate transformer runtime until final checkpoints and reports are downloaded and validated.

Recipe: `run.sh`. Source fingerprints: `source-sha256.txt`. Monitor status and logs every10minutes while waiting, as requested.
