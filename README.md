# jepa-lens

Does stop-gradient still matter under SIGReg at scale?

`jepa-lens` is a research harness that compares four JEPA collapse-prevention
mechanisms (`ema_stopgrad`, `sigreg_stopgrad`, `sigreg_nostopgrad`,
`none_nostopgrad`) to test whether stop-gradient remains necessary once SIGReg
regularization is in play, and whether cheap collapse diagnostics can detect
the problem earlier than an expensive linear probe.

See `docs/superpowers/specs/2026-07-29-jepa-lens-design.md` for the full design.
