# Alignment Format

Forced-alignment labels are converted to frame-level phoneme IDs before
training the FA-supervised `wav2phoneme` frontend.

The required dense frame file is:

```text
<phoneme_id>
<phoneme_id>
...
```

The number of lines is the target frame count. Consecutive repeated IDs are
kept. This is the interface consumed downstream:

```text
input waveform -> frame-level phoneme posterior -> dense repeated-frame phoneme sequence
```

For CTC ablation runs, the framewise argmax sequence keeps blank IDs. Do not
collapse repeats and do not remove blanks when reproducing the A1 ablation.

