import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


def get_padding(k, d):
    return int((k * d - d) / 2)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def save_checkpoint(
    checkpoint_dir,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    best,
    logger,
    bad_validations=0,
    early_stopping_best_metric=None,
    latest=False,
    best_loss=None,
):
    state = {
        "generator": {
            "model": generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
        "best_loss": loss if best_loss is None else best_loss,
        "bad_validations": bad_validations,
        "early_stopping_best_metric": early_stopping_best_metric,
        "feature_contract": {
            "sample_rate": 16000,
            "n_fft": 2048,
            "win_length": 2048,
            "hop_length": 320,
            "n_mels": 128,
            "power": 2.0,
            "center": True,
            "pad_mode": "reflect",
            "mel_scale": "slaney",
            "norm": "slaney",
            "log_clamp": 1e-5,
        },
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    saved = []
    if latest:
        checkpoint_path = checkpoint_dir / "latest.pt"
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        saved.append(checkpoint_path.name)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        temporary = best_path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(best_path)
        saved.append(best_path.name)
    logger.info(f"Saved checkpoint(s): {', '.join(saved)}")


def load_checkpoint(
    load_path,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    rank,
    logger,
    reset_optim=False,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    generator.load_state_dict(checkpoint["generator"]["model"])
    discriminator.load_state_dict(checkpoint["discriminator"]["model"])
    if not reset_optim:
        optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        optimizer_discriminator.load_state_dict(
            checkpoint["discriminator"]["optimizer"]
        )
        scheduler_discriminator.load_state_dict(
            checkpoint["discriminator"]["scheduler"]
        )
    return (
        checkpoint["step"],
        checkpoint.get("best_loss", checkpoint["loss"]),
        checkpoint.get("bad_validations", 0),
        checkpoint.get("early_stopping_best_metric", checkpoint["loss"]),
    )
