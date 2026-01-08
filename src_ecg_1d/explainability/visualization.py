# src_ecg_1d/explainability/visualization.py

import numpy as np
import matplotlib.pyplot as plt


def upsample_attention(attention, target_length):
    """
    Upsample attention from Transformer time steps (T)
    to ECG signal length (L).
    """
    attention = np.asarray(attention).reshape(-1)

    x_old = np.linspace(0, 1, num=len(attention))
    x_new = np.linspace(0, 1, num=target_length)

    return np.interp(x_new, x_old, attention)


def plot_ecg_with_attention(
    ecg_signal,
    attention,
    lead_idx=0,
    title=None,
):
    """
    Plot ECG waveform with upsampled Transformer attention.
    """
    signal = ecg_signal[lead_idx]
    L = signal.shape[0]

    # FORCE UPSAMPLING (this prevents your error)
    attention_up = upsample_attention(attention, L)

    # Safety check
    assert attention_up.shape[0] == L, (
        f"Attention length {attention_up.shape[0]} "
        f"does not match signal length {L}"
    )

    time = np.arange(L)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal, color="black", linewidth=1)

    ax.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=attention_up > 0,
        color="red",
        alpha=attention_up * 0.6,
    )

    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
