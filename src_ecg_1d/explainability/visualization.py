# src_ecg_1d/explainability/visualization.py

import numpy as np
import matplotlib.pyplot as plt


def upsample_attention(attention, target_length):

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

  
    signal = ecg_signal[lead_idx]
    L = signal.shape[0]
    time = np.arange(L)

    attention_up = upsample_attention(attention, L)


    assert attention_up.shape[0] == L, (
        f"Attention length {attention_up.shape[0]} "
        f"does not match signal length {L}"
    )


    att_norm = attention_up - attention_up.min()
    if att_norm.max() > 0:
        att_norm = att_norm / att_norm.max()

 
    fig, ax = plt.subplots(figsize=(12, 4))


    ax.plot(time, signal, color="black", linewidth=1)

    ax.fill_between(
        time,
        signal.min(),
        signal.max(),
        where=att_norm > 0,
        color="red",
        alpha=0.3 * att_norm,
        linewidth=0,
        label="High attribution",
    )

    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig
