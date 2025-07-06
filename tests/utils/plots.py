import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.4})
sns.set_context("talk")

def plot_error_vs_sample(
    df,
    x_col,
    error_col,
    title,
    xticks=None
):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(
        data=df,
        x=x_col,
        y=error_col,
        hue="test_type",
        style="test_type",
        markers=True,
        dashes=False,
        palette="tab10",
        linewidth=2,
        markersize=8,
        ax=ax
    )
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title, pad=14, fontsize=20)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=16)
    ax.set_ylabel(error_col.replace("_", " ").title(), fontsize=16)

    if xticks is not None:
        if max(xticks) / min(xticks) >= 10:
            ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend(
        title="Test",
        title_fontsize=14,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.
    )
    plt.tight_layout()
    plt.show()
