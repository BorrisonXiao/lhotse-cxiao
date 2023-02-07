from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.hklegco import prepare_hklegco
from lhotse.utils import Pathlike

__all__ = ["hklegco"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["auto"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p train -p validation`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use.",
)
def hklegco(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    """HKLEGCO ASR/ST data preparation."""
    if len(dataset_parts) == 1:
        dataset_parts = dataset_parts[0]
    prepare_hklegco(
        corpus_dir,
        output_dir=output_dir,
        dataset_parts=dataset_parts,
        num_jobs=num_jobs,
    )
