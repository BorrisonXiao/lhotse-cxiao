import logging
from datasets import load_dataset
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
)
from lhotse.recipes.librispeech import prepare_librispeech
from lhotse.recipes.aishell import prepare_aishell
from lhotse.recipes.nsc import prepare_nsc
from lhotse.recipes.seame import prepare_seame

SPLITS = [
    "SEAME",
    "LibriSpeech",
    "NSC",
    "AISHELL",
    "dev",
]


def prepare_merlion_dev(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    import csv

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    part = "dev"
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=[part], output_dir=output_dir
        )

    logging.info(f"Processing Merlion dev set")
    if manifests_exist(part=part, output_dir=output_dir):
        logging.info(
            f"{part} already prepared - skipping.")
    recordings = []
    supervisions = []

    # Create the supervisions first
    trans_path = corpus_dir / "_CONFIDENTIAL" / "_labels" / \
        "_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv"
    audios = set()
    with open(trans_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for idx, row in enumerate(csv_reader):
            if line_count == 0:
                line_count += 1
                continue
            line_count += 1
            audio_name, uttid, start, end, length, lang, overlap_diff_lang, dev_eval_status = row
            recording_id = Path(audio_name).stem
            start, end, length = float(start) / 1000, float(end) / 1000, float(length) / 1000
            lang = "Chinese" if lang == "Mandarin" else lang
            segment = SupervisionSegment(
                id=f"{recording_id}-{uttid}-{idx:05d}",
                recording_id=recording_id,
                start=start,
                duration=length,
                channel=0,
                language=lang
            )
            supervisions.append(segment)
            audios.add(recording_id)

    # Then, create the recordings
    audio_dir = corpus_dir / "_CONFIDENTIAL" / "_audio"
    for audio_file in tqdm(audio_dir.glob("*.wav")):
        recording_id = audio_file.stem
        if recording_id not in audios:
            continue
        # We need to resample the audio to 16kHz
        recording = Recording.from_file(audio_file, recording_id=recording_id).resample(16000)
        recordings.append(recording)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(
            output_dir / f"dev_supervisions.jsonl.gz"
        )
        recording_set.to_file(
            output_dir / f"dev_recordings.jsonl.gz"
        )

    manifests[part] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }

    return manifests


def prepare_merlion(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    alignments_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train', 'test', 'validation'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "auto":
        dataset_parts = SPLITS
    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    manifests = {}

    if "LibriSpeech" in dataset_parts:
        logging.info("Preparing LibriSpeech manifests...")
        # Prepare the LibriSpeech manifests
        librispeech_dir = corpus_dir / "LibriSpeech"
        manifests["LibriSpeech"] = prepare_librispeech(
            corpus_dir=librispeech_dir,
            dataset_parts="train-clean-100",
            alignments_dir=alignments_dir,
            num_jobs=num_jobs,
        )

    if "AISHELL" in dataset_parts:
        logging.info("Preparing AISHELL manifests...")
        # Prepare the AISHELL manifests
        aishell_dir = corpus_dir / "aishell"
        manifests["AISHELL"] = prepare_aishell(
            corpus_dir=aishell_dir,
            dataset_parts="train",
            sample_rate=16000,
        )

    if "NSC" in dataset_parts:
        logging.info("Preparing NSC manifests...")
        # Prepare the NSC manifests
        nsc_dir = corpus_dir / "nsc"
        manifests["NSC"] = prepare_nsc(
            corpus_dir=nsc_dir,
            dataset_part="merlion",
        )

    if "SEAME" in dataset_parts:
        logging.info("Preparing SEAME manifests...")
        # Prepare the SEAME manifests
        seame_dir = corpus_dir / "SEAME"
        manifests["SEAME"] = prepare_seame(
            corpus_dir=seame_dir,
            dataset_parts="auto",
            num_jobs=num_jobs,
        )

    if "dev" in dataset_parts:
        logging.info("Preparing dev manifests...")
        # Prepare the dev manifests
        dev_dir = corpus_dir / "dev"
        manifests["dev"] = prepare_merlion_dev(
            corpus_dir=dev_dir,
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        for part, manifest in manifests.items():
            supervisions = None
            recordings = None
            try:
                for subset, subset_manifest in manifest.items():
                    if not supervisions:
                        supervisions = subset_manifest["supervisions"]
                    else:
                        supervisions += subset_manifest["supervisions"]
                    if not recordings:
                        recordings = subset_manifest["recordings"]
                    else:
                        recordings += subset_manifest["recordings"]
            except KeyError as e:
                print(f"Error in {part}")
                print(e)
                breakpoint()
            supervisions.to_file(
                output_dir / f"supervisions_{part}.jsonl.gz"
            )
            recordings.to_file(
                output_dir / f"recordings_{part}.jsonl.gz"
            )

    return manifests


def parse_utterance(
    datum: dict
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    # Note: stringifying the id is important in this case, otherwise the RecordingSet behaves like a list and leads to bugs
    recording_id = str(datum['id'])
    text = datum['raw_transcription']
    audio_path = Path(datum['path'])
    # Create the Recording first
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="Cantonese Chinese",
        text=text.strip(),
        gender='M' if datum['gender'] == 0 else 'F'
    )
    return recording, segment
