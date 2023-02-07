import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, Sequence, List, Tuple

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from concurrent.futures.thread import ThreadPoolExecutor


# Note that phaseI is not supported since it is a subset of phaseII
SPLITS = ["conversation-phaseII", "interview-phaseII"]


def prepare_seame(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "auto":
        dataset_parts = SPLITS
    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing SEAME subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir):
                logging.info(
                    f"SEAME subset: {part} already prepared - skipping.")
                continue
            recordings = []
            supervisions = []
            rec_type, phase = part.split("-", maxsplit=1)
            type_path = corpus_dir / "data" / rec_type
            futures = []
            trans_dir = type_path / "transcript" / phase
            audio_dir = type_path / "audio"
            for trans_path in tqdm(
                trans_dir.rglob("*.txt"), desc="Distributing tasks", leave=False
            ):
                recording_id = trans_path.stem
                recording_path = audio_dir / f"{recording_id}.flac"
                if not recording_path.is_file():
                    logging.warning(
                        f"Recording {recording_path} is missing - skipping."
                    )
                    continue
                futures.append(ex.submit(parse_recording,
                               recording_path, trans_path, phase))

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.extend(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"seame_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"seame_recordings_{part}.jsonl.gz"
                )

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests


def parse_recording(recording_path: Path, trans_path: Path, phase: str) -> Optional[Tuple[Recording, List[SupervisionSegment]]]:
    recording_id = recording_path.stem
    # There are more information about each utterance in the recording_id, however, it's not kept for now
    # Please see the documentation file SEAME.V4.0.doc for more details
    spkid = recording_id[4:6]
    gender = recording_id[6]
    # Create the Recording first
    recording = Recording.from_file(recording_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segments = []
    with open(trans_path, "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        _, start_time, end_time, lang, text = line.strip().split("\t", maxsplit=4)
        start_time, end_time = float(start_time) / 1000, float(end_time) / 1000
        if end_time > recording.duration:
            end_time = recording.duration
            if end_time < start_time:
                # For some reason some transcripts contain utterances that are not in the audio file
                logging.warning(
                    f"Start time {start_time} for {recording_id} is greater than recording duration {recording.duration} - skipping."
                )
                continue
            logging.warning(
                f"End time {end_time} for {recording_id} is greater than recording duration {recording.duration} - trimming."
            )
        
        segment = SupervisionSegment(
            id=f"{recording_id}-{idx}",
            recording_id=recording_id,
            start=start_time,
            duration=end_time - start_time,
            channel=0,
            language=lang,
            speaker=spkid,
            text=text.strip(),
            gender=gender,
        )
        segments.append(segment)
    return recording, segments
