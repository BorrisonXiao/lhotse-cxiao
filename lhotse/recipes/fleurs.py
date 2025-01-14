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
import re
from tn.chinese.normalizer import Normalizer
cn_normalizer = Normalizer()

FLEURS = (
    "train",
    "validation",
    "test",
)


def prepare_fleurs(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "auto",
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
        dataset_parts = ["train", "validation", "test"]
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
            logging.info(f"Processing  subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir):
                logging.info(
                    f"FLEURS subset: {part} already prepared - skipping.")
                continue
            recordings = []
            supervisions = []
            futures = []
            subset = load_dataset(
                "google/fleurs", "yue_hant_hk", cache_dir=corpus_dir, split=part)
            for datum in subset:
                # We will create a separate Recording and SupervisionSegment for those.
                futures.append(
                    ex.submit(parse_utterance, datum, part)
                )
            # For some reason FLEURS data has duplicates
            added = set()
            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                if recording.id in added:
                    continue
                added.add(recording.id)
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            validate_recordings_and_supervisions(
                recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"fleurs_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"fleurs_recordings_{part}.jsonl.gz"
                )

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests


def parse_utterance(
    datum: dict,
    part: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    # Note: stringifying the id is important in this case, otherwise the RecordingSet behaves like a list and leads to bugs
    recording_id = str(datum['id'])
    text = datum['raw_transcription']
    audio_path = Path(datum['path'])
    # Create the Recording first
    if not audio_path.is_file():
        # logging.warning(f"No such file: {audio_path}")
        parent = audio_path.parent
        part = "dev" if part == "validation" else part
        audio_path = parent / part / audio_path.name
        assert audio_path.is_file(), f"No such file: {audio_path}"
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    tokenized_text = tokenize(text.strip())
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="Cantonese Chinese",
        text=tokenized_text,
        gender='M' if datum['gender'] == 0 else 'F',
        custom={'raw_text': text.strip()},
    )
    return recording, segment


def tokenize(text: str, uttid: Optional[str] = None):
    """
    Tokenize the Cantonese/traditional Chinese text.
    This function depends on the WeTextProcessing package.
    """
    chinese_punc = re.compile(r'\!|\;|\~|\！|\？|\。|\＂|\＃|\＄|\％|\＆|\＇|\（|\）|\＊|\＋|\，|\－|\／|\：|\︰|\；|\＜|\＝|\＞|\＠|\［|\＼|\］|\＾|\＿|\｀|\｛|\｜|\｝|\～|\｟|\｠|\｢|\｣|\､|\〃|\《|\》|\》|\「|\」|\『|\』|\【|\】|\〔|\〕|\〖|\〗|\〘|\〙|\〚|\〛|\〜|\〝|\〞|\〟|\〰|\〾|\〿|\–—|\|\‘|\’|\‛|\“|\”|\"|\„|\‟|\…|\‧|\﹏|\、|\,|\.|\:|\?|\'|\"')
    extra_puncs = re.compile(
        r'\'|\.|\\|\/|\*|\-|\<|\>|\#|\$|\%|\^|\&|\(|\)|\_|\+|\=|\:|\"|\`|\~')
    res = text
    # Replace spaces between English letters with "▁" (same as sentencepiece), note that this should
    # be executed twice due to python re's implementation
    res = re.sub(r"([a-zA-Z\d]+)\s+([a-zA-Z\d]+)", r"\1▁\2", res)
    res = re.sub(r"([a-zA-Z\d]+)\s+([a-zA-Z\d]+)", r"\1▁\2", res)
    # WeTextProcessing seems to have a bug that automatically merges spaces between English letters
    res = cn_normalizer.normalize(res)
    # Remove all spaces
    res = re.sub(r"\s+", "", res)
    # Replace the ▁ with a space
    res = re.sub(r"▁", " ", res)
    # Remove all the punctuations.
    res = re.subn(chinese_punc, '', res)[0]
    res = re.subn(extra_puncs, '', res)[0]
    # Add space after Chinese chars
    res = re.sub(r"([^a-zA-Z\d\s])", r"\1 ", res)
    # Add space between non-Chinese and Chinese chars
    res = re.sub(r"([a-zA-Z\d])([^a-zA-Z\d\s])", r"\1 \2", res)

    return res.strip() if not uttid else (uttid, res.strip())
