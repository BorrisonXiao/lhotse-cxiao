import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions, fix_manifests
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
)
import re
from concurrent.futures.thread import ThreadPoolExecutor
from tn.chinese.normalizer import Normalizer
cn_normalizer = Normalizer()

SPLITS = (
    "train",
    "dev-asr",
    "dev-mt",
    "test",
)


def prepare_hklegco(
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

    for part in tqdm(dataset_parts, desc="Dataset parts"):
        logging.info(f"Processing subset: {part}")
        if manifests_exist(part=part, output_dir=output_dir):
            logging.info(
                f"HKLEGCO subset: {part} already prepared - skipping.")
            continue
        can_stm_path = corpus_dir / 'st' / f"asr-can.{part}.stm"
        eng_stm_path = corpus_dir / 'st' / f"st-can2eng.{part}.stm"

        # We will create a separate Recording and SupervisionSegment for those.
        manifests[part] = stm_to_supervisions_and_recordings(
            can_stm_path, eng_stm_path, num_jobs=num_jobs)

        if output_dir is not None:
            manifests[part]['supervisions'].to_file(
                output_dir / f"hklegco_supervisions_{part}.jsonl.gz"
            )
            manifests[part]['recordings'].to_file(
                output_dir / f"hklegco_recordings_{part}.jsonl.gz"
            )

    return manifests


def stm_to_supervisions_and_recordings(src_stm: Path, tgt_stm: Path, permissive: bool = True, num_jobs: int = 1):
    with open(str(src_stm), 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(str(tgt_stm), 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()

    stm_entries = {}
    for idx, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        wav, _, spk, beg, end, src_txt = src_line.strip().split(None, 5)
        _, _, _, _, _, tgt_txt = tgt_line.strip().split(None, 5)

        beg, end = float(beg), float(end)
        uttid = Path(wav).stem + f"_{idx:06d}"
        stm_entries[uttid] = {
            'wav': wav,
            'chn': 0,
            'spk': spk,
            'beg': beg,
            'end': end,
            'src_txt': src_txt,
            'tgt_txt': tgt_txt,
        }

    recordings = []
    added_recordings = set()
    end_times = {}
    for uttid, entry in tqdm(stm_entries.items(), desc=f"Making recordings from {src_stm.stem} and {tgt_stm.stem}", leave=False):
        recording_id = Path(entry['wav']).stem
        if recording_id in added_recordings:
            continue
        reco = Recording.from_file(entry['wav'], recording_id=recording_id)
        recordings.append(reco)
        added_recordings.add(recording_id)
        end_times[recording_id] = reco.duration
    recording_set = RecordingSet.from_recordings(recordings)

    supervisions = []
    utts = set()

    # # For some reason parallel execution takes much longer
    # tokenized_texts = {}
    # with ThreadPoolExecutor(num_jobs) as ex:
    #     # Tokenize the text in parallel.
    #     futures = []
    #     for uttid, entry in tqdm(stm_entries.items(), desc="Distributing tasks", leave=False):
    #         futures.append(ex.submit(tokenize, entry['src_txt'], uttid))

    #     for future in tqdm(futures, desc="Tokenizing texts"):
    #         uttid, tokens = future.result()
    #         tokenized_texts[uttid] = tokens

    for uttid, entry in tqdm(stm_entries.items(), desc=f"Making supervisions from {src_stm.stem} and {tgt_stm.stem}", leave=False):
        recording_id = Path(entry['wav']).stem
        beg, end = entry['beg'], entry['end']
        # To solve the speed perturbation supervision-out-of-bound issue.
        end = min(end_times[recording_id] - 0.01, end)
        duration = end - beg
        if duration <= 0.01:
            if permissive:
                print(
                    f"The duration of {uttid} in stm file {src_stm} is too short")
                continue
            else:
                raise ValueError(
                    f"The duration of {uttid} in stm file {src_stm} is too short")

        if uttid in utts:
            if permissive:
                print(f"Duplicate utterance id {uttid}")
                continue
            else:
                raise ValueError(f"Detected duplicate utterance id {uttid}")
        utts.add(uttid)
        tokenized_text = tokenize(entry['src_txt'])
        # tokenized_text = tokenized_texts[uttid]
        supervisions.append(
            SupervisionSegment(
                id=uttid,
                recording_id=recording_id,
                start=beg,
                duration=end - beg,
                channel=entry['chn'],
                speaker=entry['spk'],
                language="Cantonese",
                text=tokenized_text,
                custom={'tgt_lang': "English",
                        "tgt_text": entry['tgt_txt'], "raw_src_text": entry['src_txt']},
            )
        )
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(
        recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    manifests = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests


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
