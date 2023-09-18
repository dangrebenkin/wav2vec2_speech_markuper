import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class ForcedAlignmentUtils:

    def merge_repeats(self,
                      path: List[Point],
                      tokenized: List[str]) -> List[Segment]:
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    tokenized[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    def backtrack(self,
                  trellis: np.ndarray,
                  emission: np.ndarray,
                  tokens_ids: List[int],
                  blank_id: int = 0) -> List[Point]:
        j = trellis.shape[1] - 1
        t_start = np.argmax(trellis[:, j])
        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens_ids[j - 1]]
            prob = np.exp(emission[t - 1, tokens_ids[j - 1] if changed > stayed else 0])
            path.append(Point(j - 1, t - 1, prob))
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]

    def get_trellis(self,
                    emission: np.ndarray,
                    tokens_ids: List[int],
                    blank_id: int = 0) -> np.ndarray:
        assert isinstance(emission, np.ndarray)
        assert len(emission.shape) == 2
        num_frame = emission.shape[0]
        num_tokens = len(tokens_ids)
        trellis = np.empty((num_frame + 1, num_tokens + 1), dtype=np.float64)
        trellis[0, 0] = 0
        trellis[1:, 0] = np.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")
        for t in range(num_frame):
            trellis[t + 1, 1:] = np.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens_ids]
            )
        return trellis

    def merge_words(self,
                    segments: list,
                    separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words
