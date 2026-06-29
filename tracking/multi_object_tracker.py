from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import numpy as np

from helper import Hungarian


@dataclass
class TrackerConfig:
    max_assignment_distance: float = 220.0
    velocity_alpha: float = 0.60
    max_missed_frames: int = 12
    max_tentative_missed: int = 1
    confirm_hits: int = 3
    new_track_min_distance: float = 35.0
    max_output_ids: int = 11
    score_cost_weight: float = 20.0
    motion_cost_weight: float = 8.0


@dataclass
class TrackPoint:
    x: float
    y: float
    observed: bool


@dataclass
class Track:
    internal_id: int
    x: float
    y: float
    last_frame: int
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    consecutive_hits: int = 1
    missed_frames: int = 0
    output_id: Optional[int] = None
    terminated: bool = False
    history: Dict[int, TrackPoint] = field(default_factory=dict)

    @classmethod
    def from_candidate(cls, internal_id, candidate, frame_index):
        track = cls(
            internal_id=internal_id,
            x=float(candidate.x),
            y=float(candidate.y),
            last_frame=int(frame_index),
        )
        track.record(frame_index, observed=True)
        return track

    @property
    def confirmed(self):
        return self.output_id is not None

    def record(self, frame_index, observed):
        self.history[int(frame_index)] = TrackPoint(
            x=float(self.x),
            y=float(self.y),
            observed=bool(observed),
        )

    def predicted_position(self, frame_index):
        gap = max(1, int(frame_index) - int(self.last_frame))
        return np.array(
            [self.x + self.vx * gap, self.y + self.vy * gap],
            dtype=float,
        )

    def update_from_detection(self, candidate, frame_index, config):
        gap = max(1, int(frame_index) - int(self.last_frame))
        new_vx = (float(candidate.x) - self.x) / gap
        new_vy = (float(candidate.y) - self.y) / gap

        self.vx = config.velocity_alpha * self.vx + (1.0 - config.velocity_alpha) * new_vx
        self.vy = config.velocity_alpha * self.vy + (1.0 - config.velocity_alpha) * new_vy
        self.x = float(candidate.x)
        self.y = float(candidate.y)
        self.last_frame = int(frame_index)
        self.hits += 1
        self.consecutive_hits += 1
        self.missed_frames = 0
        self.record(frame_index, observed=True)

    def mark_missed(self, frame_index, config):
        gap = max(1, int(frame_index) - int(self.last_frame))
        next_missed = self.missed_frames + gap

        if self.confirmed:
            if next_missed > config.max_missed_frames:
                self.terminated = True
                return
        elif next_missed > config.max_tentative_missed:
            self.terminated = True
            return

        self.x += self.vx * gap
        self.y += self.vy * gap
        self.last_frame = int(frame_index)
        self.missed_frames = next_missed
        self.consecutive_hits = 0
        self.record(frame_index, observed=False)


class MultiObjectTracker:
    """Automatic multi-object tracker with tentative and confirmed tracks."""

    def __init__(self, config):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_internal_id = 0
        self.next_output_id = 0

    def update(self, candidates, frame_index):
        candidates = list(candidates)
        active_tracks = self._active_tracks()

        if not active_tracks:
            for candidate in self._rank_candidates(candidates):
                self._create_track(candidate, frame_index)
            return

        if not candidates:
            for track in active_tracks:
                track.mark_missed(frame_index, self.config)
            return

        cost_matrix, distance_matrix = self._build_cost_matrix(
            active_tracks,
            candidates,
            frame_index,
        )

        hungarian = Hungarian(cost_matrix)
        hungarian.calculate()

        matched_track_indices: Set[int] = set()
        matched_candidate_indices: Set[int] = set()

        for track_index, candidate_index in hungarian.get_results():
            if track_index >= len(active_tracks) or candidate_index >= len(candidates):
                continue

            distance = distance_matrix[track_index, candidate_index]

            if distance > self.config.max_assignment_distance:
                continue

            track = active_tracks[track_index]
            track.update_from_detection(
                candidates[candidate_index],
                frame_index,
                self.config,
            )
            self._maybe_confirm(track)

            matched_track_indices.add(track_index)
            matched_candidate_indices.add(candidate_index)

        for track_index, track in enumerate(active_tracks):
            if track_index not in matched_track_indices:
                track.mark_missed(frame_index, self.config)

        for candidate_index in self._rank_candidate_indices(candidates):
            if candidate_index in matched_candidate_indices:
                continue

            candidate = candidates[candidate_index]

            if self._is_far_from_existing_tracks(candidate, frame_index):
                self._create_track(candidate, frame_index)

    def confirmed_tracks(self):
        tracks = [
            track
            for track in self.tracks.values()
            if track.confirmed and track.history
        ]
        return sorted(tracks, key=lambda track: track.output_id)

    def _active_tracks(self):
        return [
            track
            for track in self.tracks.values()
            if not track.terminated
        ]

    def _create_track(self, candidate, frame_index):
        track = Track.from_candidate(
            internal_id=self.next_internal_id,
            candidate=candidate,
            frame_index=frame_index,
        )
        self.tracks[track.internal_id] = track
        self.next_internal_id += 1
        self._maybe_confirm(track)
        return track

    def _maybe_confirm(self, track):
        if track.confirmed:
            return

        if track.consecutive_hits < self.config.confirm_hits:
            return

        if self.next_output_id >= self.config.max_output_ids:
            track.terminated = True
            return

        track.output_id = self.next_output_id
        self.next_output_id += 1

    def _build_cost_matrix(self, tracks, candidates, frame_index):
        cost_matrix = np.zeros((len(tracks), len(candidates)), dtype=float)
        distance_matrix = np.zeros((len(tracks), len(candidates)), dtype=float)

        for track_index, track in enumerate(tracks):
            predicted_xy = track.predicted_position(frame_index)

            for candidate_index, candidate in enumerate(candidates):
                candidate_xy = np.array([candidate.x, candidate.y], dtype=float)
                distance = float(np.linalg.norm(predicted_xy - candidate_xy))

                distance_matrix[track_index, candidate_index] = distance
                score_cost = self.config.score_cost_weight * (1.0 - float(candidate.score))
                motion_cost = self.config.motion_cost_weight * (
                    1.0 - min(float(candidate.motion) / 0.20, 1.0)
                )

                cost_matrix[track_index, candidate_index] = distance + score_cost + motion_cost

        return cost_matrix, distance_matrix

    def _is_far_from_existing_tracks(self, candidate, frame_index):
        candidate_xy = np.array([candidate.x, candidate.y], dtype=float)

        for track in self._active_tracks():
            track_xy = track.predicted_position(frame_index)

            if np.linalg.norm(candidate_xy - track_xy) < self.config.new_track_min_distance:
                return False

        return True

    def _rank_candidates(self, candidates):
        return [
            candidates[index]
            for index in self._rank_candidate_indices(candidates)
        ]

    @staticmethod
    def _rank_candidate_indices(candidates):
        return sorted(
            range(len(candidates)),
            key=lambda index: (
                float(candidates[index].score) + 0.25 * float(candidates[index].motion)
            ),
            reverse=True,
        )
