# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import av
import collections
import csv
import cv2
import functools
import json
import logging
import math
import matplotlib.collections as mc
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
import random
import uuid

from celluloid import Camera

from fractions import Fraction

import av


# in: video_path, frame_number, boxes: [{ object_type, bbox: {x, y, width, height} }]}, draw_labels
# out: path to image of bboxes rendered onto the video frame
def render_frame_with_bboxes(video_path, frame_number, boxes, draw_labels = True):
    colormap = { # Custom colors for FHO annotations
        'object_of_change': (0, 255, 255),
        'left_hand': (0, 0, 255),
        'right_hand': (0, 255, 0)
    }
    defaultColor = (255, 255, 0)
    rect_thickness = 5
    rectLineType = cv2.LINE_4
    fontColor = (0, 0, 0)
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    fontThickness = 1
    with av.open(video_path) as input_video:
        frames = list(_get_frames([frame_number], input_video, include_audio=False, audio_buffer_frames=0))
        assert len(frames) == 1
        img = frames[0].to_ndarray(format="bgr24")
        for box in boxes:
            label, bbox = box['object_type'], box['bbox']
            rectColor = colormap.get(label, defaultColor) if label else defaultColor
            x, y, width, height = list(map(lambda x: int(x), [bbox['x'], bbox['y'], bbox['width'], bbox['height']]))
            cv2.rectangle(img, pt1=(x,y), pt2=(x+width, y+height), color=rectColor, thickness=rect_thickness, lineType=rectLineType)
            if label and draw_labels:
                textSize, baseline = cv2.getTextSize(label, fontFace, fontScale, fontThickness)
                textWidth, textHeight = textSize
                cv2.rectangle(img, pt1=(x - rect_thickness//2, y - rect_thickness//2), pt2=(x + textWidth + 10 + rect_thickness, y - textHeight - 10 - rect_thickness), color=rectColor, thickness=-1)
                cv2.putText(img, text=label, org=(x + 10, y - 10), fontFace=fontFace, fontScale=fontScale, color=fontColor, thickness=fontThickness, lineType=cv2.LINE_AA)
    path = f"/tmp/{frame_number}_{str(uuid.uuid1())}.jpg"
    cv2.imwrite(path, img)
    return path

# in: video_path, frames: [{ frame_number, frame_type, boxes: [{ object_type, bbox: {x, y, width, height} }] }]
# out: void; as a side-effect, renders frames from the video with matplotlib
def plot_frames_with_bboxes(video_path, frames, max_cols = 3, fpath=None, key='boxes'):
    cols = max(min(max_cols, len(frames)), 2)
    rows = max(math.ceil(len(frames) / cols), 2)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10*cols, 7 * rows))
    if len(frames) > 1:
        [axi.set_axis_off() for axi in axes.ravel()] # Hide axes
    for idx, frame_data in enumerate(frames):
        row = idx // max_cols
        col = idx % max_cols
        print(frame_data['frame_number'])
        frame_path = render_frame_with_bboxes(video_path, frame_data['frame_number'], frame_data[key])
        # axes[row, col].title.set_text(frame_data['frame_type'])
        axes[row, col].imshow(mpimg.imread(frame_path, format='jpeg'))
    plt.subplots_adjust(wspace=.05, hspace=.05)
    if fpath is None:
        plt.show()
    else:
        print('save to ', fpath)
        plt.savefig(fpath + '.png')
    
# in: video_path, frames: [{ frame_number, frame_label, ?boxes: [{ label, bbox: {x, y, width, height }}] }]
# out: matplotlib.ArtistAnimation of frames rendered with bounding boxes (if provided)
def render_frames_animation(video_path, frames, **kwargs):
    fig, ax = plt.subplots(figsize=(15, 9))
    camera = Camera(fig)
    for frame in frames:
        boxes = frame.get('boxes', [])
        frame_path = render_frame_with_bboxes(video_path, frame['frame_number'], boxes)
        ax.text(0, 1.01, frame['frame_label'], fontsize=20.0, transform=ax.transAxes)
        plt.imshow(mpimg.imread(frame_path, format='jpeg'))
        camera.snap()
    plt.close(fig)
    return camera.animate(**kwargs)

# in: segments: [{<start_key>: int, <end_key>: int}]
# out: void; as a side effect, renders a plot showing all segments passed in
def plot_segments(segments, start_key, end_key):
    ordered_segs = sorted(segments, key=lambda x: x[start_key])
    lines = [[(x[start_key], i), (x[end_key], i)] for i, x in enumerate(ordered_segs)]

    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel('Frame', fontsize=15)
    ax.set_ylabel('Segment', fontsize=15)
    start, end = ax.get_xlim()
    stepsize = (end-start)/30
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))
    plt.show()
    
# in: track: [ [{<start_key>: int, <end_key>: int, <label>: str}] ]
# out: void; as a side effect, renders a plot showing segments of each track passed in
def plot_multitrack_segments(tracks, start_key, end_key, label_key):
    cmap = plt.cm.get_cmap('tab20')
    color_palette = [cmap(x) for x in range(0, 20)]
    
    lines, colors, patches = [], [], []
    for i, segments in enumerate(tracks):
        lines += [[(x[start_key], i), (x[end_key], i)] for x in segments]
        color = color_palette[i % len(color_palette)]
        colors += [color for _ in segments]
        patches += [mpatches.Patch(color=color, label=segments[0][label_key])]

    lc = mc.LineCollection(lines, colors = colors, linewidths=550/len(tracks))
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.legend(handles=patches, loc='upper left')
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel('Frame', fontsize=15)
    ax.set_ylabel('Track', fontsize=15)
    start, end = ax.get_xlim()
    stepsize = (end-start)/30
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))
    plt.show()




def pts_to_time_seconds(pts: int, base: Fraction) -> Fraction:
    return pts * base


def frame_index_to_pts(frame: int, start_pt: int, diff_per_frame: int) -> int:
    return start_pt + frame * diff_per_frame


def pts_difference_per_frame(fps: Fraction, time_base: Fraction) -> int:
    pt = (1 / fps) * (1 / time_base)
    assert pt.denominator == 1, "should be whole number"
    return int(pt)


def _get_frames_pts(
    video_pts_set,  # this has type List[int]
    container: av.container.Container,
    include_audio: bool,
    include_additional_audio_pts: int,
):  # -> Iterable[av.frame.Frame]
    assert len(container.streams.video) == 1

    min_pts = min(video_pts_set)
    max_pts = max(video_pts_set)
    video_pts_set = set(video_pts_set)  # for O(1) lookup
    
    video_stream = container.streams.video[0]
    # import pdb
    # pdb.set_trace()
    fps: Fraction = video_stream.average_rate
    video_base: Fraction = video_stream.time_base
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    # [start, end) time
    clip_start_sec = pts_to_time_seconds(min_pts, video_base)
    clip_end_sec = pts_to_time_seconds(max_pts, video_base)
    print('start sec', clip_start_sec, clip_end_sec)

    # add some additional time for audio packets
    clip_end_sec += max(
        pts_to_time_seconds(include_additional_audio_pts, video_base), 1 / fps
    )

    # --- setup
    streams_to_decode = {"video": 0}
    if (
        include_audio
        and container.streams.audio is not None
        and len(container.streams.audio) > 0
    ):
        assert len(container.streams.audio) == 1
        streams_to_decode["audio"] = 0
        audio_base: Fraction = container.streams.audio[0].time_base

    # seek to the point we need in the video
    # with some buffer room, just in-case the seek is not precise
    seek_pts = max(0, min_pts - 2 * video_pt_diff)
    # video_stream.seek(seek_pts)
    container.seek(seek_pts)
    if "audio" in streams_to_decode:
        assert len(container.streams.audio) == 1
        audio_stream = container.streams.audio[0]
        audio_seek_pts = int(seek_pts * video_base / audio_base)
        audio_stream.seek(audio_seek_pts)

    # --- iterate over video

    # used for validation
    previous_video_pts = None
    previous_audio_pts = None

    for frame in container.decode(**streams_to_decode):

        if isinstance(frame, av.AudioFrame):
            assert include_audio
            # ensure frames are in order
            assert previous_audio_pts is None or previous_audio_pts < frame.pts
            previous_audio_pts = frame.pts

            # pyre-fixme[61]: `audio_base` may not be initialized here.
            audio_time_sec = pts_to_time_seconds(frame.pts, audio_base)

            # we want all the audio frames in this region
            if audio_time_sec >= clip_start_sec and audio_time_sec < clip_end_sec:
                yield frame
            elif audio_time_sec >= clip_end_sec:
                break

        elif isinstance(frame, av.VideoFrame):
            video_time_sec = pts_to_time_seconds(frame.pts, video_base)
            if video_time_sec >= clip_end_sec:
                break

            # ensure frames are in order
            assert previous_video_pts is None or previous_video_pts < frame.pts

            if frame.pts in video_pts_set:
                # check that the frame is in range
                assert (
                    video_time_sec >= clip_start_sec and video_time_sec < clip_end_sec
                ), f"""
                video frame at time={video_time_sec} (pts={frame.pts})
                out of range for time [{clip_start_sec}, {clip_end_sec}]
                """

                yield frame


def _get_frames(
    video_frames,  # this has type List[int]
    container: av.container.Container,
    include_audio: bool,
    audio_buffer_frames: int = 0,
):  # -> Iterable[av.frame.Frame]
    assert len(container.streams.video) == 1

    video_stream = container.streams.video[0]
    video_start: int = video_stream.start_time
    video_base: Fraction = video_stream.time_base
    fps: Fraction = video_stream.average_rate
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    audio_buffer_pts = (
        frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff)
        if include_audio
        else 0
    )

    time_pts_set = [
        frame_index_to_pts(f, video_start, video_pt_diff) for f in video_frames
    ]
    return _get_frames_pts(time_pts_set, container, include_audio, audio_buffer_pts)


if __name__ == '__main__':
    video_path = '/glusterfs/yufeiy2/download_data/EGO4D/ego4d_data/v1/full_scale/d02bd0da-e84e-40b8-81d7-47106f1911ae.mp4'
    with av.open(video_path) as input_video:
        frames = list(_get_frames([20], input_video, include_audio=False, audio_buffer_frames=0))
    