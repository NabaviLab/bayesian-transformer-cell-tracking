from __future__ import annotations
import numpy as np
import pandas as pd

def update_tracking_with_divisions(
    tracking: dict,
    matches,
    frame_idx: int,
    num_frames: int,
    names1: list[str],
    names2: list[str],
    sec1_df,
    sec2_df,
    centroid_x_col: str,
    centroid_y_col: str,
    lineage_info: dict
):
    matched_in_f1, matched_in_f2 = set(), set()

    for m in matches:
        parent, child_or_list = m
        children = child_or_list if isinstance(child_or_list, list) else [child_or_list]
        i = int(parent.split('_')[1])
        pid = int(names1[i].split('_')[0])
        pc = np.array([sec1_df.iloc[i][centroid_x_col], sec1_df.iloc[i][centroid_y_col]], dtype=np.float32)
        c_ids, c_cent = [], []
        for ch in children:
            j = int(ch.split('_')[1])
            cid = int(names2[j].split('_')[0])
            c_ids.append(cid)
            c_cent.append(np.array([sec2_df.iloc[j][centroid_x_col], sec2_df.iloc[j][centroid_y_col]], dtype=np.float32))

        matched_in_f1.add(pid)
        matched_in_f2.update(c_ids)

        if len(children) == 1:
            cid = c_ids[0]
            cent = c_cent[0]
            track_id = None
            for t_id, arr in tracking.items():
                if isinstance(arr[frame_idx], dict) and arr[frame_idx]['cell_id'] == pid:
                    track_id = t_id; break
            if track_id is None:
                track_id = len(tracking)
                tracking[track_id] = [None] * num_frames
                lineage_info[track_id] = {'start': frame_idx, 'end': frame_idx, 'parent': 0}
            tracking[track_id][frame_idx] = {'cell_id': pid, 'centroid': pc}
            tracking[track_id][frame_idx + 1] = {'cell_id': cid, 'centroid': cent}
            lineage_info[track_id]['end'] = frame_idx + 1
        else:
            # division: close parent track and spawn children
            track_id = None
            for t_id, arr in tracking.items():
                if isinstance(arr[frame_idx], dict) and arr[frame_idx]['cell_id'] == pid:
                    track_id = t_id; break
            if track_id is None:
                track_id = len(tracking)
                tracking[track_id] = [None] * num_frames
                tracking[track_id][frame_idx] = {'cell_id': pid, 'centroid': pc}
                lineage_info[track_id] = {'start': frame_idx, 'end': frame_idx, 'parent': 0}
            # terminate parent
            for f in range(frame_idx + 1, num_frames):
                tracking[track_id][f] = None
            lineage_info[track_id]['end'] = frame_idx

            # children tracks
            for cid, ccent in zip(c_ids, c_cent):
                new_id = len(tracking)
                tracking[new_id] = [None] * num_frames
                tracking[new_id][frame_idx + 1] = {'cell_id': cid, 'centroid': ccent}
                lineage_info[new_id] = {'start': frame_idx + 1, 'end': frame_idx + 1, 'parent': track_id}

    # init tracks for unmatched appearing cells at t+1
    all2 = set(int(n.split('_')[0]) for n in names2)
    for cid in (all2 - matched_in_f2):
        t_id = len(tracking)
        tracking[t_id] = [None] * num_frames
        tracking[t_id][frame_idx + 1] = {'cell_id': cid, 'centroid': np.array([0, 0], dtype=np.float32)}
        lineage_info[t_id] = {'start': frame_idx + 1, 'end': frame_idx + 1, 'parent': 0}

    return tracking

def export_lineage_to_mantrack(lineage_info: dict, output_path: str):
    with open(output_path, 'w') as f:
        for tid in sorted(lineage_info.keys()):
            L = tid + 1
            B = lineage_info[tid]['start']
            E = lineage_info[tid]['end']
            P = 0 if lineage_info[tid]['parent'] == 0 else lineage_info[tid]['parent'] + 1
            f.write(f"{L} {B} {E} {P}\n")

def generate_output_csv_with_divisions(tracking: dict, num_frames: int, output_path: str):
    data = []
    for tid, arr in tracking.items():
        row = [tid] + arr[:num_frames]
        data.append(row)
    cols = ["Tracking_ID"] + [f"Frame_{i}" for i in range(num_frames)]
    pd.DataFrame(data, columns=cols).to_csv(output_path, index=False)
