from collections import defaultdict

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import multiprocessing
from functools import partial

from utils.trajectory_utils import prediction_output_to_trajectories


def isstring(string_test):
    return isinstance(string_test, str)


def is_path_valid(pathname):
    try:
        if not isstring(pathname) or not pathname: return False
    except TypeError:
        return False
    else:
        return True


def isfolder(pathname):
    """
	if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
	if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
	"""
    if is_path_valid(pathname):
        pathname = os.path.normpath(pathname)
        if pathname == './': return True
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False


def mkdir_if_missing(input_path):
    folder = input_path if isfolder(input_path) else os.path.dirname(
        input_path)
    os.makedirs(folder, exist_ok=True)



def _lineseg_dist(a, b):
    """
    https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    """
    # reduce computation
    if np.all(a == b):
        return np.linalg.norm(-a, axis=1)

    # normalized tangent vector
    d = np.zeros_like(a)
    # assert np.all(np.all(a == b, axis=-1) == np.isnan(ans))
    a_eq_b = np.all(a == b, axis=-1)
    d[~a_eq_b] = (b - a)[~a_eq_b] / np.linalg.norm(b[~a_eq_b] - a[~a_eq_b], axis=-1, keepdims=True)

    # signed parallel distance components
    s = (a * d).sum(axis=-1)
    t = (-b * d).sum(axis=-1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros_like(t)], axis=0)

    # perpendicular distance component
    c = np.cross(-a, d, axis=-1)

    ans = np.hypot(h, np.abs(c))

    # edge case where agent stays still
    ans[a_eq_b] = np.linalg.norm(-a, axis=1)[a_eq_b]

    return ans


def _get_diffs_pred(traj):
    """Same order of ped pairs as pdist.
    Input:
        - traj: (ts, n_ped, 2)"""
    num_peds = traj.shape[1]
    return np.concatenate([np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds - ped_i - 1, 1)) - traj[:, ped_i + 1:]
                           for ped_i in range(num_peds)], axis=1)

def check_collision_per_sample_k2(sample, ped_radius=0.1):
    """sample: (num_peds, ts, 2)"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    collision_0_pred = pdist(sample[0]) < ped_radius * 2
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred_t_bool = np.stack([squareform(cm) for cm in np.concatenate([collision_0_pred[np.newaxis,...], collision_t_pred])])
    collision_mat_pred = squareform(np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.sum(collision_mat_pred, axis=0)

    return n_ped_with_col_pred_per_sample, collision_mat_pred_t_bool


def check_collision_per_sample_no_gt(sample, ped_radius=0.1):
    """sample: (num_peds, ts, 2)"""

    sample = sample.transpose(1, 0, 2)  # (ts, n_ped, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    collision_0_pred = pdist(sample[0]) < ped_radius * 2
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred_t_bool = np.stack([squareform(cm) for cm in np.concatenate([collision_0_pred[np.newaxis,...], collision_t_pred])])
    collision_mat_pred = squareform(np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)

    return n_ped_with_col_pred_per_sample, collision_mat_pred_t_bool


def save_trajectories(trajectory, save_dir, seq_name, frame, suffix=''):
    """Save trajectories in a text file.
    Input:
        trajectory: (np.array/torch.Tensor) Predcited trajectories with shape
                    of (n_pedestrian, future_timesteps, 4). The last elemen is
                    [frame_id, track_id, x, y] where each element is float.
        save_dir: (str) Directory to save into.
        seq_name: (str) Sequence name (e.g., eth_biwi, coupa_0)
        frame: (num) Frame ID.
        suffix: (str) Additional suffix to put into file name.
    """
    fname = f"{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt"
    mkdir_if_missing(fname)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    np.savetxt(fname, trajectory, fmt="%.3f")



def get_noncolliding_samples(eval_stg,
                              scene,
                              timesteps,
                              ph,
                              num_samples=20,
                              min_history_timesteps=7,
                              min_future_timesteps=12,
                              z_mode=False,
                              gmm_mode=False,
                              full_dist=False,
                              max_num_samples=300,
                              collisions_ok=True,
                              collision_rad=0.1):

    num_tries = 0
    MAX_NUM_SAMPLES = max_num_samples
    NUM_SAMPLES_PER_FORWARD = 30
    num_samples_w_cols = {}
    num_samples_gathered = defaultdict(lambda: 0)
    predictions_to_return = {}
    while not num_samples_gathered or np.any(np.array(list(num_samples_gathered.values())) < num_samples):
        num_tries += 1
        predictions, _ = eval_stg.predict(scene,
                                       timesteps,
                                       ph,
                                       num_samples=NUM_SAMPLES_PER_FORWARD,
                                       min_history_timesteps=min_history_timesteps,
                                       min_future_timesteps=min_future_timesteps,
                                       z_mode=z_mode,
                                       gmm_mode=gmm_mode,
                                       full_dist=full_dist)

        if not predictions:
            return None, None

        if collisions_ok:  # if there's only one ped, there are necessarily no collisions
            return predictions, num_samples_w_cols

        for frame, prediction in predictions.items():
            if frame not in predictions_to_return:
                predictions_to_return[frame] = {}

            num_peds = len(prediction.values())
            if num_peds == 1:  # if there's only one ped, there are necessarily no collisions
                num_samples_gathered[frame] = num_samples
                for ped_node, pred in prediction.items():
                    # print(pred[:, :num_samples].shape, f"frame: {frame} ped: {ped_node} should be 20: {pred.shape[1]}")
                    predictions_to_return[frame][ped_node] = pred[:, :num_samples]
                    if predictions_to_return[frame][ped_node].shape[1] != 20:
                        import ipdb; ipdb.set_trace()
                continue
            elif num_samples_gathered.get(frame, 0) == num_samples:
                continue

            pred_arr = np.concatenate(list(prediction.values())).swapaxes(0, 1)

            # compute collisions in parallel
            with multiprocessing.Pool(processes=min(NUM_SAMPLES_PER_FORWARD, multiprocessing.cpu_count())) as pool:
                mask = pool.map(partial(check_collision_per_sample_no_gt, ped_radius=collision_rad), pred_arr)
            # get indices of samples that have 0 collisions
            non_colliding_idx = np.where(~np.any(np.array(list(zip(*mask))[0]).astype(bool), axis=-1))[0]

            if num_tries * NUM_SAMPLES_PER_FORWARD >= MAX_NUM_SAMPLES:
                # if num_zeros > MAX_NUM_ZEROS or num_tries > MAX_NUM_TRIES:
                print(f"frame {frame} with {num_peds} peds: "
                      f"collected {num_tries * NUM_SAMPLES_PER_FORWARD} samples, only {num_samples_gathered[frame]} non-colliding. \n")
                num_samples_w_cols[frame] = (num_peds, num_samples - num_samples_gathered[frame])
                non_colliding_idx = np.arange(num_samples)
            elif non_colliding_idx.shape[0] == 0:
                continue

            # append non-colliding samples to current dict
            for ped_node, pred in prediction.items():
                if ped_node not in predictions_to_return[frame]:
                    predictions_to_return[frame][ped_node] = np.full((pred.shape[0], 0, *pred.shape[2:]), np.nan)
                predictions_to_return[frame][ped_node] = np.concatenate([predictions_to_return[frame][ped_node],
                                                                         pred[:, non_colliding_idx]], axis=1)[:, :num_samples]
                assert predictions_to_return[frame][ped_node].shape[1] <= 20, f"should <= 20: {predictions_to_return[frame][ped_node].shape[1]}"
            num_samples_gathered[frame] = min(num_samples_gathered.get(frame, 0) + non_colliding_idx.shape[0], 20)
            assert predictions_to_return[frame][ped_node].shape[1] == num_samples_gathered[frame], \
                f"{predictions_to_return[frame][ped_node].shape[1]} != {num_samples_gathered[frame]}"

    for frame, preds in predictions_to_return.items():
        assert num_samples_gathered[frame] == 20
        for ped, pred in preds.items():
            assert pred.shape[1] == 20, f"frame: {frame} ped: {ped} should be 20: {pred.shape[1]}"

    return predictions_to_return, num_samples_w_cols


def save_trajectron_prediction(predictions, sequence_name, save_dir,
                               dt,
                               max_hl,
                               ph,
                               prune_ph_to_future,
                               frame_scale=10,
                               start_index=0, ):

    prediction_dict, history_dict, future_dict = prediction_output_to_trajectories(
            predictions,
            dt,
            max_hl,
            ph,
            prune_ph_to_future=prune_ph_to_future,
    )

    if not prediction_dict or prediction_dict is None:
        return None, None, None, None

    formatted_gt_trajectories = format_trajectron_trajectories(
        future_dict,
        frame_scale=frame_scale,
        start_timestep=start_index,
        gt=True)
    formatted_obs_trajectories = format_trajectron_trajectories(
        history_dict,
        frame_scale=frame_scale,
        start_timestep=start_index,
        timesteps_duration=8,
        obs=True)
    formatted_sampled_trajectories = format_trajectron_trajectories(
        prediction_dict, frame_scale=frame_scale, start_timestep=start_index)

    print("formatted_sampled_trajectories:", formatted_sampled_trajectories)
    import ipdb; ipdb.set_trace()
    if formatted_sampled_trajectories is None:
        print("None!")
        return
    save_trajectron_trajectories(formatted_sampled_trajectories,
                                 formatted_obs_trajectories,
                                 formatted_gt_trajectories,
                                 sequence_name=sequence_name,
                                 save_dir=save_dir,
                                 frame_scale=frame_scale,
                                 start_index=start_index,
                                 save_fn=save_trajectories)
    # return formatted_sampled_trajectories, formatted_obs_trajectories, formatted_gt_trajectories


def format_trajectron_trajectories(trajectory,
                                   frame_scale=10,
                                   start_timestep=8,
                                   timesteps_duration=12,
                                   obs=False,
                                   gt=False):
    formatted_trajectories = defaultdict(list)
    for timestep, nodes_dict in trajectory.items():
        if obs:
            start = start_timestep + (timestep + 1 -
                                      timesteps_duration) * frame_scale
            stop = start + timesteps_duration * frame_scale
        else:
            start = start_timestep + (timestep + 1) * frame_scale
            stop = start + timesteps_duration * frame_scale
        formatted_trajectories_per_timestep = defaultdict(list)
        timestep_array = np.arange(start,
                                   stop,
                                   step=frame_scale,
                                   dtype=np.float32)

        for node, node_values in nodes_dict.items():
            # node.value = (1, n_samples, 12, 2)
            track_id_array = np.full((timesteps_duration, ),
                                     node.id,
                                     dtype=np.float32)
            attribute_array = np.stack((timestep_array, track_id_array),
                                       axis=1)
            if gt or obs:
                update_data = np.concatenate(
                    (attribute_array, node_values.astype(np.float32)), axis=1)
                formatted_trajectories_per_timestep[0].append(update_data)
            else:
                for idx, sample in enumerate(np.squeeze(node_values)):
                    update_data = np.concatenate(
                        (attribute_array, sample.astype(np.float32)), axis=1)
                    formatted_trajectories_per_timestep[idx].append(
                        update_data)

        for sample in formatted_trajectories_per_timestep.values():
            formatted_trajectories[timestep].append(np.vstack(sample))

    return formatted_trajectories


def save_trajectron_trajectories(sampled_trajectories,
                                 obs_trajectories,
                                 gt_trajectories,
                                 latent_z=None,
                                 sequence_name='biwi_eth',
                                 save_dir=None,
                                 start_index=780,
                                 frame_scale=10,
                                 save_fn=None):
    for t, sampled_t in sampled_trajectories.items():
        frame_id = t * frame_scale + start_index
        for idx, sample in enumerate(sampled_t):
            save_fn(sample,
                    save_dir,
                    sequence_name,
                    frame_id,
                    suffix=f"/sample_{idx:03d}")
        # Save observation states
        save_fn(obs_trajectories[t][0],
                save_dir,
                sequence_name,
                frame_id,
                suffix="/obs")
        # Save ground truth
        save_fn(gt_trajectories[t][0],
                save_dir,
                sequence_name,
                frame_id,
                suffix="/gt")
