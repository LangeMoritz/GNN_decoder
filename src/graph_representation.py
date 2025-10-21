import numpy as np
import torch

def sample_syndromes(n_shots, compiled_sampler, device):
    # distinguish between training and testing:
    if compiled_sampler.__class__ == list:
        # sample for each error rate:
        n_trivial_syndromes = 0
        detection_events_list, observable_flips_list = [], []
        n_shots_one_p = n_shots // len(compiled_sampler)
        for sampler in compiled_sampler:
            # repeat each experiments multiple times to get enough non-empty:
            detections_one_p, observable_flips_one_p = [], []
            while len(detections_one_p) < n_shots_one_p:
                detection_events, observable_flips = sampler.sample(
                shots=n_shots_one_p * 10,
                separate_observables=True)
                # sums over the detectors to check if we have a parity change
                shots_w_flips = np.sum(detection_events, axis=1) != 0
                # save only data for measurements with non-empty syndromes
                # but count how many trivial (identity) syndromes we have
                n_trivial_syndromes += np.invert(shots_w_flips).sum()
                detections_one_p.extend(detection_events[shots_w_flips, :])
                observable_flips_one_p.extend(observable_flips[shots_w_flips, :])
            # if there are more non-empty syndromes than necessary
            detection_events_list.append(detections_one_p[:n_shots_one_p])
            observable_flips_list.append(observable_flips_one_p[:n_shots_one_p])
        # interleave lists to mix error rates: 
        # [sample(p1), sample(p2), ..., sample(p_n), sample(p1), sample(p2), ...]
        # instead of [sample(p1), sample(p1), ..., sample(p_2), sample(p2), ...]
        detection_events_list = [val for tup in zip(*detection_events_list) for val in tup]
        observable_flips_list = [val for tup in zip(*observable_flips_list) for val in tup]
    else:
        detection_events_list, observable_flips_list = compiled_sampler.sample(
            shots=n_shots,
            separate_observables=True)
        # sums over the detectors to check if we have a parity change
        shots_w_flips = np.sum(detection_events_list, axis=1) != 0
        # save only data for measurements with non-empty syndromes
        # but count how many trivial (identity) syndromes we have
        n_trivial_syndromes = np.invert(shots_w_flips).sum()
        detection_events_list = detection_events_list[shots_w_flips, :]
        observable_flips_list = observable_flips_list[shots_w_flips, :]

    # make an array from the list:
    detection_events = np.array(detection_events_list)
    observable_flips = np.array(observable_flips_list)
    observable_flips = torch.tensor(observable_flips, dtype=torch.float32).to(device)
    return detection_events, observable_flips, n_trivial_syndromes
