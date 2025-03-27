from pathlib import Path
from re import A, I
import numpy as np
import collections
import supervision as sv
from supervision import Detections
from supervision.detection.utils import box_iou_batch
from supervision.tracker.byte_tracker import matching
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import statistics
from boxmot import DeepOcSort, BoostTrack
from PIL import Image
import argparse
import itertools



def detections2boxes(detections):
  """
  Convert Supervision Detections to numpy tensors for further computation.
  Args:
      detections (Detections): Detections/Targets in the format of sv.Detections.
  Returns:
      (np.ndarray): Detections as numpy tensors as in
          `(x_min, y_min, x_max, y_max, confidence, class_id)` order.
  """
  return np.hstack(
      (
          detections.xyxy,
          detections.confidence[:, np.newaxis],
          detections.class_id[:, np.newaxis],
      )
  )

def update_with_detections(tracker, detections):
 
  tensors = detections2boxes(detections=detections)
  tracks = tracker.update_with_tensors(tensors=tensors)

  if len(tracks) > 0:
    detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
    track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

    ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

    iou_costs = 1 - ious

    matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
    detections.tracker_id = np.full(len(detections), -1, dtype=int)
    for i_detection, i_track in matches:
      detections.tracker_id[i_detection] = int(
          tracks[i_track].external_track_id
      )

    # return all detections, even those with no tracker_id (will be -1)
    # return detections[detections.tracker_id != -1]
    return detections

  else:
    detections = Detections.empty()
    detections.tracker_id = np.array([], dtype=int)

    return detections

def update_with_detections_boxmot(tracker, detections, img):
 
  tensors = detections2boxes(detections=detections)
  tracks = tracker.update(tensors, img)

  if len(tracks) > 0:
    detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
    track_bounding_boxes = np.asarray([track[:4] for track in tracks])

    ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

    iou_costs = 1 - ious

    matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
    detections.tracker_id = np.full(len(detections), -1, dtype=int)
    for i_detection, i_track in matches:
      detections.tracker_id[i_detection] = int(
          tracks[i_track][4]
      )

    # return all detections, even those with no tracker_id (will be -1)
    # return detections[detections.tracker_id != -1]
    return detections

  else:
    detections = Detections.empty()
    detections.tracker_id = np.array([], dtype=int)

    return detections  
  
def convert_annot_to_dets(annots, cats, img_info):
  # person:0, coco cats are index 1 based
  class_id = np.array([annot['category_id']-1 for annot in annots], np.int32)
  class_names = np.array([cats[annot['category_id']]['name'] \
                          for annot in annots])
  tracker_id = np.array([int(annot['attributes']['tracklet_id']) if \
                         annot['attributes']['tracklet_id'] != '-1' else \
                         None for annot in annots])
  confidences = np.array([float(annot['confidence']) \
                         for annot in annots], np.float32)
  
  masks = []
  for annot in annots:
    if 'segmentation' in annot:
      annot_seg = {
        'size': annot['segmentation']['size'],
        'counts': annot['segmentation']['counts'].encode('utf-8')
      }
      decoded_mask = maskUtils.decode(annot_seg)
      if decoded_mask is None or decoded_mask.size == 0:
        decoded_mask = np.zeros([img_info['height'], img_info['width']],
                                dtype=np.bool_)
      masks.append(decoded_mask)
    else:
      blank = np.zeros([img_info['height'], img_info['width']], dtype=np.bool_)
      masks.append(blank)
      
  masks = np.stack(masks).astype(np.bool_) if len(masks) > 0 else None
    
  # convert horz, vert, horz_sz, vert_sz to xyxy
  annots_as_dets = sv.Detections(
    xyxy=np.array([[annot['bbox'][0], annot['bbox'][1],
                   annot['bbox'][2], annot['bbox'][3]] \
                    for annot in annots],
                   dtype=np.float32) if len(annots) > 0 else \
                   np.zeros((0,4), dtype=np.float32),
    confidence=confidences,
    class_id=class_id,
    mask=masks,
    tracker_id=tracker_id,
    data={"class_name":class_names}
  )
  
  return annots_as_dets

def compute_track_metrics(coco_data, fps=None):
  """
  Computes:
      * number of unique people (unique track_ids)
      * min length of track (#frames)
      * min length of track (time)
      * max length of track (#frames)
      * max length of track (time)
      * avg length of track (#frames)
      * avg length of track (time)
      * median length of track (#frames)
      * median length of track (time)

  Args:
      coco_data: A COCO object (from pycocotools.coco import COCO).
      fps: Optional, compute time from frame_id = frame_index / fps.
           Otherwise, assume 'time' is directly available in ann['time'].

  Returns:
      A dictionary of computed metrics.
  """
  # track_info will map track_id -> dict with min/max frame/time
  track_info = collections.defaultdict(lambda: {
    'min_frame': float('inf'),
    'max_frame': float('-inf'),
    'min_time': float('inf'),
    'max_time': float('-inf')
  })

  # collect min/max frame/time for each track
  for ann in coco_data.anns.values():
    track_id = ann['attributes']['track_id']
    
    # Get the frame index
    # If 'frame_id' isn't present, use something else
    frame_id = ann.get('frame_id', ann.get('image_id'))
    
    # Compute or retrieve time
    #  - If ann['time'] exists, use it
    #  - Else, if fps is provided, derive it: time_value = frame_id / fps
    if 'time' in ann:
      time_value = ann['time']
    elif fps is not None:
      time_value = frame_id / fps
    else:
      # No explicit time, no fps
      time_value = None
    
    if frame_id < track_info[track_id]['min_frame']:
      track_info[track_id]['min_frame'] = frame_id
    if frame_id > track_info[track_id]['max_frame']:
      track_info[track_id]['max_frame'] = frame_id

    if time_value is not None:
      if time_value < track_info[track_id]['min_time']:
        track_info[track_id]['min_time'] = time_value
      if time_value > track_info[track_id]['max_time']:
        track_info[track_id]['max_time'] = time_value

  # compute min/max/avg/median track lengths (frames & time)
  unique_people = len(track_info)

  min_track_len = float('inf')
  max_track_len = float('-inf')
  min_track_time = float('inf')
  max_track_time = float('-inf')

  track_lengths_frames = []
  track_lengths_times = []

  for tid, info in track_info.items():
    track_len_frames = info['max_frame'] - info['min_frame'] + 1
    
    # Track length (time)
    if (info['min_time'] == float('inf') and 
        info['max_time'] == float('-inf')):
      track_len_time = 0
    else:
      track_len_time = info['max_time'] - info['min_time']

    if track_len_frames < min_track_len:
      min_track_len = track_len_frames
    if track_len_frames > max_track_len:
      max_track_len = track_len_frames

    if track_len_time < min_track_time:
      min_track_time = track_len_time
    if track_len_time > max_track_time:
      max_track_time = track_len_time

    # collect lengths for avg/median
    track_lengths_frames.append(track_len_frames)
    track_lengths_times.append(track_len_time)

  if unique_people > 0:
    avg_track_len_frames = statistics.mean(track_lengths_frames)
    median_track_len_frames = statistics.median(track_lengths_frames)
    avg_track_len_time = statistics.mean(track_lengths_times)
    median_track_len_time = statistics.median(track_lengths_times)
  else:
    avg_track_len_frames = 0
    median_track_len_frames = 0
    avg_track_len_time = 0
    median_track_len_time = 0
  
  if min_track_len == float('inf'):
    min_track_len = 0
  if max_track_len == float('-inf'):
    max_track_len = 0
  if min_track_time == float('inf'):
    min_track_time = 0
  if max_track_time == float('-inf'):
    max_track_time = 0

  metrics = {
    'num_unique_people': unique_people,
    'min_length_track_frames': min_track_len,
    'max_length_track_frames': max_track_len,
    'min_length_track_time': min_track_time,
    'max_length_track_time': max_track_time,
    'avg_length_track_frames': avg_track_len_frames,
    'median_length_track_frames': median_track_len_frames,
    'avg_length_track_time': avg_track_len_time,
    'median_length_track_time': median_track_len_time,
  }

  return metrics

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def convert_to_serializable(obj):
    """
    Recursively convert NumPy data types (np.int32, np.float32, np.ndarray) 
    into Python-native types so they can be JSON-serialized.
    """
    if isinstance(obj, np.integer):
        return int(obj)  # Convert np.int32, np.int64 -> int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert np.float32, np.float64 -> float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array -> Python list
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}  # Recursively fix dictionaries
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]  # Recursively fix lists
    return obj  # If already serializable, return as is

def compute_max_precision_recall(gt_data,
                                 pred_data,
                                 pred_key='track_id',
                                 gt_key='track_id'):
  """
  Computes 'Maximum Precision' for each predicted track
  and 'Maximum Recall' for each ground-truth ID

  :param gt_data:  COCO-format dict for ground-truth
                   (containing 'annotations').
  :param pred_data: COCO-format dict for predictions
                    (containing 'annotations').
  :param pred_key:  Key under 'attributes' in pred annotations
                    that indicates the *predicted* track ID.
  :param gt_key:    Key under 'attributes' in pred annotations
                    that indicates the *ground-truth* track ID.

  :return: (pred_track_precision, gt_track_recall), both dicts.
  """
  # count how many bounding boxes each ground-truth ID has (for recall denominator)
  gt_id_counts = collections.defaultdict(int)
  for ann in gt_data:
    gt_id = ann['attributes']['track_id']
    gt_id_counts[gt_id] += 1

  # build mappings from each predicted track -> list of GT IDs
  # and from each ground-truth ID -> list of predicted tracks
  # (as they appear in predictions).
  pred_track_to_gt_ids = collections.defaultdict(list)
  gt_id_to_pred_tracks = collections.defaultdict(list)

  for ann_idx in range(len(pred_data)):
    # predicted track ID
    p_tid = pred_data[ann_idx]['attributes'][pred_key]
    # GTuth track ID
    g_tid = gt_data[ann_idx]['attributes'][gt_key]

    pred_track_to_gt_ids[p_tid].append(g_tid)
    gt_id_to_pred_tracks[g_tid].append(p_tid)

  # compute Maximum Precision for each predicted track
  pred_track_precision = {}
  for p_tid, gt_ids in pred_track_to_gt_ids.items():
    if not gt_ids:  # No detections at all for this track
      pred_track_precision[p_tid] = 0.0
      continue

    # count occurrences of each GT ID in this predicted track
    count_map = collections.defaultdict(int)
    for g_id in gt_ids:
      count_map[g_id] += 1

    # max overlap count
    max_count = max(count_map.values())

    # for tie breaks the ratios should be the same..
    pred_track_precision[p_tid] = max_count / len(gt_ids)

  # compute Maximum Recall for each ground-truth ID
  gt_track_recall = {}
  for g_tid, pred_tids in gt_id_to_pred_tracks.items():
    total_gt_dets = gt_id_counts[g_tid]  # how many boxes belong to this GT ID (for denominator)

    if total_gt_dets == 0:
      gt_track_recall[g_tid] = 0.0
      continue

    # count how often each predicted track appears in this GT identity
    count_map = collections.defaultdict(int)
    for p_tid in pred_tids:
      count_map[p_tid] += 1

    max_count = max(count_map.values())

    # no need for tie breaks
    gt_track_recall[g_tid] = max_count / total_gt_dets

  return {
    'pred_track_precision':pred_track_precision,
    'avg_pred_track_precision': np.mean(list(pred_track_precision.values())),
    'gt_track_recall':gt_track_recall,
    'avg_gt_track_recall': np.mean(list(gt_track_recall.values()))
  }


def calc_id_sw(gt_data, pred_data, gap_threshold=1):
  """
  Compute ID-switches given two COCO-style JSON files: ground truth (GT) and predictions.
  
  Args:
      gt_json_path (str): Path to the ground-truth COCO JSON file.
      pred_json_path (str): Path to the predicted COCO JSON file.
      gap_threshold (int): Max gap (in frames) to consider the track continuous.
                           If the gap is larger, do not count a label change as a switch.
  
  Returns:
      dict: {
          "total_id_switches": <int>,
          "id_switches_per_identity": {<gt_id>: <count>, ...},
          "gap_threshold": <int>
      }
  """

  # build dictionaries keyed by annotation 'id'
  #   * GT: ann_id -> (frame_index, ground_truth_ID)
  #   * predictions: ann_id -> predicted_ID
  gt_info = {}
  for ann in gt_data:
    ann_id = ann['id']
    gt_id  = ann['attributes']['track_id']       # ground-truth ID
    frame  = ann['image_id']       # treat image_id as frame index
    gt_info[ann_id] = (frame, gt_id)

  pred_info = {}
  for ann in pred_data:
    ann_id = ann['id']
    pred_id = ann['attributes']['track_id'] 
    pred_info[ann_id] = pred_id

  # group all detections by their ground-truth ID
  #   * for each GT ID, collect a list of (frame_index, predicted_ID)
  gt_tracks = collections.defaultdict(list)
  for ann_id, (frame, gt_id) in gt_info.items():
    if ann_id in pred_info:  # only consider annotations that appear in both
      p_id = pred_info[ann_id]
      gt_tracks[gt_id].append((frame, p_id))

  # count ID switches per ground-truth ID
  total_id_switches = 0
  id_switches_per_gt = collections.defaultdict(int)
  len_per_identity = collections.defaultdict(int)
  for gt_id, entries in gt_tracks.items():
    # Sort by frame (time)
    entries.sort(key=lambda x: x[0])
    if not entries:
      continue

    switches = 0
    for i in range(1, len(entries)):
      # [0]:curr_frame, [1]:curr_pred_id
      switches += int(entries[i][1] != entries[i-1][1])

    len_per_identity[gt_id] = len(entries)
    id_switches_per_gt[gt_id] = switches
    total_id_switches += switches

  results = {
    "total_id_switches": total_id_switches,
    "id_switches_per_identity": dict(id_switches_per_gt),
    'length_per_identity':dict(len_per_identity),
    "gap_threshold": gap_threshold
  }

  return results

def run_tracker_bytetrack(tracker_args=None, output_json_path='output_testing.json'):
    '''
    This is for running internal evals of trackers. In general, the tracker predictions would
    be provided in a COCO object (as a file or after loading) with no access to the GT data. For
    now we loop the GT data and run predictions all in one go.
    '''
    # Convert string paths to Path objects
    gt_paths = [Path(p) for p in ['2021-11-10_lunch_2_post_cam0.json', '2021-11-10_lunch_2_post_cam1.json']]

    uq = collections.defaultdict(list)

    for p in gt_paths:
        mergedp = '_'.join(p.stem.split('_')[:3])  # Extract the first 3 parts of the filename (excluding extension)
        uq[mergedp].append(p)


    if len(uq) == 0:
        raise ValueError(f'No pairs of annots found')
    
    results = {}

    for uq_key, pairs in uq.items():
        print(f'Processing key {uq_key}')
        results[str(uq_key)] = []

        for pair in pairs:
            print(f'\tProcessing pair {pair}')
            next_cam = COCO(pair)
            next_cam_orig = COCO(pair)
            if tracker_args:
              print('Custom Tracker Parameters')
              tracker = sv.ByteTrack(
                track_activation_threshold=tracker_args['track_thresh'],
                lost_track_buffer=tracker_args['track_buffer'],
                minimum_matching_threshold=tracker_args['match_thresh'],
                frame_rate=tracker_args['frame_rate']
              )
            else:
                print('Default Tracker Parameters')
                tracker = sv.ByteTrack(frame_rate=9)

            for i, img_info in next_cam.imgs.items():
                annots = next_cam.imgToAnns.get(img_info['id'], [])
                if len(annots) > 0:
                    dets = convert_annot_to_dets(annots, next_cam.cats, img_info)
                    res_dets = update_with_detections(tracker, dets)

                    # pre-reset IDs
                    for i in range(len(annots)):
                        annots[i]['attributes'] = { 'track_id':-1 }

                    if len(annots) != len(res_dets):
                        pass

                    for didx in range(len(res_dets)): # detections_with_track_id
                        # unassigned ID defaults to -1
                        if res_dets.tracker_id[didx] in [None]:
                            res_dets.tracker_id[didx] = -1

                        # replace the annot
                        annots[didx]['attributes']['track_id'] = res_dets.tracker_id[didx]

            # we have the detection outputs, now do the eval
            stat_res = compute_track_metrics(next_cam_orig, 9)
            metric_res = compute_max_precision_recall(list(next_cam_orig.anns.values()), list(next_cam.anns.values()))
            id_sw_res = calc_id_sw(list(next_cam_orig.anns.values()), list(next_cam.anns.values()))

            # Store results in dictionary
            results[str(uq_key)].append({
              "pair": str(pair),
              "track_metrics": convert_to_serializable(stat_res),
              "precision_recall_metrics": convert_to_serializable(metric_res),
              "id_switches_metrics": convert_to_serializable(id_sw_res)
            })

    # Save results to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
      json.dump(results, json_file, indent=4)
    
    print(f"Tracking results saved to: {output_json_path}")
    

def run_tracker_BoxMot(args=None, tracker_type='BoostTrack',output_json_path='output_testing.json'):
    '''
    This is for running internal evals of trackers. In general, the tracker predictions would
    be provided in a COCO object (as a file or after loading) with no access to the GT data. For
    now we loop the GT data and run predictions all in one go.
    '''
    # Convert string paths to Path objects
    gt_paths = [Path(p) for p in ['/mnt/drv8tb/tempo_remapped/export_output_path/2021-11-10_lunch_2_post_cam0.json', '/mnt/drv8tb/tempo_remapped/export_output_path/2021-11-10_lunch_2_post_cam1.json']]

    #modify to point to where you have the images saved (rel path)
    images_loc = '../images/'

    uq = collections.defaultdict(list)

    for p in gt_paths:
        mergedp = '_'.join(p.stem.split('_')[:3])  # Extract the first 3 parts of the filename (excluding extension)
        uq[mergedp].append(p)


    if len(uq) == 0:
        raise ValueError(f'No pairs of annots found')
    
    results = {}

    for uq_key, pairs in uq.items():
        print(f'Processing key {uq_key}')
        results[str(uq_key)] = []

        for pair in pairs:
            print(f'\tProcessing pair {pair}')
            next_cam = COCO(pair)
            next_cam_orig = COCO(pair)
            if args != None:
              print('Custom Tracker Parameters')
              if tracker_type == 'BoostTrack':
                tracker = BoostTrack(det_thresh=args[0], iou_threshold=args[1], max_age=args[2], reid_weights=args[3], device="0", half=False)
              elif tracker_type == 'DeepOcSort':
                tracker = DeepOcSort(det_thresh=args[0], iou_threshold=args[1], max_age=args[2], reid_weights=args[3], device="0", half=False)
            else:
                print('Default Tracker Parameters')
                tracker = BoostTrack(Path('osnet_x1_0_msmt17.pt'), device="0", half=False)

            for i, img_info in next_cam.imgs.items():
                annots = next_cam.imgToAnns.get(img_info['id'], [])
                if len(annots) > 0:
                    dets = convert_annot_to_dets(annots, next_cam.cats, img_info)
                    image = np.array(Image.open(images_loc+img_info['file_name']))
                    res_dets = update_with_detections_boxmot(tracker, dets, image)

                    # pre-reset IDs
                    for i in range(len(annots)):
                        annots[i]['attributes'] = { 'track_id':-1 }

                    if len(annots) != len(res_dets):
                        pass

                    for didx in range(len(res_dets)): # detections_with_track_id
                        # unassigned ID defaults to -1
                        if res_dets.tracker_id[didx] in [None]:
                            res_dets.tracker_id[didx] = -1

                        # replace the annot
                        annots[didx]['attributes']['track_id'] = res_dets.tracker_id[didx]

            # we have the detection outputs, now do the eval
            stat_res = compute_track_metrics(next_cam_orig, 9)
            metric_res = compute_max_precision_recall(list(next_cam_orig.anns.values()), list(next_cam.anns.values()))
            id_sw_res = calc_id_sw(list(next_cam_orig.anns.values()), list(next_cam.anns.values()))

            # Store results in dictionary
            results[str(uq_key)].append({
              "pair": str(pair),
              "track_metrics": convert_to_serializable(stat_res),
              "precision_recall_metrics": convert_to_serializable(metric_res),
              "id_switches_metrics": convert_to_serializable(id_sw_res)
            })
            data.append({
               'filename': str(pair),
               'track_thresh': tracker_args['track_thresh'],
               'match_thresh': tracker_args['match_thresh'],
               'track_buffer': tracker_args['track_buffer'],
               'avg_precision': avg_precision,
               'avg_recall': avg_recall,
               'total_IDsw': idsw
            })

    '''# Save results to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
      json.dump(results, json_file, indent=4)
    
    print(f"Tracking results saved to: {output_json_path}")'''
    
    return data

if __name__ == '__main__':
  det_thresh_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  Iou_thresh_values = [0.3, 0.5, 0.7]
  max_age_values = [15, 30, 60]
  models = ['osnet_x0_25_market1501.pt','osnet_x1_0_msmt17.pt']
  param_grid = list(itertools.product(
     det_thresh_values,
     Iou_thresh_values,
     max_age_values,
     models
  ))
  
  results = []
  for det_thresh, Iou_thresh, max_age, model in param_grid:
     print(f"Running BoostTrack with: T={det_thresh}, M={Iou_thresh}, B={max_age}, M={model}")

     args = [det_thresh, Iou_thresh, max_age, Path(model)]
     results.extend(run_tracker_BoxMot(args, tracker_type='BoostTrack', output_json_path=f"BoostTrack/detthresh{det_thresh}_iouthresh{Iou_thresh}_maxage{max_age}_model{model}.json"))
   
  df = pd.DataFrame(results)
  df.to_csv("BoostTrack_results.csv", index=False)
  print("CSV saved to: BoostTrack_results.csv")
  
  results = []
  for det_thresh, Iou_thresh, max_age, model in param_grid:
     print(f"Running DeepOcSort with: T={det_thresh}, M={Iou_thresh}, B={max_age}, M={model}")

     args = [det_thresh, Iou_thresh, max_age, Path(model)]
     results.extend(run_tracker_BoxMot(args, tracker_type='DeepOcSort', output_json_path=f"DeepOcSort/detthresh{det_thresh}_iouthresh{Iou_thresh}_maxage{max_age}_model{model}.json"))
   
  df = pd.DataFrame(results)
  df.to_csv("DeepOcSort_results.csv", index=False)
  print("CSV saved to: DeepOcSort_results.csv")