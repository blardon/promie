import math

def get_frame_index_from_seconds(fps, seconds):
    """
    Converts seconds into frame count, based on FPS (frames per second)
    """
    return math.ceil(fps * seconds)

def get_frame_annotation_dict(fps, annotations, fading_time_in_s, last_frame_index):
    next_start_index = 0
    result = []
    last_time = 0
    for annotation, time_in_s in annotations:
        if time_in_s == -1:
            time_in_s = last_frame_index/fps
        assert time_in_s > last_time
        frame_index = get_frame_index_from_seconds(fps, time_in_s)

        length = time_in_s - last_time
        assert length > (fading_time_in_s / 2)

        entry = {
            "annotation": annotation,
            "from_frame": next_start_index,
            "to_frame": frame_index
        }
        result.append(entry)
        next_start_index = frame_index + 1
        last_time = time_in_s
    return result

