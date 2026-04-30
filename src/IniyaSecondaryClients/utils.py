import os
import uuid

def to_camel_case(data):
    if isinstance(data, dict):
        return {
            snake_to_camel(k): to_camel_case(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [to_camel_case(i) for i in data]
    return data


def snake_to_camel(s):
    parts = s.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

def get_device_id():
    DEVICE_PATH = os.path.expanduser("~/.iniya/device_id")
    with open(DEVICE_PATH, 'r') as f:
        devid = f.read()
    return devid

def _check_cuda():
    import torch

    if not torch.cuda.is_available():
        return False
    return True
