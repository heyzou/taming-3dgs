import torch


_enabled = False
_pending = {}
_samples = {
    "ssim_backward": [],
    "rasterizer_backward": [],
    "rasterizer_render_backward": [],
    "rasterizer_preprocess_backward": [],
}


def configure(enabled):
    global _enabled
    _enabled = enabled


def enabled():
    return _enabled


def record_cuda_interval(name, fn, *args, **kwargs):
    if not _enabled:
        return fn(*args, **kwargs)

    # Record GPU intervals without synchronizing inside each measured region.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    _pending[name] = (start, end)
    return result


def finish_iteration():
    if not _enabled:
        return {}

    # The caller synchronizes the total backward event before resolving these.
    elapsed = {}
    for name, (start, end) in _pending.items():
        elapsed[name] = start.elapsed_time(end)
        _samples[name].append(elapsed[name])
    _pending.clear()
    return elapsed


def record_sample(name, value):
    if _enabled:
        _samples[name].append(value)


def samples(name):
    return _samples[name]
