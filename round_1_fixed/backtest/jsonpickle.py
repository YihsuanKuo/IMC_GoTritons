import json


def encode(obj):
    return json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True)
