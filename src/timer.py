import time

timers = {}


def start(key):
    print('[Timer] Start: ' + key)
    timers[key] = time.time()


def end(key):
    diff = time.time() - timers[key]
    print('[Timer] End: ' + key + ' in %.2fs' % diff)
