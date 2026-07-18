from corridors.nn.az_selfplay import SelfPlayPool


class _Queue:
    def __init__(self):
        self.items = []
        self.cancelled = False
        self.closed = False

    def put(self, item):
        self.items.append(item)

    def cancel_join_thread(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class _Process:
    def __init__(self, exits_on_join=False):
        self.alive = True
        self.exits_on_join = exits_on_join
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.exits_on_join:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False


def _pool(workers, servers=()):
    pool = SelfPlayPool.__new__(SelfPlayPool)
    pool._closed = False
    pool.workers = list(workers)
    pool.servers = list(servers)
    pool.result_queue = _Queue()
    pool.cmd_queues = [_Queue() for _ in workers]
    pool.request_queues = [_Queue() for _ in servers]
    pool.response_queues = None
    pool.ack_queue = None
    return pool


def test_close_immediately_terminates_busy_workers_and_closes_queues():
    workers = [_Process(), _Process()]
    pool = _pool(workers)
    # Capture queue refs before close(): close() clears pool.* refs so their
    # Finalize hooks can run and release the underlying POSIX semaphores.
    result_queue = pool.result_queue
    cmd_queues = list(pool.cmd_queues)

    pool.close(grace_period=0)

    assert all(worker.terminated for worker in workers)
    assert all(queue.items == [("stop",)] for queue in cmd_queues)
    assert result_queue.closed is True
    assert result_queue.cancelled is True
    assert all(queue.closed and queue.cancelled for queue in cmd_queues)
    # References cleared so GC can run the Queue Finalize hooks and unlink
    # the /dev/shm semaphores before Python's atexit sweep complains.
    assert pool.result_queue is None
    assert pool.cmd_queues == []


def test_close_is_idempotent():
    worker = _Process()
    pool = _pool([worker])
    cmd_queue = pool.cmd_queues[0]

    pool.close(grace_period=0)
    pool.close(grace_period=0)

    assert worker.terminated is True
    assert len(cmd_queue.items) == 1
