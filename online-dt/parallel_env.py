import multiprocessing as mp

class SimpleParallelEnv:
    def __init__(self, env_fns):
        self.pipes = [mp.Pipe() for _ in range(len(env_fns))]
        self.processes = [mp.Process(target=self._worker, args=(p[1], fn)) for p, fn in zip(self.pipes, env_fns)]
        self.parent_conns = [p[0] for p in self.pipes]
        for p in self.processes:
            p.start()

    def _worker(self, conn, env_fn):
        env = env_fn()
        while True:
            cmd, data = conn.recv()
            if cmd == "reset": conn.send(env.reset())
            elif cmd == "step": conn.send(env.step(data))
            elif cmd == "close": env.close(); conn.close(); break

    def reset(self):
        [c.send(("reset", None)) for c in self.parent_conns]
        return [c.recv() for c in self.parent_conns]

    def step(self, actions):
        [c.send(("step", a)) for c, a in zip(self.parent_conns, actions)]
        return zip(*[c.recv() for c in self.parent_conns])

    def close(self):
        [c.send(("close", None)) for c in self.parent_conns]
        [p.join() for p in self.processes]