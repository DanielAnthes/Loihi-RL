import atexit
import nengo
from nengo.params import Default
from nengo.probe import Probe


class ProbeMaxLength(nengo.Probe):
    def __init__(
        self,
        target,
        max_length,
        attr=None,
        sample_every=Default,
        synapse=Default,
        solver=Default,
        label=Default,
        seed=Default
    ):
        self.max_length = max_length
        self.curr_idx = 0
        super().__init__(target, attr, sample_every, synapse, solver, label, seed)
    

class ProbeToFile(nengo.Probe):
    def __init__(
        self,
        target,
        path,
        attr=None,
        sample_every=Default,
        synapse=Default,
        solver=Default,
        label=Default,
        seed=Default
    ):
        self.path = path
        self.file = open(path, "w")
        @atexit.register
        def on_end():
            self.file.close()

        super().__init__(target, attr, sample_every, synapse, solver, label, seed)

    def clear(self):
        self.file.close()
        self.file = open(self.path, "w")
        self.file.truncate()


class mSimulator(nengo.Simulator):
    def _probe(self):
        """Copy all probed signals to buffers."""
        self._probe_step_time()

        for probe in self.model.probes:
            period = 1 if probe.sample_every is None else probe.sample_every / self.dt
            if self.n_steps % period < 1:
                tmp = self.signals[self.model.sig[probe]["in"]].copy()
                if type(probe) is ProbeMaxLength:
                    l = probe.max_length
                    if len(self._sim_data[probe]) < l:
                        self._sim_data[probe].append(tmp)
                    else:
                        self._sim_data[probe][probe.curr_idx] = tmp
                        probe.curr_idx = (probe.curr_idx + 1) % l
                elif type(probe) is ProbeToFile:
                    raise NotImplementedError("Release date tbd")
                else:
                    self._sim_data[probe].append(tmp)

    def close(self):
        super().close()
        for probe in self.model.probes:
            if type(probe) is ProbeToFile:
                probe.file.close()

    def clear_probes(self):
        """Clear all probe histories.
        .. versionadded:: 3.0.0
        """
        for probe in self.model.probes:
            if type(probe) is ProbeMaxLength:
                probe.curr_idx = 0
            if type(probe) is ProbeToFile:
                probe.clear()
            self._sim_data[probe] = []
        self.data.reset()  # clear probe cache