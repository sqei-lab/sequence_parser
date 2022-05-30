import copy
from typing import Callable

import numpy as np

from .instruction.acquire import Acquire
from .instruction.pulse.pulse import Pulse
from .port import Port


class IQPort(Port):
    """A Port which compensates for the amplitude and delay imbalances of an IQ mixer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i_factor = lambda freq: 1
        self.q_factor = lambda freq: 1
        self.i_delay = lambda freq: 0
        self.q_delay = lambda freq: 0
        self.i_phas_offset = lambda freq: 0
        self.q_phas_offset = lambda freq: 0

    def set_i_factor(self, i_factor: Callable[[float], float]):
        """multiply I waveform by `i_factor(if_freq)`"""
        # self.i_factor = i_factor
        self.i_factor = lambda freq: i_factor

    def set_q_factor(self, q_factor: Callable[[float], float]):
        """multiply Q waveform by `q_factor(if_freq)`"""
        # self.q_factor = q_factor
        self.q_factor = lambda freq: q_factor

    def set_i_delay(self, i_delay: Callable[[float], float]):
        """delay I waveform by `i_delay(if_freq)` ns"""
        # self.i_delay = i_delay
        self.i_delay = lambda freq: i_delay

    def set_q_delay(self, q_delay: Callable[[float], float]):
        """delay Q waveform by `q_delay(if_freq)` ns"""
        # self.q_delay = q_delay
        self.q_delay = lambda freq: q_delay

    def set_i_phas_offset(self, i_phas_offset: Callable[[float], float]):
        """offset phase I waveform by `i_phas(if_freq)` ns"""
        # self.i_phas = i_phas
        self.i_phas_offset = lambda freq: i_phas_offset

    def set_q_phas_offset(self, q_phas_offset: Callable[[float], float]):
        """offset phase Q waveform by `q_phas(if_freq)` ns"""
        # self.q_phas = q_phas
        self.q_phas_offset = lambda freq: q_phas_offset

    def _write_waveform(self, waveform_length):
        """Write waveform by the Pulse instructions
        Args:
            waveform_length (float): total waveform time length
        """
        self.time = np.arange(0, waveform_length, self.DAC_STEP)
        i_waveform = np.zeros(self.time.size, dtype=np.complex128)
        q_waveform = np.zeros_like(i_waveform)
        for instruction in self.syncronized_instruction_list:
            if isinstance(instruction, Pulse):
                # get the compensation parameters at the IF frequency of the pulse
                if_freq = self.if_freq + self.detuning
                i_factor = self.i_factor(if_freq)
                q_factor = self.q_factor(if_freq)
                i_delay = self.i_delay(if_freq)
                q_delay = self.q_delay(if_freq)
                i_phas_offset = self.i_phas_offset(if_freq)
                q_phas_offset = self.q_phas_offset(if_freq)
                instruction._write(self, out=i_waveform, delay=i_delay,
                                   factor=i_factor, phas_offset=i_phas_offset)
                instruction._write(self, out=q_waveform, delay=q_delay,
                                   factor=q_factor, phas_offset=q_phas_offset)
            elif isinstance(instruction, Acquire):
                instruction._acquire(self)

        self.waveform = i_waveform.real + 1j * q_waveform.imag
