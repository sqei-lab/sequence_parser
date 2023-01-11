from .instruction import Instruction

class Trigger(Instruction):
    def __init__(self, align="left", marker=None):
        super().__init__()
        self.align = align
        self.trigger_index = None
        self.params = {"align" : align}
        self.marker = marker