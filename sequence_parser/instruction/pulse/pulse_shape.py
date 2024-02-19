import copy
import numpy as np

class PulseShape:
    def __init__(self):
        pass

    def set_params(self):
        raise NotImplementedError()

    def model_func(self, time):
        raise NotImplementedError()

class SquareShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.duration = pulse.duration

    def model_func(self, time):
        waveform = self.amplitude*np.ones(time.size)
        return waveform

class StepShape(PulseShape):
    def __init__(self):
        super().__init__()
        
    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.edge = pulse.tmp_params["edge"]
        self.duration = pulse.duration
        
    def model_func(self, time):
        ftime = time[np.where(time <= -0.5*self.edge)]
        btime = time[np.where(time >= +0.5*self.edge)]
        fwaveform = np.zeros(ftime.size)
        bwaveform = self.amplitude*np.ones(btime.size)
        mwaveform = np.linspace(0, self.amplitude, time.size - ftime.size - btime.size, dtype=np.complex128)
        waveform = np.hstack([fwaveform, mwaveform, bwaveform])
        return waveform

class GaussianShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.fwhm = pulse.tmp_params["fwhm"]
        self.zero_end = pulse.tmp_params["zero_end"]
        self.duration = pulse.duration

    def model_func(self, time):
        waveform = self.amplitude*np.exp(-4*np.log(2)*(time/self.fwhm)**2)
        if self.zero_end and abs(self.amplitude) > 0:
            edge = self.amplitude*np.exp(-4*np.log(2)*(0.5*self.duration/self.fwhm)**2)
            waveform = self.amplitude*(waveform - edge)/(self.amplitude - edge)
        return waveform

class GaussianShape2(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.fwhm = pulse.tmp_params["fwhm"]
        self.zero_end = pulse.tmp_params["zero_end"]
        self.duration = pulse.duration

    def model_func(self, time):
        waveform = self.amplitude*np.exp(-4*np.log(2)*(time/self.fwhm)**2)
        if self.zero_end and abs(self.amplitude) > 0:
            edge = self.amplitude*np.exp(-4*np.log(2)*(0.5*self.duration/self.fwhm)**2)
            waveform = self.amplitude*(waveform - edge)/(self.amplitude - edge)
        return waveform

class RaisedCosShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.duration = pulse.duration

    def model_func(self, time):
        phase = np.pi*time/(0.5*self.duration)
        waveform = 0.5*self.amplitude*(1 + np.cos(phase))
        return waveform

class HyperbolicSecantShape(PulseShape):
    def __init__(self):
        super().__init__()
        
    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.zero_end = pulse.tmp_params["zero_end"]
        self.fwhm = pulse.tmp_params["fwhm"]
        self.duration = pulse.duration
        
    def model_func(self, time):
        waveform = self.amplitude/np.cosh(2*np.log(2+3**0.5)/self.fwhm*time)
        if self.zero_end and abs(self.amplitude) > 0:
            edge = self.amplitude/np.cosh(2*np.log(2+3**0.5)/self.fwhm*0.5*self.duration)
            waveform = self.amplitude*(waveform - edge)/(self.amplitude - edge)
        if self.amplitude == 0:
            waveform = 0*time
        return waveform
    
class HalfDRAGShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.pulseshape = copy.deepcopy(pulse.insts[0].pulse_shape)
        self.beta = pulse.tmp_params["beta"]

    def model_func(self, time):
        tmp = self.pulseshape.model_func(time)
        waveform = tmp - 1j*self.beta*np.gradient(tmp)/np.gradient(time)
        return waveform

class DRAG5thShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.pulseshape = copy.deepcopy(pulse.insts[0].pulse_shape)
        self.beta = pulse.tmp_params["beta"]

    def model_func(self, time):
        lam = np.sqrt(2)
        delta = 1/self.beta # -0.15 # 
        tmp = self.pulseshape.model_func(time)

        ex = tmp + ((lam**2-4)*tmp**3)/8/delta**2
        ex -= ((13*lam**4-76*lam**2+112)*tmp**5)/128/delta**4

        dex0 = np.gradient(tmp)/np.gradient(time)
        ey = -dex0/delta + 33*(lam**2-2)*(tmp**2)*dex0
        
        del_t = ((lam**2-4)*tmp**2)/4/delta - ((lam**4-7*lam**2+12)*tmp**4)/16/delta**3
        int_del_t = np.cumsum(del_t*np.gradient(time))
        del_ave = int_del_t[-1]/(time[-1]-time[0])
        theta = np.cumsum((del_t-del_ave)*np.gradient(time))

        Ex = ex*np.cos(theta) - ey*np.sin(theta)
        Ey = ey*np.cos(theta) + ex*np.sin(theta)
        print(del_ave)
        waveform = Ex + 1j*Ey
        return waveform

class WAHWAH1Shape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.pulseshape = copy.deepcopy(pulse.insts[0].pulse_shape)
        self.beta = pulse.tmp_params["beta"]
        self.fm = pulse.tmp_params["fm"]
        self.am = pulse.tmp_params["am"]

    def model_func(self, time):
        tmp = self.pulseshape.model_func(time)
        tg = time[-1] - time[0]
        wahwah = 1 - self.am*np.cos(2*np.pi*self.fm*(time - tg/2))
        tmp = tmp*wahwah
        waveform = tmp - 1j*self.beta*np.gradient(tmp)/np.gradient(time)
        return waveform
    
class FlatTopShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.pulseshape = copy.deepcopy(pulse.insts[0].pulse_shape)
        self.top_duration = pulse.tmp_params["top_duration"]

    def model_func(self, time):
        ftime = time[np.where(time <= -0.5*self.top_duration)] + 0.5*self.top_duration
        btime = time[np.where(time >= +0.5*self.top_duration)] - 0.5*self.top_duration
        fwaveform = self.pulseshape.model_func(ftime)
        bwaveform = self.pulseshape.model_func(btime)
        mwaveform = self.pulseshape.amplitude*np.ones(time.size - ftime.size - btime.size)
        waveform = np.hstack([fwaveform, mwaveform, bwaveform])
        return waveform

class CRABShape(PulseShape):
    def __init__(self):
        super().__init__()
        
    def set_params(self, pulse):
        self.envelope_shape = copy.deepcopy(pulse.insts[0].pulse_shape)
        self.coefficients = pulse.coefficients
        self.polynominals = pulse.polynominals
        
    def model_func(self, time):
        envelope = self.envelope_shape.model_func(time)
        distortion = 0j
        for coeff, func in zip(self.coefficients, self.polynominals):
            distortion += coeff*func(time/max(abs(time)))
        waveform = distortion * envelope
        return waveform

class DeriviativeShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.pulseshape = copy.deepcopy(pulse.insts[0].pulse_shape)

    def model_func(self, time):
        waveform = self.pulseshape.model_func(time)
        return np.gradient(waveform)/np.gradient(time)

class ProductShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.pulseshape_a = copy.deepcopy(pulse.insts[0].pulse_shape)
        self.pulseshape_p = copy.deepcopy(pulse.insts[1].pulse_shape)

    def model_func(self, time):
        waveform_a = self.pulseshape_a.model_func(time)
        waveform_p = self.pulseshape_p.model_func(time)
        waveform = waveform_a * np.exp(1j*np.pi*waveform_p)
        return waveform


class PolynomialRaisedCosShape(PulseShape):
    def __init__(self):
        super().__init__()

    def set_params(self, pulse):
        self.amplitude = pulse.tmp_params["amplitude"]
        self.coeffs = pulse.tmp_params["coefficients"]
        self.duration = pulse.duration

    def model_func(self, time):
        a = 0
        for key in self.coeffs:
            istr = copy.deepcopy(key)
            i = istr.replace('c', '')
            a += self.coeffs[key] * (time)**int(i)

        return np.cos(
            np.pi * time / self.duration)**2 * self.amplitude * a 
