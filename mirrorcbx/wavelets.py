import pywt
import numpa as np

class wavelet_data_term:
    def __init__(self, data, operator, x, wave = None):
        self.data = data[None, None, ...]
        self.operator = operator
        self.wave = pywt.Wavelet('haar') if wave is None else wave
        self.init_coeffs(x)
        
    def init_coeffs(self, x):
        c = pywt.wavedec2(x, wavelet=self.wave, mode='periodization', axes=(-3,-2))
        self.c, self.slices = pywt.coeffs_to_array(c, axes=(-2,-3))
        
    def coeeffs_to_img(self, c):
        return pywt.waverec2(
                pywt.array_to_coeffs(c, self.slices, output_format='wavedec2'), 
                wavelet=self.wave, 
                mode='periodization',
                axes=(-3,-2)
            )
        
        
    def __call__(self, c):
        x = self.coeeffs_to_img(c)
        return self.data_fidelity(x)
    
    def data_fidelity(self, x):
        z = self.operator(x) - self.data
        return 0.5 * np.linalg.norm(z.reshape(z.shape[:2]+ (-1,)), axis=-1)**2