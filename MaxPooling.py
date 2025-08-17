import numpy as np 
class MaxPooling:
    def forward(self, input):
        self.last_input = input
        n_samples, h, w, c = input.shape
        new_h, new_w = h // 2, w // 2
        output = np.zeros((n_samples, new_h, new_w, c))

        for i in range(new_h):
            for j in range(new_w):
                im_region = input[:, i*2:i*2+2, j*2:j*2+2, :]
                output[:, i, j, :] = np.amax(im_region, axis=(1, 2))
        return output

    def backprop(self, d_L_d_out):
        n_samples, h, w, c = self.last_input.shape
        d_L_d_input = np.zeros(self.last_input.shape)
        new_h, new_w = h // 2, w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = self.last_input[:, i*2:i*2+2, j*2:j*2+2, :]
                amax_mask = (im_region == np.amax(im_region, axis=(1, 2), keepdims=True))
                d_L_d_input[:, i*2:i*2+2, j*2:j*2+2, :] = amax_mask * d_L_d_out[:, i:i+1, j:j+1, :]
        return d_L_d_input
