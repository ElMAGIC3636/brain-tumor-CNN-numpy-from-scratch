import numpy as np


class Conv:
    def __init__(self, num_filters, filter_size=3, l2_lambda=1e-4):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l2_lambda = l2_lambda
        self.filters = None

    def forward(self, input):
        if input.ndim == 3:
            input = input[:, :, :, np.newaxis]

        self.last_input = input
        n_samples, h, w, c = input.shape

        if self.filters is None:
            # He initialization
            scale = np.sqrt(2.0 / (self.filter_size * self.filter_size * c))
            self.filters = np.random.randn(
                self.num_filters, self.filter_size, self.filter_size, c
            ) * scale

        out_h, out_w = h - self.filter_size + 1, w - self.filter_size + 1
        output = np.zeros((n_samples, out_h, out_w, self.num_filters))

        for i in range(out_h):
            for j in range(out_w):
                im_region = input[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for f in range(self.num_filters):
                    output[:, i, j, f] = np.sum(im_region * self.filters[f], axis=(1,2,3))
        return output

    def backprop(self, d_L_d_out, optimizer):
        n_samples, h, w, c = self.last_input.shape
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_input = np.zeros(self.last_input.shape)

        out_h, out_w = h - self.filter_size + 1, w - self.filter_size + 1

        for i in range(out_h):
            for j in range(out_w):
                im_region = self.last_input[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for f in range(self.num_filters):
                    d_L_d_filters[f] += np.sum(
                        d_L_d_out[:, i, j, f, np.newaxis, np.newaxis, np.newaxis] * im_region,
                        axis=0
                    )
                    d_L_d_input[:, i:i+self.filter_size, j:j+self.filter_size, :] += (
                        d_L_d_out[:, i, j, f, np.newaxis, np.newaxis, np.newaxis] * self.filters[f]
                    )

        d_L_d_filters /= n_samples
        d_L_d_filters += self.l2_lambda * self.filters

        self.filters = optimizer.update(self.filters, d_L_d_filters)
        return d_L_d_input
