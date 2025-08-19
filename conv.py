import numpy as np


class Conv:
    def __init__(self, num_filters, filter_size=3, l2_lambda=1e-4): # lambda for preventing overfitting
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l2_lambda = l2_lambda
        self.filters = None

    def forward(self, x_input):
        if x_input.ndim == 3:    # For grayscale image
            x_input = x_input[:, :, :, np.newaxis]  # newaxis here for the broadcasting

        self.last_input = x_input
        batch_size, height, width, channels = x_input.shape

        if self.filters is None:
            scale = np.sqrt(2.0 / (self.filter_size * self.filter_size * channels))    # He initialization: keeps weights scaled to avoid vanishing/exploding gradients with ReLU
            self.filters = np.random.randn(
                self.num_filters, self.filter_size, self.filter_size, channels
            ) * scale

        out_h, out_w = height - self.filter_size + 1, width - self.filter_size + 1
        conv_out = np.zeros((batch_size, out_h, out_w, self.num_filters))

        for i in range(out_h):
            for j in range(out_w):
                region = x_input[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for f in range(self.num_filters):
                    conv_out[:, i, j, f] = np.sum(region * self.filters[f], axis=(1, 2, 3))
        return conv_out  # now we have the feature map 

    def backprop(self, grad_output, optimizer):
        batch_size, height, width, channels = self.last_input.shape
        grad_filters = np.zeros(self.filters.shape)
        grad_input = np.zeros(self.last_input.shape)

        out_h, out_w = height - self.filter_size + 1, width - self.filter_size + 1

        for i in range(out_h):
            for j in range(out_w):
                region = self.last_input[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for f in range(self.num_filters):
                    grad_filters[f] += np.sum(
                        grad_output[:, i, j, f, np.newaxis, np.newaxis, np.newaxis] * region,
                        axis=0
                    )  # sum over the batch dimension (axis=0) to combine gradients from all samples

            
                    grad_input[:, i:i+self.filter_size, j:j+self.filter_size, :] += (
                        grad_output[:, i, j, f, np.newaxis, np.newaxis, np.newaxis] * self.filters[f]
                    ) 

        grad_filters /= batch_size
        grad_filters += self.l2_lambda * self.filters # reducing overfitting 

        self.filters = optimizer.update(self.filters, grad_filters)
        return grad_input
