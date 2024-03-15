import numpy as np


class Convolution:
    def __init__(self, data, filter_num, kernel_size) -> None:
        self.data = data
        self.filter_num = filter_num
        self.kernel_height, self.kernel_width = kernel_size

    # 1. conv + padding

    def convolution(self, image, strides: int = 1, padding: bool = False):
        """_summary_

        Args:
            image_matrix (matrix): image matrix. n x n x 3(RGB)
            filters (int): number of filter
            width (int): filter width
            height (int): filter height
            strides (int): stride
        """
        image_channel, image_height, image_width = (
            image.shape[0],
            image.shape[1],
            image.shape[2],
        )

        padding_size = 1 if padding == True else 0

        # 1. filter / output shape
        filters = np.random.random(
            (self.filter_num, image_channel, self.kernel_height, self.kernel_width)
        )

        output_height = int(
            (image_height - self.kernel_height + 2 * padding_size) / strides + 1
        )
        output_width = int(
            (image_width - self.kernel_width + 2 * padding_size) / strides + 1
        )
        output_channel = image_channel

        # output = np.zeros(
        #     (self.filter_num, output_channel, output_height, output_width)
        # )
        output = []
        if padding:
            image = self.zero_padding(image)

        # 2. padding
        print(image.shape)
        # 3. convolution 2d
        for filter in range(self.filter_num):
            output_per_filter = np.zeros((output_channel, output_height, output_width))
            for channel in range(0, output_channel):
                output_per_channel = np.zeros((output_height, output_width))
                for height in range(0, output_height):
                    if (height * strides + self.kernel_height) <= image_height:
                        for width in range(0, output_width):
                            if (width * strides + self.kernel_width) <= image_width:
                                output_per_channel[height][width] = np.sum(
                                    image[
                                        :,
                                        height * strides : height * strides
                                        + self.kernel_height,
                                        width * strides : width * strides
                                        + self.kernel_width,
                                    ]
                                    * filters[filter]
                                ).astype(np.float32)
                output_per_filter[channel, :, :] = output_per_channel
            output.append(output_per_filter)
            output_np = np.vstack(output)
        return output_np

    def ReLU(self, matrix):
        len_w = len(matrix[0][0])
        len_h = len(matrix[0])
        channel = len(matrix)

        for l in range(channel):
            for r in range(len_h):
                for c in range(len_w):
                    if matrix[l][r][c] < 0:
                        matrix[l][r][c] = 0

        return matrix

    def max_pooling(matrix):
        pass

    def zero_padding(self, image_matrix):
        len_w = len(image_matrix[0][0])
        img_channel = len(image_matrix)

        target_x = len_w + 2

        padding_matrix = []

        for l in range(img_channel):  # img channel (한 겹)
            layer = []
            layer.append([0 for _ in range(target_x)])  # 위 padding

            for row in image_matrix[l]:
                p_row = [0]
                p_row.extend(row)
                p_row.append(0)
                layer.append(p_row)

            layer.append([0 for _ in range(target_x)])  # 아래 padding

            padding_matrix.append(layer)

        return padding_matrix


# test_matrix = [
#     [
#         [1, 2, 3],
#         [4, -9, 6],
#         [-3, 8, 9],
#     ],
#     [
#         [1, 2, 3],
#         [4, -9, 6],
#         [-3, 8, 9],
#     ],
#     [
#         [1, 2, 3],
#         [4, -9, 6],
#         [-3, 8, 9],
#     ],
# ]

# Conv = Convolution(test_matrix, 4, (2, 2))

# zero = Conv.zero_padding(test_matrix)
# print(zero)
# print()
# print(Conv.ReLU(zero))
