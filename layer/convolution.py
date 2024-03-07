class Convolution:
    # 1. conv + padding

    def convolution(self, image, filterNum: int, width: int, height: int, strides: int):
        """_summary_

        Args:
            image_matrix (matrix): image matrix. n x n x 3(RGB)
            filters (int): number of filter
            width (int): filter width
            height (int): filter height
            strides (int): stride
        """
        # 1. padding

        matrix = self.zero_padding(image)
        # 2. convolution
        filters = None  # get filter든 뭐든 get_filter(filterNum=filterNum, size=width)

        channel_num = len(image)
        len_w = len(image[0][0])
        len_h = len(image[0])

        conv_layer = []

        for filter in filters:  # filter = w x h x 3 / image = n x n x 3
            output_layer = []
            s_x, s_y = 0, 0
            while not (s_x + width >= len_w and s_y + height >= len_h):
                for channel in range(channel_num):
                    layersum = 0
                    for y in range(height):
                        for x in range(width):
                            image[channel][s_y + y][s_x + x] * filter[y][x]

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


test_matrix = [
    [
        [1, 2, 3],
        [4, -9, 6],
        [-3, 8, 9],
    ],
]

Conv = Convolution()
zero = Conv.zero_padding(test_matrix)
print(zero)
print()
print(Conv.ReLU(zero))
