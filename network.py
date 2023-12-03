import numpy as np


class Network(object):
    def __init__(self, sizes):
        """
        Khởi tạo một mạng neural với biases và weights ngẫu nhiên.

        Tham số:
        - sizes: Danh sách các số nguyên, mỗi số nguyên đại diện cho số lượng neuron trong một tầng.
        """
        # Số lượng tầng trong mạng
        self.num_layers = len(sizes)
        # Kích thước của mỗi tầng
        self.sizes = sizes
        # Khởi tạo biases ngẫu nhiên cho mỗi tầng (trừ tầng đầu vào)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Khởi tạo weights ngẫu nhiên cho các kết nối giữa các tầng
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Biến hỗ trợ cho các tính toán sau này
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum(
            [self.weights[i].size for i in range(self.num_layers - 2)]
        )

    def feedforward(self, a):
        """
        Trả về đầu ra của mạng cho một đầu vào cho trước.

        Tham số:
        - a: Dữ liệu đầu vào

        Trả về:
        - Đầu ra của mạng
        """
        for b, w in zip(self.biases, self.weights):
            # Áp dụng hàm kích hoạt sigmoid cho tổng có trọng số của các đầu vào
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sigmoid(self, z):
        """
        Hàm kích hoạt sigmoid.

        Tham số:
        - z: Đầu vào cho hàm sigmoid

        Trả về:
        - Kết quả của hàm sigmoid
        """
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, X, y):
        """
        Tính điểm (score) của các dự đoán của mạng trên dữ liệu kiểm thử.

        Tham số:
        - X: Dữ liệu kiểm thử
        - y: Nhãn thực cho dữ liệu kiểm thử

        Trả về:
        - Điểm của các dự đoán của mạng (càng thấp càng tốt)
        """
        total_score = 0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1, 1))
            actual = y[i].reshape(-1, 1)
            # Tính mean-squared error
            total_score += np.sum(np.power(predicted - actual, 2) / 2)
        return total_score

    def accuracy(self, X, y):
        """
        Tính độ chính xác của các dự đoán của mạng trên dữ liệu kiểm thử.

        Tham số:
        - X: Dữ liệu kiểm thử
        - y: Nhãn thực cho dữ liệu kiểm thử

        Trả về:
        - Độ chính xác (%) (càng cao càng tốt)
        """
        accuracy = 0
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1, 1))
            # So sánh chỉ số của lớp có xác suất cao nhất trong output với chỉ số tương ứng trong nhãn y
            accuracy += int(np.argmax(output) == np.argmax(y[i]))
        return accuracy / X.shape[0] * 100

    def predict(self, X_sample):
        """
        Dự đoán lớp của mẫu dữ liệu cho trước.

        Tham số:
        - X_sample: Dữ liệu mẫu để kiểm thử.

        Trả về:
        - Dự đoán về lớp của mẫu dữ liệu.
        """
        # Dự đoán đầu ra của mạng neural cho mẫu kiểm thử
        probabilities = self.feedforward(X_sample.reshape(-1, 1))
        # Chỉ số của lớp có xác suất cao nhất trong output
        prediction = np.argmax(probabilities)
        return prediction, probabilities

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s
