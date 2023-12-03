from network import Network
import random
import copy


class NNGeneticAlgo:
    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate, X, y):
        """
        Khởi tạo một đối tượng NNGeneticAlgo.

        @n_pops: Số lượng cá thể trong quần thể của thuật toán di truyền (Genetic Algorithm).
        @net_size: Kích thước của mỗi mạng neural trong quần thể.
        @mutation_rate: Xác suất đột biến - xác định tỷ lệ phần trăm trọng số và bias bị đột biến.
        @crossover_rate: Xác suất lai ghép - xác định tỷ lệ phần trăm trọng số và bias sẽ được lai ghép giữa các cá thể.
        @retain_rate: Tỷ lệ cá thể tốt nhất sẽ được giữ lại từ mỗi thế hệ.
        @X: Dữ liệu được sử dụng để kiểm thử độ chính xác của mạng neural.
        @y: Nhãn của dữ liệu - sử dụng để kiểm thử độ chính xác của mạng neural.
        """
        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]

    def get_random_point(self, type):
        """
        Trả về một điểm ngẫu nhiên trong mạng neural, có thể là trọng số (weight) hoặc bias.

        @type: Loại điểm, có thể là 'weight' hoặc 'bias'.
        @returns: Tuple chứa (layer_index, point_index) - nếu type là 'weight', point_index sẽ là (row_index, col_index).
        """
        nn = self.nets[0]
        layer_index, point_index = random.randint(0, nn.num_layers - 2), 0
        if type == "weight":
            row = random.randint(0, nn.weights[layer_index].shape[0] - 1)
            col = random.randint(0, nn.weights[layer_index].shape[1] - 1)
            point_index = (row, col)
        elif type == "bias":
            point_index = random.randint(0, nn.biases[layer_index].size - 1)
        return (layer_index, point_index)

    def get_all_scores(self):
        """
        Trả về danh sách các điểm số của mỗi mạng trong quần thể dựa trên dữ liệu đầu vào và nhãn.

        @returns: Danh sách các điểm số.
        """
        return [net.score(self.X, self.y) for net in self.nets]

    def get_all_accuracy(self):
        """
        Trả về danh sách độ chính xác của mỗi mạng trong quần thể dựa trên dữ liệu đầu vào và nhãn.

        @returns: Danh sách độ chính xác.
        """
        return [net.accuracy(self.X, self.y) for net in self.nets]

    def get_predict_results(self, X_sample):
        """
        Trả về dự đoán của mỗi mạng trong quần thể dựa trên một mẫu dữ liệu kiểm thử.

        @X_sample: Mẫu dữ liệu kiểm thử.
        @returns: Danh sách các dự đoán.
        """
        return [net.predict(X_sample) for net in self.nets]

    def crossover(self, father, mother):
        """
        Tạo một cá thể mới dựa trên thông tin di truyền của cha và mẹ.

        @father: Đối tượng neural-net đại diện cho cha.
        @mother: Đối tượng neural-net đại diện cho mẹ.
        @returns: Một đối tượng neural-net mới (con).
        """
        nn = copy.deepcopy(father)

        # Lai ghép bias
        for _ in range(self.nets[0].bias_nitem):
            # Chọn ngẫu nhiên một số điểm
            layer, point = self.get_random_point("bias")
            # Thay thế giá trị genetic (bias) bằng giá trị của mẹ
            if random.uniform(0, 1) < self.crossover_rate:
                nn.biases[layer][point] = mother.biases[layer][point]

        # Lai ghép trọng số
        for _ in range(self.nets[0].weight_nitem):
            # Chọn ngẫu nhiên một số điểm
            layer, point = self.get_random_point("weight")
            # Thay thế giá trị genetic (weight) bằng giá trị của mẹ
            if random.uniform(0, 1) < self.crossover_rate:
                nn.weights[layer][point] = mother.weights[layer][point]

        return nn

    def mutation(self, child):
        """
        Đột biến đối tượng neural-net.

        @child: Đối tượng neural-net cần đột biến trọng số và bias.
        @returns: Đối tượng neural-net mới sau khi đột biến.
        """
        nn = copy.deepcopy(child)

        # Đột biến bias
        for _ in range(self.nets[0].bias_nitem):
            # Chọn ngẫu nhiên một số điểm
            layer, point = self.get_random_point("bias")
            # Thêm một giá trị ngẫu nhiên trong khoảng từ -0.5 đến 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                nn.biases[layer][point] += random.uniform(-0.5, 0.5)

        # Đột biến trọng số
        for _ in range(self.nets[0].weight_nitem):
            # Chọn ngẫu nhiên một số điểm
            layer, point = self.get_random_point("weight")
            # Thêm một giá trị ngẫu nhiên trong khoảng từ -0.5 đến 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)

        return nn

    def evolve(self):
        """
        Tiến hóa quần thể neural-net bằng cách chọn lọc, lai ghép, và đột biến.
        """
        # Tính điểm cho từng mạng trong quần thể
        score_list = list(zip(self.nets, self.get_all_scores()))

        # Sắp xếp các mạng theo điểm số
        score_list.sort(key=lambda x: x[1])

        # Loại bỏ điểm số vì không cần thiết nữa
        score_list = [obj[0] for obj in score_list]

        # Giữ lại chỉ những mạng tốt nhất
        retain_num = int(self.n_pops * self.retain_rate)
        score_list_top = score_list[:retain_num]

        # Giữ lại một số cá thể không tốt nhất
        retain_non_best = int((self.n_pops - retain_num) * self.retain_rate)
        for _ in range(random.randint(0, retain_non_best)):
            score_list_top.append(random.choice(score_list[retain_num:]))

        # Tạo các con mới nếu số lượng cá thể hiện tại ít hơn mong muốn
        while len(score_list_top) < self.n_pops:
            father = random.choice(score_list_top)
            mother = random.choice(score_list_top)

            if father != mother:
                new_child = self.crossover(father, mother)
                new_child = self.mutation(new_child)
                score_list_top.append(new_child)

        # Sao chép quần thể mới vào đối tượng hiện tại
        self.nets = score_list_top
