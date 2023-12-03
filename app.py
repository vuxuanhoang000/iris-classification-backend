from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import OneHotEncoder

from nngenetic_algo import NNGeneticAlgo

import numpy as np
import pandas as pd
import pickle
import time

app = Flask(__name__)
CORS(app)

@app.cli.command("train-model")
def train_model():
    # Đọc dữ liệu từ tệp iris.csv và lưu vào biến X và y
    df = pd.read_csv("iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Chuyển đổi y thành định dạng mã hóa one-hot
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()

    # Các tham số cho thuật toán di truyền và mạng neural
    N_POPS = 30
    NET_SIZE = [4, 6, 5, 3]
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.4
    RETAIN_RATE = 0.4

    # Khởi tạo mạng neural và tối ưu hóa sử dụng thuật toán di truyền
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X, y)

    start_time = time.time()

    # Chạy trong n vòng lặp
    for i in range(150):
        if i % 10 == 0:
            print("Vòng lặp hiện tại: {}".format(i + 1))
            print("Thời gian đã trôi qua: %.1f giây" % (time.time() - start_time))
            print("Số lượng mạng trong quần thể:", len(nnga.nets))
            print("Quần thể hiện tại: ", nnga.nets)
            print("Độ chính xác của mạng hiện tại: \n", nnga.get_all_accuracy())
            print(
                "Độ chính xác của mạng đầu tiên: %.2f%%\n" % nnga.get_all_accuracy()[0]
            )

            # Tìm mạng neural có độ chính xác cao nhất trong quần thể
            best_network = max(nnga.nets, key=lambda net: net.accuracy(X, y))

            # Lấy độ chính xác của mạng neural tốt nhất
            best_accuracy = best_network.accuracy(X, y)
            # In ra thông tin
            print(f"Độ chính xác cao nhất: {best_accuracy:.2f}% \n\n")

        # Tiến hóa quần thể
        nnga.evolve()
        # print(best_network)

    # Lưu mô hình mạng neural vào tập tin
    with open("best_network_model.pkl", "wb") as model_file:
        pickle.dump(best_network, model_file)


# Tên nhãn (điều này cần được biết trước)
label_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


@app.post("/predict")
def iris_prediction():
    try:
        data = request.get_json()

        # Lấy dữ liệu đầu vào từ request
        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]

        # Đọc mô hình từ tập tin khi cần dự đoán
        with open("best_network_model.pkl", "rb") as model_file:
            loaded_model = pickle.load(model_file)

        # Dự đoán với mẫu dữ liệu mới
        new_x_sample = np.array(
            [sepal_length, sepal_width, petal_length, petal_width]
        ).reshape(1, -1)
        probabilities = loaded_model.predict(new_x_sample)[1]

        # Chuyển mảng đầu ra thành mảng 1D
        prediction = probabilities.flatten()

        # Tạo đối tượng JSON để trả về
        result = {
            label: float(probability)
            for label, probability in zip(label_names, prediction)
        }

        return jsonify(result)

    except Exception as error:
        return str(error)
