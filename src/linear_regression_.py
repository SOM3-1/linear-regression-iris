import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
DATA_PATH = os.path.join("data", "iris.data")
IMG_DIR = "images"

COST_GRAPH = os.path.join(IMG_DIR, "cost_vs_iterations.png")
SCATTER_GRAPH = os.path.join(IMG_DIR, "data_scatter_petal.png")
LINE_INPUTS_GRAPH = os.path.join(IMG_DIR, "initial_model_slice.png")
MODEL_GRAPH = os.path.join(IMG_DIR, "trained_model_slice.png")

LABEL_MAP = {
    "Iris-setosa": -1.0,
    "Iris-versicolor": 0.0,
    "Iris-virginica": 1.0,
}

LABEL_NAME = {
    -1.0: "Iris-setosa",
     0.0: "Iris-versicolor",
     1.0: "Iris-virginica",
}

LABEL_SET = np.array([-1.0, 0.0, 1.0], dtype=float)

def ensure_dataset():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

def load_iris(path):
    X_rows = []
    y_rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            feat = parts[:4]
            cls = parts[4]
            if cls not in LABEL_MAP:
                continue
            X_rows.append([float(v) for v in feat])
            y_rows.append(LABEL_MAP[cls])
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    return X, y

def stratified_split_120_30(X, y, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for lab in LABEL_SET:
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:40].tolist())
        test_idx.extend(idx[40:50].tolist())
    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def minmax_fit(X_train):
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    return x_min, denom

def minmax_transform(X, x_min, denom):
    return (X - x_min) / denom

def add_bias_column(X):
    return np.c_[np.ones((X.shape[0], 1), dtype=float), X]

def predict(Xb, w):
    return Xb @ w

def mse_cost(y_hat, y):
    m = y.shape[0]
    diff = y_hat - y
    return (diff @ diff) / (2.0 * m)

def gradient(Xb, y_hat, y):
    m = y.shape[0]
    return (Xb.T @ (y_hat - y)) / m

def gradient_descent(Xb, y, alpha=0.05, iterations=3000):
    w = np.zeros(Xb.shape[1], dtype=float)
    cost_history = []
    for _ in range(iterations):
        y_hat = predict(Xb, w)
        cost_history.append(mse_cost(y_hat, y))
        grad = gradient(Xb, y_hat, y)
        w = w - alpha * grad
    return w, cost_history

def nearest_label(y_hat):
    diffs = np.abs(y_hat.reshape(-1, 1) - LABEL_SET.reshape(1, -1))
    idx = np.argmin(diffs, axis=1)
    return LABEL_SET[idx]

def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))

def cost_graph(cost_history, out_path):
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost vs Iterations")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def petal_scatter_graph(X, y, out_path):
    os.makedirs(IMG_DIR, exist_ok=True)
    petal_len = X[:, 2]
    petal_wid = X[:, 3]

    plt.figure()
    for lab in LABEL_SET:
        idx = np.where(y == lab)[0]
        plt.scatter(petal_len[idx], petal_wid[idx], label=f"Class {int(lab)}")
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.title("Iris Data: Petal Length vs Petal Width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def line_inputs_graph(X_train, x_min, denom, feature_index=2, points=200):
    means = X_train.mean(axis=0)
    x_feat_min = X_train[:, feature_index].min()
    x_feat_max = X_train[:, feature_index].max()
    x_line = np.linspace(x_feat_min, x_feat_max, points)

    X_line = np.tile(means, (points, 1))
    X_line[:, feature_index] = x_line

    X_line_n = minmax_transform(X_line, x_min, denom)
    Xb_line = add_bias_column(X_line_n)
    return x_line, Xb_line

def plot_slice_with_model(X_train, y_train, x_min, denom, w, out_path, title, feature_index=2):
    os.makedirs(IMG_DIR, exist_ok=True)
    x_vals = X_train[:, feature_index]
    y_vals = y_train

    x_line, Xb_line = line_inputs_graph(X_train, x_min, denom, feature_index=feature_index, points=200)
    y_line = predict(Xb_line, w)

    plt.figure()
    plt.scatter(x_vals, y_vals, alpha=0.6)
    plt.plot(x_line, y_line)
    plt.xlabel("Petal length")
    plt.ylabel("Numeric label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def test_samples(samples, x_min, denom, w):
    print("\nManual test samples:")
    for s in samples:
        x = np.array(s, dtype=float).reshape(1, -1)
        x_n = minmax_transform(x, x_min, denom)
        x_b = add_bias_column(x_n)
        y_hat = predict(x_b, w)[0]
        y_cls = nearest_label(np.array([y_hat]))[0]
        class_name = LABEL_NAME[y_cls]
        print(
            f"Input: {s} -> regression output: {y_hat:.3f}, "
            f"predicted class: {int(y_cls)} ({class_name})"
        )
def main():
    ensure_dataset()
    X, y = load_iris(DATA_PATH)

    if X.shape != (150, 4) or y.shape != (150,):
        raise RuntimeError(f"Unexpected dataset shape: X={X.shape}, y={y.shape}")

    os.makedirs(IMG_DIR, exist_ok=True)
    petal_scatter_graph(X, y, SCATTER_GRAPH)

    X_train, y_train, X_test, y_test = stratified_split_120_30(X, y, seed=42)

    x_min, denom = minmax_fit(X_train)
    X_train_n = minmax_transform(X_train, x_min, denom)
    X_test_n = minmax_transform(X_test, x_min, denom)

    Xb_train = add_bias_column(X_train_n)
    Xb_test = add_bias_column(X_test_n)

    w_init = np.zeros(Xb_train.shape[1], dtype=float)
    plot_slice_with_model(
        X_train, y_train, x_min, denom, w_init,
        LINE_INPUTS_GRAPH,
        "Initial Model Slice (Before Training): Petal Length vs Label",
        feature_index=2
    )

    w, cost_history = gradient_descent(Xb_train, y_train, alpha=0.05, iterations=3000)

    y_hat_test = predict(Xb_test, w)
    y_pred_test = nearest_label(y_hat_test)
    acc = accuracy(y_pred_test, y_test)

    plot_slice_with_model(
        X_train, y_train, x_min, denom, w,
        MODEL_GRAPH,
        "Trained Model Slice (After Gradient Descent): Petal Length vs Label",
        feature_index=2
    )

    cost_graph(cost_history, COST_GRAPH)

    print("Saved plots:")
    print("-", SCATTER_GRAPH)
    print("-", LINE_INPUTS_GRAPH)
    print("-", MODEL_GRAPH)
    print("-", COST_GRAPH)

    print("\n=== Training Summary ===")
    print(f"{'Metric':<25}{'Value':>15}")
    print("-" * 40)
    print(f"{'Training samples':<25}{X_train.shape[0]:>15}")
    print(f"{'Test samples':<25}{X_test.shape[0]:>15}")
    print(f"{'Final training cost':<25}{cost_history[-1]:>15.6f}")
    print(f"{'Test accuracy (%)':<25}{acc * 100:>15.2f}")

    test_cases = [
    [5.0, 3.4, 1.5, 0.2],
    [4.9, 3.1, 1.4, 0.1],
    [6.1, 2.8, 4.7, 1.2],
    [6.7, 3.1, 5.6, 2.1],
    [5.2, 3.6, 1.3, 0.3]]

    test_samples(test_cases, x_min, denom, w)

if __name__ == "__main__":
    main()
