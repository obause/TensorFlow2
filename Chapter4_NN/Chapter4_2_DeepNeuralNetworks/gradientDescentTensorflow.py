"""Gradient Descent Example similar to the Rosenbrock lecture in the course.
You can play around with this code, but it is not used in the course.
"""
# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
import tensorflow as tf

from helper import plot_rosenbrock


class Model:
    def __init__(self) -> None:
        self.x = tf.Variable(tf.random.uniform(minval=-2.0, maxval=2.0, shape=[2]))  # x = [x0, x1]
        self.learning_rate = 0.0005
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        self.current_loss_val = self.loss()

    def loss(self) -> tf.Tensor:
        self.current_loss_val = 100 * (self.x[0] ** 2 - self.x[1]) ** 2 + (self.x[0] - 1) ** 2
        return self.current_loss_val

    def fit(self) -> None:
        self.optimizer.minimize(self.loss, self.x)


def main() -> None:
    model = Model()
    gradient_steps = []
    x_start = model.x.numpy()
    num_iterations = 5000

    for it in range(num_iterations):
        model.fit()
        if it % 100 == 0:
            x = model.x.numpy()
            y = model.current_loss_val.numpy()
            print(f"X = {x} Y = {y}")
        gradient_steps.append(x)

    plot_rosenbrock(x_start, gradient_steps)


if __name__ == "__main__":
    main()
