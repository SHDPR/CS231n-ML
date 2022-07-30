import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    model = PoissonRegression(step_size=lr, eps=1e-3)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('real')
    plt.ylabel('prediction')


    margin = (max(y_eval[:]) - min(y_eval[:])) * 0.2
    x1 = np.arange(min(y_eval[:]) - margin, max(y_eval[:]) + margin, 1)
    x2 = x1
    plt.plot(x1, x2, c = 'red', linewidth = 2)
    plt.savefig('output/p03d.png')

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)

        ## Batch Descent
        while False:
            theta = np.copy(self.theta)
            self.theta += self.step_size * x.T.dot(y - np.exp(x.dot(self.theta))) / m

            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break

        ## Stochastic Descent
        while True:
            theta_prev = np.copy(self.theta)
            for i in range(0, m):
                self.theta += self.step_size * x[i,:] * (y[i] - np.exp(x[i,:].dot(self.theta))) / m

            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***

        return np.exp(x.dot(self.theta))

        # *** END CODE HERE ***
