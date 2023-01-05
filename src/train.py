import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from torch.utils import data
import matplotlib.pyplot as plt

from data_utils import SATTrainingDataset, JraphDataLoader
from model import network_definition

NUM_EPOCHS = 5  # 10
f = 0.1
batch_size = 10
path = "../Data/blocksworld"
# "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/ml_based_sat_solver/BroadcastTestSet_subset"
# img_path = (
#     "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/"
# )
# model_path = (
#     "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/"
# )


#  AUXILIARY METHODS


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


vmap_one_hot = jax.vmap(one_hot, in_axes=(0, None), out_axes=0)


def compute_log_probs(decoded_nodes, mask, candidate):
    a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
    return candidate * a


vmap_compute_log_probs = jax.vmap(
    compute_log_probs, in_axes=(None, None, 1), out_axes=1
)


def plot_accuracy_fig(test_acc_list, train_acc_list):
    plt.plot(
        np.arange(0, NUM_EPOCHS + 1, 1),
        test_acc_list,
        "o--",
        label="test accuracy",
        alpha=0.4,
    )
    plt.plot(
        np.arange(0, NUM_EPOCHS + 1, 1),
        train_acc_list,
        "o--",
        label="train accuracy",
        alpha=0.4,
    )
    plt.xlabel("epoch")
    plt.ylabel("accuracy of model / loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def train(
    batch_size,
    f,
    NUM_EPOCHS,
    path,
    img_path=False,
    model_path=False,
):

    sat_data = SATTrainingDataset(path)

    train_data, test_data = data.random_split(sat_data, [0.8, 0.2])
    train_eval_data, _ = data.random_split(sat_data, [0.2, 0.8])

    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = JraphDataLoader(test_data, batch_size=batch_size)
    train_eval_loader = JraphDataLoader(train_eval_data, batch_size=batch_size)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state, batch, f):

        g = jax.grad(prediction_loss)(params, batch, f)

        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def prediction_loss(params, batch, f: float):
        (mask, graph), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)  # (B*N, 2)
        candidates = vmap_one_hot(candidates, 2)  # (B*N, K, 2))
        log_prob = vmap_compute_log_probs(
            decoded_nodes, mask, candidates
        )  # (B*N, K, 2)
        weights = jax.nn.softmax(-f * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()
        return loss

    print("Entering training loop")
    test_acc_list = np.zeros(NUM_EPOCHS + 1)
    train_acc_list = np.zeros(NUM_EPOCHS + 1)

    evaluate = lambda loader: np.mean([prediction_loss(params, b, f) for b in loader])

    test_acc_list[0] = evaluate(test_loader)
    train_acc_list[0] = evaluate(train_eval_loader)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            print("batch_number", counter)
            params, opt_state = update(params, opt_state, batch, f)

        epoch_time = time.time() - start_time

        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        test_acc_list[epoch + 1] = evaluate(test_loader)
        train_acc_list[epoch + 1] = evaluate(train_eval_loader)
        print("Training set accuracy {}".format(train_acc_list[epoch + 1]))
        print("Test set accuracy {}".format(test_acc_list[epoch + 1]))

    plot_accuracy_fig(test_acc_list, train_acc_list)

    if img_path:
        plt.savefig(img_path + "accuracy.jpg", dpi=300, format="jpg")

    if model_path:
        model_params = [params, batch_size, f, NUM_EPOCHS]
        np.save(model_path, [model_params, train_acc_list, test_acc_list])


if __name__ == "__main__":
    train(batch_size, f, NUM_EPOCHS, path=path)
