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
batch_size = 1
path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/ml_based_sat_solver/BroadcastTestSet_subset"
img_path = (
    "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/"
)
model_path = (
    "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/"
)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


vmap_one_hot = jax.vmap(one_hot, in_axes=(0, None), out_axes=0)


def train(
    batch_size,
    f,
    NUM_EPOCHS,
    path="/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/ml_based_sat_solver/BroadcastTestSet_subset",
    img_path=False,
    model_path=False,
):  # used previously as path: "../Data/blocksworld"

    sat_data = SATTrainingDataset(path)

    train_data, test_data = data.random_split(sat_data, [0.8, 0.2])

    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def compute_log_probs(decoded_nodes, mask, candidate):
        a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
        return candidate * a

    vmap_compute_log_probs = jax.vmap(
        compute_log_probs, in_axes=(None, None, 1), out_axes=1
    )

    @jax.jit
    def update(params, opt_state, batch_masks, batch_graphs, batch_c, batch_e, f):

        g = jax.grad(prediction_loss)(
            params, batch_masks, batch_graphs, batch_c, batch_e, f
        )

        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def prediction_loss(params, mask, graph, candidates, energies, f: float):
        decoded_nodes = network.apply(params, graph)  # (B*N, 2)
        candidates = vmap_one_hot(candidates, 2)  # (B*N, K, 2))
        log_prob = vmap_compute_log_probs(
            decoded_nodes, mask, candidates
        )  # (B*N, K, 2)
        weights = jax.nn.softmax(-f * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()
        return loss

    # @jax.jit
    def test_loss(params, graph, mask, candidates, energies, f):
        decoded_nodes = network.apply(params, graph)  # (N, 2)
        candidates = vmap_one_hot(candidates, 2)  # (N, K, 2)
        a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
        log_prob = candidates * a
        energies_new = np.array(
            np.repeat([energies], len(mask), axis=0)
        ).T  # maps energies (K,) to energies_new (N, K)
        weights = jax.nn.softmax(-f * energies_new)  # (N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()
        return loss

    def compute_loss(dataset):
        summed_loss = 0
        counter = 0
        for (p, ce) in dataset:
            candidates = np.array([c for c in ce[0]])
            counter = counter + 1
            loss = test_loss(params, p[0], p[1], candidates, ce[1], f)
            summed_loss = summed_loss + loss
        return summed_loss / counter

    print("Entering training loop")
    test_acc_list = np.zeros(NUM_EPOCHS + 1)
    train_acc_list = np.zeros(NUM_EPOCHS + 1)

    test_acc_list[0] = compute_loss(test_data)
    train_acc_list[0] = compute_loss(train_data)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, (batch_p, batch_ce) in enumerate(train_loader):
            print("batch_number", counter)
            params, opt_state = update(params, opt_state, *batch_p, *batch_ce, f)

        epoch_time = time.time() - start_time

        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        test_acc_list[epoch + 1] = compute_loss(test_data)
        train_acc_list[epoch] = compute_loss(train_data)
        print("Training set accuracy {}".format(train_acc_list[epoch + 1]))
        print("Test set accuracy {}".format(test_acc_list[epoch + 1]))
    if img_path != False:
        plt.plot(
            np.arange(0, NUM_EPOCHS + 1, 1),
            test_acc_list,
            "o--",
            label="test accuracy",
            alpha=0.4,
        )
        plt.plot(
            np.arange(0, NUM_EPOCHS + 1, 1),
            test_acc_list,
            "o--",
            label="train accuracy",
            alpha=0.4,
        )
        plt.xlabel("epoch")
        plt.ylabel("accuracy of model / loss")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path + "accuracy.jpg", dpi=300, format="jpg")
    if model_path != False:
        model_params = [batch_size, f, NUM_EPOCHS]
        np.save(model_path, [model_params, train_acc_list, test_acc_list])

        # print("Training set accuracy {}".format(train_acc))
        # print("Test set accuracy {}".format(jnp.mean(test_acc_now)))
    # print(test_acc_list)
    plt.plot(np.arange(1, NUM_EPOCHS + 1, 1), test_acc_list, "o--")
    plt.ylabel("test loss")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig("test_acc.jpg", dpi=300, format="jpg")
    plt.show()
    # TODO: Save the model here


if __name__ == "__main__":
    train(
        batch_size, f, NUM_EPOCHS, path=path, img_path=img_path, model_path=model_path
    )
