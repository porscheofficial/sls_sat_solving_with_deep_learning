import time
import optax
import haiku as hk
import jax.numpy as jnp
import jax
from torch.utils import data

from data_utils import SATTrainingDataset, JraphDataLoader
from model import network_definition

NUM_EPOCHS = 10


# # Make a batched version of the forwarding
# batched_predict = jax.vmap(network.apply, in_axes=(None, 0))


# def loss(params, problems, targets):
#     preds = batched_predict(params, problems)
#     return -jnp.mean(preds * targets)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def train(path="../Data/BroadcastTestSet"):
    sat_data = SATTrainingDataset(path)
    train_data, test_data = data.random_split(sat_data, [0.8, 0.2])

    train_loader = JraphDataLoader(train_data, batch_size=1, shuffle=True)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state, x, y):
        g = jax.grad(prediction_loss)(params, *x, y)
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def prediction_loss(params, mask, graph, solution):
        decoded_nodes = network.apply(params, graph)
        solution = one_hot(solution, 2)
        # We interpret the decoded nodes as a pair of logits for each node.
        log_prob = jax.nn.log_softmax(decoded_nodes) * solution
        return -jnp.sum(log_prob * mask[:, None]) / jnp.sum(mask)

    # batched_loss = jax.vmap(prediction_loss, in_axes=(None, 0))

    print("Entering training loop")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for (x, y) in train_loader:
            params, opt_state = update(params, opt_state, x, y)
            # print(f"{batch} done")
        epoch_time = time.time() - start_time

        # train_acc = accuracy(params, train_images, train_labels)
        # test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        test_acc = jnp.mean(jnp.asarray([prediction_loss(params, p.mask, p.graph, s) for (p, s) in test_data]))
        # print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

    # TODO: Save the model here

if __name__ == "__main__":
    train()
