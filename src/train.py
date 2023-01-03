import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from torch.utils import data

from data_utils import SATTrainingDataset, JraphDataLoader
from model import network_definition

NUM_EPOCHS = 3  # 10
f = 0.1


# # Make a batched version of the forwarding
# batched_predict = jax.vmap(network.apply, in_axes=(None, 0))


# def loss(params, problems, targets):
#     preds = batched_predict(params, problems)
#     return -jnp.mean(preds * targets)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


vmap_one_hot = jax.vmap(one_hot, in_axes=(0, None), out_axes=0)


def train(
    path="../Data/BroadcastTestSet",
):  # used previously as path: "../Data/blocksworld" ##
    sat_data = SATTrainingDataset(path)
    train_data, test_data = data.random_split(sat_data, [0.8, 0.2])

    train_loader = JraphDataLoader(train_data, batch_size=2, shuffle=True)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def compute_log_probs(decoded_nodes, mask, candidate):
        a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
        b = candidate * a
        return b
        # b = jnp.dot(candidate, a.T)
        # return b

    vmap_compute_log_probs = jax.vmap(
        compute_log_probs, in_axes=(None, None, 0), out_axes=0
    )

    @jax.jit
    def update(params, opt_state, batch_masks, batch_graphs, batch_c, batch_e, f):

        batchsize = len(batch_e)
        # print("batchsize", batchsize)
        if batchsize == 1:
            g = jax.grad(prediction_loss)(
                params, batch_masks[0], batch_graphs[0], batch_c[0], batch_e[0], f
            )
        else:
            g = jax.grad(batched_loss_slow)(
                params, batch_masks, batch_graphs, batch_c, batch_e, f
            )

        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def prediction_loss(params, mask, graph, candidates, energies, f: float):
        decoded_nodes = network.apply(params, graph)
        candidates = vmap_one_hot(candidates, 2)
        log_prob = vmap_compute_log_probs(decoded_nodes, mask, candidates)
        weights = jax.nn.softmax(-f * energies)
        weighted_log_probs = jax.vmap(jnp.dot, axis_name=(0, 0), out_axes=0)(
            log_prob, weights
        )
        loss = -jnp.sum(weighted_log_probs) / jnp.sum(
            mask
        )  # sum over all candidates and variables
        # loss = -jnp.sum(summed_weighted_log_probs @ mask[:, None]) / jnp.sum(mask)
        # print(np.shape(loss))
        return loss

    # batched_loss = jnp.sum(jax.vmap(new_prediction_loss, in_axes=(None, 0, 0, 0,0, None), out_axes=0))

    # @jax.jit (do not use it here! Otherwise it does not work!)
    def batched_loss_slow(
        params, batch_masks, batch_graphs, batch_candidates, batch_energies, f: float
    ):
        batchsize = len(batch_energies)
        loss_vec = np.zeros(batchsize)
        for i in range(batchsize):
            loss_vec[i] = prediction_loss(
                params,
                batch_masks[i],
                batch_graphs[i],
                batch_candidates[i],
                batch_energies[i],
                f,
            )
        loss_sum = np.sum(loss_vec) / batchsize
        return loss_sum

    # @jax.jit
    def test_loss(params, graph, mask, candidates, energies, f):
        decoded_nodes = network.apply(params, graph)
        candidates = vmap_one_hot(candidates, 2)
        log_prob = vmap_compute_log_probs(decoded_nodes, mask, candidates)
        weights = jax.nn.softmax(-f * energies)
        weighted_log_probs = jax.vmap(jnp.dot, axis_name=(0, 0), out_axes=0)(
            log_prob, weights
        )
        summed_weighted_log_probs = np.sum(
            weighted_log_probs, axis=0
        )  # sum over all candidates
        loss = -jnp.sum(summed_weighted_log_probs @ mask[:, None]) / jnp.sum(mask)
        # print(np.shape(loss))
        return loss

    print("Entering training loop")
    test_acc_list = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, (batch_p, batch_ce) in enumerate(train_loader):
            print("batch_number", counter)
            params, opt_state = update(params, opt_state, *batch_p, *batch_ce, f)
            print("batch", counter, "done")

        epoch_time = time.time() - start_time

        # train_acc = accuracy(params, train_images, train_labels)
        # test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        # test_acc = jnp.mean(jnp.asarray([prediction_loss(params, p.mask, p.graph, s) for (p, s) in test_data]))

        # TBD!!!

        # test_acc_now=[]
        # for (p, ce) in test_data:
        #    loss=test_loss(params, p[0], p[1], ce[0],ce[1] , f)
        #    print(loss)
        #    test_acc_now.append(loss)
        # print(test_acc_now)
        # test_acc_list.append(jnp.mean(test_acc_now))

        ##

        # print("Training set accuracy {}".format(train_acc))
        # print("Test set accuracy {}".format(jnp.mean(test_acc_now)))
    # print(test_acc_list)
    # plt.plot(np.arange(0,NUM_EPOCHS,1), test_acc_list)
    # plt.savefig("test_acc.jpg", dpi=300, format="jpg")
    # plt.show()
    # TODO: Save the model here


if __name__ == "__main__":
    train()
