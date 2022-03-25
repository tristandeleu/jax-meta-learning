import jax.numpy as jnp
import jax
import jaxopt

from jax_meta.metalearners.maml import MAML


class iMAML(MAML):
    def __init__(
            self,
            model,
            num_steps=5,
            alpha=0.1,
            lambda_=1.,
            regu_coef=1.,
            cg_damping=10.,
            cg_steps=5
        ):
        super().__init__(model, num_steps=num_steps, alpha=alpha)
        self.lambda_ = lambda_
        self.regu_coef = regu_coef
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps

    def adapt(self, init_params, state, inputs, targets, args):
        def inner_loss(params):
            loss, logs = self.loss(params, state, inputs, targets, args)
            proximal_term = jax.tree_util.tree_map(
                lambda p, p0: 0.5 * jnp.sum((p - p0) ** 2),
                params, init_params
            )
            loss = loss + self.lambda_ * jaxopt.tree_util.tree_sum(proximal_term)
            return (loss, logs)

        def implicit_diff_solve(matvec, b, init=None, **kwargs):
            damping = lambda h, t: (1. + self.regu_coef) * t + h / (self.lambda_ + self.cg_damping)
            def matvec_damped(inputs):
                return jax.tree_util.tree_map(damping, matvec(inputs), inputs)

            return jax.scipy.sparse.linalg.cg(matvec_damped, b, x0=init, **kwargs)[0]

        solver = jaxopt.GradientDescent(
            inner_loss,
            stepsize=self.alpha,
            maxiter=self.num_steps,
            tol=float('inf'),
            implicit_diff=True,
            implicit_diff_solve=implicit_diff_solve,
            has_aux=True,
            unroll=False
        )
        params, state = solver.run(init_params)

        return (params, state.aux)
