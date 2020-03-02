<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>
------------------------------------------------------------------------------------------------

# Notes for Variational Inference with Normalizing Flows

## Step 1: Understand and Implement V.I and Mean Field Approximation

From Bishop 2006, we know that for a mixture of gaussians, we can use the following model:

$$ p(X,Z,\Pi,\mu,\Lambda) = p(X|Z,\Pi,\mu,\Lambda)p(Z|\Pi)p(\Pi)p(\mu|\Lambda)p(\Lambda)$$

with conjugate priors distributions:

$$ p(\Pi) = Dir(\Pi | \alpha_{0})$$

$$ p(\mu | \Lambda)p(\Lambda) = p(\mu , \Lambda) = \prod_{k=1}^{K}\mathcal{N}(\mu_{k} | m_{0}, (\beta_{0}\Lambda_{k})^{-1})\mathcal{W}(\Lambda_{k} | W_{0}, \nu_{0}) $$

$$ p(Z|\Pi) = \prod_{n=1}^{N}\prod_{k=1}^{K}\Pi_{k}^{z_{nk}} $$

$$ p(X|Z,\Pi,\mu,\Lambda) = \prod_{n=1}^{N}\prod_{k=1}^{K} \mathcal{N}(x_{n}|\mu_{k},\Lambda^{-1})^{z_{nk}} $$

With the mean field approximation $ q(Z,\Pi,\mu,\Lambda) = q(Z)q(\Pi,\mu,\Lambda) $ that splits the latent variables from the parameters of the model, iterative update rules, similar to EM can be derived:

__Expectation__

Approximating the distribution q results in the necessity to compute:

$$
ln(\rho_{nk}) = \mathbb{E}(ln(\Pi_{k})) + \frac{1}{2}\mathbb{E}(ln|\Lambda_{k}|) - \frac{D}{2}ln(2\pi) - \frac{1}{2}\mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda}
$$

to obtain the statistics, for $ r_{nk} = \frac{\rho_{nk}}{\sum_{j=1}^{K} \rho_{nj}}$

$$ N_{k} = \sum_{n=1}^{N} r_{nk}$$
$$ \overline{x_{k}} = \frac{1}{N_{k}} \sum_{n=1}^{N} r_{nk}x_{n} $$
$$ S_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N} r_{nk}(x_{n} - \overline{x_{k}})(x_{n} - \overline{x_{k}})^{T} $$

__Maximization__


## Step 2: Implement Normalizing Flows on Simple Example

TODO