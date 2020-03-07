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
\ln(\rho_{nk}) = \mathbb{E}(\ln(\Pi_{k})) + \frac{1}{2}\mathbb{E}(\ln|\Lambda_{k}|) - \frac{D}{2}\ln(2\pi) - \frac{1}{2}\mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda}
$$

to obtain the statistics, for $ r_{nk} = \frac{\rho_{nk}}{\sum_{j=1}^{K} \rho_{nj}}$

$$ N_{k} = \sum_{n=1}^{N} r_{nk}$$
$$ \overline{x_{k}} = \frac{1}{N_{k}} \sum_{n=1}^{N} r_{nk}x_{n} $$
$$ S_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N} r_{nk}(x_{n} - \overline{x_{k}})(x_{n} - \overline{x_{k}})^{T} $$

The original expression of $ \rho_{nk} $ might be simplified and allows to directly compute the $ r_{nk} $. By expressing:

$$ \ln(\tilde\Pi_{k}) = \mathbb{E}(\ln(\Pi_{k})) $$
$$ \ln(\tilde{\Lambda_{k}}) = \mathbb{E}(\ln(\Lambda_{k})) $$

Then

$$ r_{nk} = \frac{\tilde{\Pi_{k}}\tilde{\Lambda_{k}}e^{\mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda}}}{C} $$

So if it is possible to derive analytical expressions for non constant elements in this expression, it will be feasible to obtain directly the $ r_{nk} $.

First, according to [wikipedia](https://en.wikipedia.org/wiki/Dirichlet_distribution), if $ X_{i} \sim \mathcal{D}(\alpha)$ , then $ \mathbb{E}(\ln(X_{i})) = \psi(\alpha_{i}) - \psi(\sum_{j} \alpha_{j}) $ where $ \psi $ is the digamma function. It thus follows that:

$$ \ln(\tilde\Pi_{k}) = \psi(\alpha_{k}) - \psi(\sum_{j} \alpha_{j}) $$

Similarly, according to [wikipedia](https://en.wikipedia.org/wiki/Wishart_distribution) again (section Properties/Log-expectation) if $ X \sim \mathcal{W}(V, n) $,  then $ \mathbb{E}(\ln(X_{i})) = \psi_{p}(\frac{n}{2}) + p\ln(2) + \ln |V| $ with $ \psi_{p} $ the multivariate digamma function. [wikipedia](https://en.wikipedia.org/wiki/Multivariate_gamma_function) (section Derivatives) gives 
$\psi_{p}(a) = \frac{\partial \log \Gamma_{p}(a)}{\partial a} = \sum_{i=0}^{p} \psi(a + \frac{1-i}{2})$. Thus it follows that:

$$ \ln(\tilde{\Lambda_{k}}) = D\ln(2) + \ln |W_{k}| + \sum_{i=0}^{D} \psi(\frac{\nu_{k} + 1 - i}{2}) $$

Finally, it is necessary to also derive an analytical expression for $E_{k} =  \mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda} $.

First, it is important to notice that in the general case $ x^{T}\Sigma x = tr(x^{T}\Sigma x) = tr(\Sigma xx^{T}) $, thus, using the factorization $ q^{\*}(\mu_{k},\Lambda_{k}) = q^{\*}(\mu_{k}|\Lambda_{k})q^{\*}(\Lambda_{k}) $:

$$ E_{k} = \int \int tr(\Lambda_{k}(x_{n}-\mu_{k})(x_{n}-\mu_{k})^{T})q^{\*}(\mu_{k}|\Lambda_{k})q^{\*}(\Lambda_{k}) d\mu_{k}d\Lambda_{k} $$


We furthermore know that $ q^{\*}(\mu_{k}|\Lambda_{k}) \sim \mathcal{N}(\mu_{k}|m_{k},(\beta_{k}\Lambda_{k})^{-1}) $, and thus it follows that

$$ \mathbb{E}(\mu_{k}) = m_{k} $$

and also, from the [matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf) (page 35, equation (321)) that

$$ \mathbb{E}(\mu_{k}\mu_{k}^{T}) = m_{k}m_{k}^{T} + (\beta_{k}\Lambda_{k})^{-1} $$

and therefore:

$$ \mathbb{E}((x_{n}-\mu_{k})(x_{n}-\mu_{k})^{T}) = \mathbb{E}(x_{n}x_{n}^{T} - \mu_{k}x_{n}^{T} - \mu_{k}x_{n}^{T} + \mu_{k}\mu_{k}^{T}) = (x_{n}-m_{k})(x_{n}-m_{k})^{T} + (\beta_{k}\Lambda_{k})^{-1} $$


Thus by linearity (is it actually linearity ?) of the expected value and of the trace:

$$ E_{k} = \int tr[\Lambda_{k}((x_{n}-m_{k})(x_{n}-m_{k})^{T} + (\beta_{k}\Lambda_{k})^{-1})]q^{\*}(\Lambda_{k})d\Lambda_{k} $$

$$ = \frac{D}{\beta_{k}} + \int tr(\Lambda_{k}((x_{n}-m_{k})(x_{n}-m_{k})^{T})q^{\*}(\Lambda_{k})d\Lambda_{k} $$

$$ = \frac{D}{\beta_{k}} + \int ((x_{n}-m_{k})^{T}\Lambda_{k}(x_{n}-m_{k}))q^{\*}(\Lambda_{k})d\Lambda_{k} $$

which finally yields, as $ \mathbb{E}(\Lambda_{k}) = \nu_{k}W_{k}$

$$ = \frac{D}{\beta_{k}} + \nu_{k}(x_{n}-m_{k})^{T}W_{k}(x_{n}-m_{k}) $$


__Maximization__

TODO


## Step 2: Implement Normalizing Flows on Simple Example

### Study of Article

__Finite Flows__

Considering invertible smooth mappings $ f: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d} $, the mapping of a random variable $ z $ with distribution $ q(z) $ will result in a new random variable $ z' = f(z) $ which follows the distribution :

$$ q(z') = q(z) |det\frac{\partial f^{-1}}{\partial z'}| = q(z) |det\frac{\partial f}{\partial z}|^{-1} $$

The right equality is the [inverse function theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem)

Thus for a succession of k mappings:

$$ z_{K} = f_{K} \circ ... \circ f_{1} (z_{0})$$
$$ \ln(q_{K}(z_{K}) = \ln(q_{0}(z_{0})) - \sum_{k=1}^{K} \ln|det\frac{\partial f_{k}}{\partial z_{k-1}}|$$

Exemple in 1D: Let $ z_{0} \sim q_{o}(z_{0}) $ and $ z_{1} \sim q_{1}(z_{1}) $ such that $ z_{1} = f(z_{0}) $ with f invertible and smooth such that $ f^{-1}(z_{1}) = z_{0} $. Both distribution $ q_{0} and q_{1} $ have integrals that sum up to 1. So,

$$ \int_{-\inf}^{\inf} q_{0}(z_{0})dz_{0} = \int_{-\inf}^{\inf} q_{1}(z_{1})dz_{1} $$

Therefore,

$$ q_{1}(z_{1}) = q_{0}(z_{0})\frac{dz_{0}}{dz_{1}} = q_{0}(z_{0})\frac{df^{-1}(z_{1})}{dz_{1}} $$

Which in turns, thanks to the inverse function theorem yields:

$$ q_{1}(z_{1}) = q_{0}(z_{0})(\frac{df(z_{0})}{dz_{0}})^{-1}$$

_note_: absolute value of f can be considered to make sure its rate of change is positive.

__Planar Flows__

Considering a family of transformations:

$$ f(z) = z + uh(w^{T}z+b) $$

where $ h $ is a smooth element-wise non-linearity. Then it is possible to derive, through the [chain rule](https://en.wikipedia.org/wiki/Chain_rule)

$$ |det\frac{\partial f}{\partial z}| = |det(\frac{\partial z}{\partial z} + u\frac{\partial h(w^{T}z+b)}{\partial z})| = |det(I + uh'(w^{T}z+b)\frac{\partial (w^{T}z+b)}{\partial})| = |det(I + u(h'(w^{T}z+b)w)^{T})|$$

which in turn yields, through the [matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma), with the notation $ \psi(z) = h'(w^{T}z+b)w $

$$ |det\frac{\partial f}{\partial z}| = |(1+u^{T}\psi(z))det(I)| = |1+u^{T}\psi(z)| $$

_note_: why the transpose act like this? And why $\frac{\partial z}{\partial z} = I$ ? [-> [Implicit notation of Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)]

Resulting distributions, from a unit 2D gaussian originally:

![Original Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/original_unit_gaussian_3D.png)

![Original Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/original_unit_gaussian_contour.png)

After a single flow, which contracts along the {y=1} hyperplan:

![1 Planar Flow Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/1planar_flows_3D.png)

![1 Planar Flow Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/1planar_flows_contour.png)

After applying three other flows which expand along the {x=1} hyperplan:

![2 Planar Flow Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/2planar_flows_3D.png)

![2 Planar Flow Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/2planar_flows_contour.png)

__note__: I wrote a script to understand the influence of the flow on the resulting distribution, see below:

for the contraction along {y=1}

![Singular Planar Flow Factor Arg {y=1}](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/single_flow_arg_y=1.png)

![Singular Planar Flow Factor {x=1}](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/single_flow_exp_y=1.png)

for the expansion along {x=1}

![Singular Planar Flow Factor Arg {x=1}](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/single_flow_arg_x=1.png)

![Singular Planar Flow Factor {x=1}](https://raw.githubusercontent.com/pierresegonne/VariationalInferenceNormalizingFlows/master/assets/single_flow_exp_x=1.png)



# MISC

[blog post with pytorch implementation of flows](https://www.ritchievink.com/blog/2019/10/11/sculpting-distributions-with-normalizing-flows/)