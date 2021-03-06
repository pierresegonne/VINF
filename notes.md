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

$ \require{ams} $

# Notes for Variational Inference with Normalizing Flows

### Table of content

1. [Introduction to Variational Inference](#introduction-to-variational-inference)
   * [Variational Inference](#variational-inference)
   * [Mean Field Approximation](#mean-field-approximation)
   * [Example, Mixture of Gaussians](#mixture-of-gaussians)
     * [Expectation Derivation](#expectation)
     * [Maximization Derivation](#maximization)
2. [Understanding Normalizing Flows](#normalizing-flows)
   * [Finite Flows](#finite-flows)
   * [Planar Flows](#planar-flows)
     * [Planar Flows Derivation](#planar-flow-derivation)
     * [Planar Flows Visualization](#planar-flow-visualization)
   * [Radial Flows](#radial-flows)
     * [Radial Flows Derivation](#radial-flow-derivation)
     * [Radial Flows Visualization](#radial-flow-visualization)
3. [Implement Normalizing Flows](#normalizing-flows-as-neural-network)
4. [Evaluating Variational Inference](#evaluating-variational-inference)

[Unanswered Questions](#unanswered-questions)

[TODO](#todo)

[Misc](#misc)

[References](#references)

## Introduction to Variational Inference

### Variational Inference


Probability theory is key to quantify uncertainty in any phenomenon and is therefore essential to the development of thinking machines.
Indeed, any machine, or model, which needs to generate representations from given observations in order to accomplish its mission requires the generation and the numerical evaluation of beliefs,
and probability theory provides well defined rules to induce and manipulate such beliefs.

Furthermore, probabilistic modelling provides directly a way to generate machine learning models respecting the probability theory. Probabilistic models describe how the interactions between observed variables,
latent (in the sense of unobserved) variables and weights can predict target variables, while considering all variables as random variables with their associated probabilities.

To demonstrate how probabilistic models work, let's consider a probabilistic linear regression

\[INSERT GRAPHICAL MODEL\]

The graphical model indicates

$$
p(y, x, \omega) = p(y|x, \omega)p(\omega)
$$

Bayes therorem

$$
p(A|B) = \frac{p(B|A)p(A)}{p(B)}
$$

Gives

$$
p(\omega | y, x) = \frac{p(y|\omega, x)p(\omega | x)}{p(y|x)} = \frac{p(y|\omega, x)p(\omega)}{p(y|x)}
$$

Training the model corresponds to finding the weights that have the highest probability given the observed data and the target variable

$$
\omega^{*} = \max_{\omega} p(\omega| y, x)
$$

Then if it is wanted to determined these weights, providing the model additional information about its variables, allows training.

It will be assumed that the observed data follows a normal distribution with mean $\mu$ and covariance matrix $\Sigma$,

$$
x \sim \mathcal{N}(\mu,\Sigma)
$$

$$
\mathcal{N}(x|\mu,\Sigma) = \frac{1}{\sqrt{2\pi|\Sigma|}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}
$$

That the prior distribution on the weights is

$$
\omega \sim \mathcal{N}(0,\gamma^{2}I)
$$

And that the target variables also follow a normal distribution around the prediction $\omega^{T}X$

$$
y | \omega, x \sim \mathcal{N}(\omega^{T}x,\sigma^{2}I)
$$


This example provides an insight into probabilistic modelling and why the determination of the posterior distribution $p(\omega|y,x)$ is so useful. Once the posterior distribution learned, it becomes possible
to compute $p(\hat{y}|\hat{x},\omega)$, ADD JOINT PARAMETRISATION FOR NEW DATA POINT which can be interpreted as a quantified belief about a new data point.

Inference about unknown quantities as calculations about the posterior.

In real applications, posterior is difficult.

### Mean Field Approximation

### Mixture of Gaussians

From Bishop 2006, we know that for a mixture of gaussians, we can use
the following model:

$$ p(X,Z,\Pi,\mu,\Lambda) =
p(X|Z,\Pi,\mu,\Lambda)p(Z|\Pi)p(\Pi)p(\mu|\Lambda)p(\Lambda)$$

with conjugate priors distributions:

$$ p(\Pi) = Dir(\Pi | \alpha_{0})$$

$$ p(\mu | \Lambda)p(\Lambda) = p(\mu , \Lambda) =
\prod_{k=1}^{K}\mathcal{N}(\mu_{k} | m_{0},
(\beta_{0}\Lambda_{k})^{-1})\mathcal{W}(\Lambda_{k} | W_{0}, \nu_{0}) $$

$$ p(Z|\Pi) = \prod_{n=1}^{N}\prod_{k=1}^{K}\Pi_{k}^{z_{nk}} $$

$$ p(X|Z,\Pi,\mu,\Lambda) = \prod_{n=1}^{N}\prod_{k=1}^{K}
\mathcal{N}(x_{n}|\mu_{k},\Lambda^{-1})^{z_{nk}} $$

With the mean field approximation $ q(Z,\Pi,\mu,\Lambda) =
q(Z)q(\Pi,\mu,\Lambda) $ that splits the latent variables from the
parameters of the model, iterative update rules, similar to EM can be
derived:

#### Expectation

Approximating the distribution q results in the necessity to compute:

$$ \ln(\rho_{nk}) = \mathbb{E}(\ln(\Pi_{k})) +
\frac{1}{2}\mathbb{E}(\ln|\Lambda_{k}|) - \frac{D}{2}\ln(2\pi) -
\frac{1}{2}\mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda}
$$

to obtain the statistics, for $ r_{nk} = \frac{\rho_{nk}}{\sum_{j=1}^{K}
\rho_{nj}}$

$$ N_{k} = \sum_{n=1}^{N} r_{nk}$$ $$ \overline{x_{k}} = \frac{1}{N_{k}}
\sum_{n=1}^{N} r_{nk}x_{n} $$ $$ S_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N}
r_{nk}(x_{n} - \overline{x_{k}})(x_{n} - \overline{x_{k}})^{T} $$

The original expression of $ \rho_{nk} $ might be simplified and allows
to directly compute the $ r_{nk} $. By expressing:

$$ \ln(\tilde\Pi_{k}) = \mathbb{E}(\ln(\Pi_{k})) $$ $$
\ln(\tilde{\Lambda_{k}}) = \mathbb{E}(\ln(\Lambda_{k})) $$

Then

$$ r_{nk} =
\frac{\tilde{\Pi_{k}}\tilde{\Lambda_{k}}e^{\mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda}}}{C}
$$

So if it is possible to derive analytical expressions for non constant
elements in this expression, it will be feasible to obtain directly the
$ r_{nk} $.

First, according to
[wikipedia](https://en.wikipedia.org/wiki/Dirichlet_distribution), if $
X_{i} \sim \mathcal{D}(\alpha)$ , then $ \mathbb{E}(\ln(X_{i})) =
\psi(\alpha_{i}) - \psi(\sum_{j} \alpha_{j}) $ where $ \psi $ is the
digamma function. It thus follows that:

$$ \ln(\tilde\Pi_{k}) = \psi(\alpha_{k}) - \psi(\sum_{j} \alpha_{j}) $$

Similarly, according to
[wikipedia](https://en.wikipedia.org/wiki/Wishart_distribution) again
(section Properties/Log-expectation) if $ X \sim \mathcal{W}(V, n) $,
then $ \mathbb{E}(\ln(X_{i})) = \psi_{p}(\frac{n}{2}) + p\ln(2) + \ln
|V| $ with $ \psi_{p} $ the multivariate digamma function.
[wikipedia](https://en.wikipedia.org/wiki/Multivariate_gamma_function)
(section Derivatives) gives $\psi_{p}(a) = \frac{\partial \log
\Gamma_{p}(a)}{\partial a} = \sum_{i=0}^{p} \psi(a + \frac{1-i}{2})$.
Thus it follows that:

$$ \ln(\tilde{\Lambda_{k}}) = D\ln(2) + \ln |W_{k}| + \sum_{i=0}^{D}
\psi(\frac{\nu_{k} + 1 - i}{2}) $$

Finally, it is necessary to also derive an analytical expression for
$E_{k} =
\mathbb{E}((x_{n}-\mu_{k})^{T}\Lambda_{k}(x_{n}-\mu_{k}))_{\mu,\Lambda}
$.

First, it is important to notice that in the general case $ x^{T}\Sigma
x = tr(x^{T}\Sigma x) = tr(\Sigma xx^{T}) $, thus, using the
factorization $ q^{\*}(\mu_{k},\Lambda_{k}) =
q^{\*}(\mu_{k}|\Lambda_{k})q^{\*}(\Lambda_{k}) $:

$$ E_{k} = \int \int
tr(\Lambda_{k}(x_{n}-\mu_{k})(x_{n}-\mu_{k})^{T})q^{\*}(\mu_{k}|\Lambda_{k})q^{\*}(\Lambda_{k})
d\mu_{k}d\Lambda_{k} $$


We furthermore know that $ q^{\*}(\mu_{k}|\Lambda_{k}) \sim
\mathcal{N}(\mu_{k}|m_{k},(\beta_{k}\Lambda_{k})^{-1}) $, and thus it
follows that

$$ \mathbb{E}(\mu_{k}) = m_{k} $$

and also, from the
[matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf)
(page 35, equation (321)) that

$$ \mathbb{E}(\mu_{k}\mu_{k}^{T}) = m_{k}m_{k}^{T} +
(\beta_{k}\Lambda_{k})^{-1} $$

and therefore:

$$ \mathbb{E}((x_{n}-\mu_{k})(x_{n}-\mu_{k})^{T}) =
\mathbb{E}(x_{n}x_{n}^{T} - \mu_{k}x_{n}^{T} - \mu_{k}x_{n}^{T} +
\mu_{k}\mu_{k}^{T}) = (x_{n}-m_{k})(x_{n}-m_{k})^{T} +
(\beta_{k}\Lambda_{k})^{-1} $$


Thus by linearity (is it actually linearity ?) of the expected value and
of the trace:

$$ E_{k} = \int \text{tr}\[\Lambda_{k}((x_{n}-m_{k})(x_{n}-m_{k})^{T} +
(\beta_{k}\Lambda_{k})^{-1})\]q^{\*}(\Lambda_{k})d\Lambda_{k} $$

$$ = \frac{D}{\beta_{k}} + \int
tr(\Lambda_{k}((x_{n}-m_{k})(x_{n}-m_{k})^{T})q^{\*}(\Lambda_{k})d\Lambda_{k}
$$

$$ = \frac{D}{\beta_{k}} + \int
((x_{n}-m_{k})^{T}\Lambda_{k}(x_{n}-m_{k}))q^{\*}(\Lambda_{k})d\Lambda_{k}
$$

which finally yields, as $ \mathbb{E}(\Lambda_{k}) = \nu_{k}W_{k}$

$$ = \frac{D}{\beta_{k}} + \nu_{k}(x_{n}-m_{k})^{T}W_{k}(x_{n}-m_{k}) $$


#### Maximization

TODO


## Normalizing Flows

### Finite Flows

Considering invertible smooth mappings $ f: \mathbb{R}^{d} \rightarrow
\mathbb{R}^{d} $, the mapping of a random variable $ z $ with
distribution $ q(z) $ will result in a new random variable $ z' = f(z) $
which follows the distribution :

$$ q(z') = q(z) |det\frac{\partial f^{-1}}{\partial z'}| = q(z)
|det\frac{\partial f}{\partial z}|^{-1} $$

The right equality is the
[inverse function theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem)

Thus for a succession of k mappings:

$$ z_{K} = f_{K} \circ ... \circ f_{1} (z_{0})$$ $$ \ln(q_{K}(z_{K}) =
\ln(q_{0}(z_{0})) - \sum_{k=1}^{K} \ln|det\frac{\partial f_{k}}{\partial
z_{k-1}}|$$

Exemple in 1D: Let $ z_{0} \sim q_{o}(z_{0}) $ and $ z_{1} \sim
q_{1}(z_{1}) $ such that $ z_{1} = f(z_{0}) $ with f invertible and
smooth such that $ f^{-1}(z_{1}) = z_{0} $. It can be seen as a
[change of variables](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function),
for which the differential area must be invariant under change of
variables.

Therefore,

$$ q_{1}(z_{1}) = q_{0}(z_{0})|\frac{dz_{0}}{dz_{1}}| =
q_{0}(z_{0})|\frac{df^{-1}(z_{1})}{dz_{1}}| $$

Which in turns, thanks to the inverse function theorem yields:

$$ q_{1}(z_{1}) = q_{0}(z_{0})|\frac{df(z_{0})}{dz_{0}}|^{-1}$$

_note_: absolute value of f can be considered to make sure its rate of
change is positive.

__Example__:

Let's apply an exponential flow to a standard normal distribution:

Thus let $q_{0} \sim \mathcal{N}(0,1)$ and $f$ be defined on $U_{0}$ as
$f(z_{0}) = \exp(z_{0})$.

We immediately have that the inverse of $f$, $f^{-1}$, is defined on
$U_{1}$ as $f^{-1}(z_{1}) = \log(z_{1})$, and its jacobian is $J(z_{0})
= \frac{df}{dz_{0}}(z_{0}) = f(z_{0}) = z_{1}$.

Therefore, the log probability of an event in $U_{1}$ can be computed
as:

$$ \log(q_{1}(z_{1})) = \log(q_{0}(f^{-1}(z_{1})) - \log(z_{1}) =
\log(q_{0}(\log(z_{1})) - \log(z_{1}) $$

For example in the case of $z_{1} = 1$, then $ \log(q_{1}(1)) =
\log(\mathcal{N}(\log(1)|0,1)) - \log(1) = \log(\frac{1}{\sqrt{2\pi}}) $

### Planar Flows

#### Planar Flow Derivation

Considering a family of transformations:

$$ f(z) = z + uh(w^{T}z+b) $$

where $ h $ is a smooth element-wise non-linearity. Then it is possible
to derive, through the
[chain rule](https://en.wikipedia.org/wiki/Chain_rule)

$$ |det\frac{\partial f}{\partial z}| = |det(\frac{\partial z}{\partial
z} + u^{T}\frac{\partial h(w^{T}z+b)}{\partial z})| = |det(I +
u^{T}h'(w^{T}z+b)\frac{\partial (w^{T}z+b)}{\partial})| = |det(I +
u^{T}h'(w^{T}z+b)w)|$$

which in turn yields, through the
[matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma),
with the notation $ \psi(z) = h'(w^{T}z+b)w $

$$ |det\frac{\partial f}{\partial z}| = |(1+u^{T}\psi(z))det(I)| =
|1+u^{T}\psi(z)| $$

_note_: why the transpose act like this? [-> eq (69), page 10 of
[matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf)
cf $ \frac{\partial w^{T}z}{\partial z} = \frac{\partial
\sum_{i}w_{i}z_{i}}{\partial z} = \sum_{i} w_{i}e_{i} = w $] And why
$\frac{\partial z}{\partial z} = I$ ? [->
[Implicit notation of Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)]

#### Planar Flow Visualization

Resulting distributions, from a unit 2D gaussian originally:

![Original Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/original_unit_gaussian_3D.png)

![Original Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/original_unit_gaussian_contour.png)

After a single flow, which contracts along the {y=1} hyperplan:

![1 Planar Flow Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/1planar_flows_3D.png)

![1 Planar Flow Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/1planar_flows_contour.png)

After applying three other flows which expand along the {x=1} hyperplan:

![2 Planar Flow Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/2planar_flows_3D.png)

![2 Planar Flow Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/2planar_flows_contour.png)

__Note__: I wrote a script to understand the influence of the flow on
the resulting distribution, see below:

For the contraction along {y=1}

![Singular Planar Flow Factor Arg {y=1}](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_planar_flow_arg_y=1.png)

![Singular Planar Flow Factor {x=1}](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_planar_flow_exp_y=1.png)

For the expansion along {x=1}

![Singular Planar Flow Factor Arg {x=1}](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_planar_flow_arg_x=1.png)

![Singular Planar Flow Factor {x=1}](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_planar_flow_exp_x=1.png)

__Note__: If the weights of the flows collapse to 0, due for example to
a training error, then as the following figure shows, the flows will
have have no effect on the original distribution, as they will converge
towards the identity transformation:

![Singular Planar Flow Factor Weights Collapse](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/1planar_flows_weights0.png)


### Radial Flows

#### Radial Flow Derivation

Considering a family of transformations:

$$ f(z) = z + \beta h(\alpha ,r)(z-z_{0})$$

where $ r = ||z-z_{0}|| $ and $ h(\alpha ,r) = \frac{1}{\alpha + r} $

It is posible to derive

$$ |det\frac{\partial f}{\partial z}| = |det(\frac{\partial z}{\partial
z} + \beta \frac{\partial h(\alpha, r)}{\partial z}(z-z_{0})^{T} + \beta
h(\alpha, r)\frac{\partial z}{\partial z})| = |det((1 + \beta h(\alpha,
r))I + \beta\frac{\partial h(\alpha, r)}{\partial z})(z-z_{0})^{T}|$$

From the
[matrix cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf),
equation (129), it can be inferred that $ \frac{\partial r}{\partial z}
= \frac{\partial ||z-z_{0}||}{\partial z} = \frac{z-z_{0}}{||z-z_{0}||}
$. The chain rule then gives $ \frac{\partial h(\alpha, r)}{\partial z}
= h'(\alpha, r)\frac{\partial r}{\partial z} = h'(\alpha,
r)\frac{z-z_{0}}{||z-z_{0}||} $. Finally, it follows that $
\frac{\partial h(\alpha, r)}{\partial z}(z-z_{0})^{T} = h'(\alpha,
r)\frac{z-z_{0}}{||z-z_{0}||}(z-z_{0})^{T}$. The jacobian
$\frac{\partial h(\alpha, r)}{\partial z})(z-z_{0})^{T}$ therefore has
rank one and can be diagonalized into

$$ \frac{\partial h(\alpha, r)}{\partial z}(z-z_{0})^{T} =
\begin{matrix} r & 0 & \cdots & 0\\ 0 & 0 & \cdots & 0\\ \ddots & \ddots
& \vdots & \ddots 0 & 0 & \cdots & 0\\ \end{matrix} $$

which yields:

$$ |det\frac{\partial f}{\partial z}| = \[1 + \beta h(\alpha,
r)]^{d-1}[1 + \beta h(\alpha, r) + \beta h'(\alpha, r)r\] $$

It is also possible to derive an analytical expression for the inverse
flow. z can be decomposed as $z = z_{0} + r\hat{z}$. Thus the expression
for $f$ can be re-arranged as:

$$ f(z) = z_{0} + r\hat{z} + \frac{\beta}{\alpha + r}r\hat{z} = z_{0} +
(1 + \frac{\beta}{\alpha + r})r\hat{z}$$.

By denoting $z' = f(z)$, it follows that:

$$ \hat{z} = \frac{z' - z_{0}}{r(1 + \frac{\beta}{\alpha + r})} $$

Determining $r$ as a function of $z'$ would then allow an analytical
expression of the inverse function.

Taking the norm of the re-arrangement of the original function yields,
as ||\hat{z}|| = 1 by definition:

$$ ||z' - z_{0}|| = (1 + \frac{\beta}{\alpha + r})r||\hat{z}|| = (1 +
\frac{\beta}{\alpha + r})r $$

Thus

$$ ||z' - z_{0}||(\alpha + r) = r(\alpha + r) + r\beta $$

That can be reorganized in a second degree equation in $r$:

$$ r^{2} + r(\alpha + \beta - ||z' + z_{0}||) - \alpha||z' + z_{0}|| = 0
$$

which has for solution, with the notation $k = ||z' + z_{0}||$

$$ r = \frac{-(\alpha + \beta - k) + \sqrt{(\alpha + \beta - k)^{2} +
4k\alpha}}{2} $$

Finally, the inverse function can be expressed as, with the above
expression for r as:

$$ f^{-1}(z') = z = z_{0} + \frac{z' - z_{0}}{1 + \frac{\beta}{\alpha +
r}} $$


#### Radial Flow Visualization

From the same 2D unit gaussian, one can apply radial flows to shape new
distributions.

After a single flow, which focus the distribution around the point
`(0.75, 0.75)`:

![1 Radial Flow Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/1radial_flows_3D.png)

![1 Radial Flow Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/1radial_flows_contour.png)

After applying an other flow which dilates the distribution around the
point `(0.85, 0.85)`:

![2 Radial Flow Unit Gaussian 3D](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/2radial_flows_3D.png)

![2 Radial Flow Unit Gaussian Contour](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/2radial_flows_contour.png)

__Note__: I wrote a script to understand the influence of the flow on
the resulting distribution, see below:

For the focusing of a point

![Singular Radial Flow Factor Arg Focus](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_radial_flow_focus_arg.png)

![Singular Radial Flow Factor Focus](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_radial_flow_focus_exp.png)

For the dilatation around a point

![Singular Radial Flow Factor Arg Dila](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_radial_flow_dila_arg.png)

![Singular Radial Flow Factor Dila](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/single_radial_flow_dila_exp.png)


## Normalizing Flows as Neural Network


## Evaluating Variational Inference

### Examples

#### 1D

We consider a model with a single parameter $\theta$:

$$ y_{i} = \theta_{i}^{2} + \epsilon_{i} $$

Where $ \epsilon_{i} \sim \mathcal{N}(0, \sigma^{2}) $, which means that
$ y_{i} | \theta_{i} \sim \mathcal{N}(\theta_{i}^{2}, \sigma^{2}) $

It will be assumed that theta is sampled from a univariate gaussian
prior:

$$ \theta \sim \mathcal{N}(0,1) $$

By Bayes theorem, the true posterior can be expressed:

$$ p(\theta_{i}|y_{i}) \propto p(y_{i}|\theta_{i})p(\theta_{i}) =
\mathcal{N}(y_{i}|\theta_{i}^{2}, \sigma^{2})\mathcal{N}(\theta_{i}|0,1)
$$

The following is obtained for $ y = 0.5 $, $ \sigma^{2} = 0.1 $

![Two hills VI Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/two_hills_mf_distribution.png)

![Two hills PF Distribution q0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/two_hills_pf_distribution_q0.png)

![Two hills PF Distribution qk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/two_hills_pf_distribution_qk.png)


#### 2D

__Figure Eight__

Let's consider a mixture of gaussians with a 2D parameter vector $
\theta $ sampled from an improper uniform prior.

$$ \theta \sim \mathcal{U} $$

Considering a gaussian mixture likelihood:

$$ p(y|\theta) = 0.5\mathcal{N}(\theta|\mu_{1},\Sigma) +
0.5\mathcal{N}(\theta|\mu_{2},\Sigma) $$

Then the posterior is also a mixture of gaussians:

$$ p(\theta|y) \propto p(y|\theta) =
0.5\mathcal{N}(\theta|\mu_{1},\Sigma) +
0.5\mathcal{N}(\theta|\mu_{2},\Sigma) $$

illustrated in the following figure:

![Figure Eight Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/figure_eight_distribution.png)

![Figure Eight VI Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/figure_eight_mf_distribution.png)

![Figure Eight PF Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/figure_eight_pf_distribution.png)

The two following figures illustrate how the flows warp the initial
distribution to fit the posterior more closely. The original gaussian,
$q0$, when sampled, generates the following set of points

![Figure Eight Posterior z0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/figure_eight_pf_posterior_z0.png)

While after the flows, the resulting distribution $qk$, generates:

![Figure Eight Posterior zk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/figure_eight_pf_posterior_zk.png)

__Banana__

![Banana Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/banana_distribution.png)

![Banana Posterior z0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/banana_pf_posterior_z0.png)

![Banana Posterior zk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/banana_pf_posterior_zk.png)

![Banana PF Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/banana_pf_distribution.png)

__Circle__

![Circle Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/circle_distribution.png)

![Circle Posterior z0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/circle_pf_posterior_z0.png)

![Circle Posterior zk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/circle_pf_posterior_zk.png)

__Energy Functions__

![EF Densities](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/energy_densities.png)


#### Higher Dimensions

__Eight Schools__

The Eight Schools model is the simplest Bayesian hierarchical normal
model. Given a set of schools in which a treatment is tested, each of
them has to report the mean effect of the treatment $y_{i}$ and its
associated standard deviation $\sigma_{i}$. All treatments are supposed
independent from each other.

The following model postulates that the mean effect of treatment $y_{i}$
is sampled from a normal distribution centered around the true
__latent__ mean effect of the treatment, $\theta_{i}$:

$$ y_{i} | \theta_{i} \sim \mathcal{N}(\theta_{i}, \sigma_{i}^{2}) $$

These latent mean effect of treatment are themselves sampled from a
shared normal distribution (the same treatment being tested in different
schools) with parameters $\mu$ and $\tau$:

$$ \theta_{i} | \mu, \tau \sim \mathcal{N}(\mu, \tau^{2}) $$

Such that the parameters themselves have priors

$$ \mu \sim \mathcal{N}(0,5) $$ $$ \tau \sim \text{Half-Cauchy}(0,5) $$

Representing the model as a graphical model can be useful to understand
the model.

![Eight Schools Graphical Model](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_gm.png)

Indeed, it makes it obvious that

$$ p(y_{i}, \theta_{i}, \sigma_{i}, \mu, \tau) = p(y_{i}| \theta_{i},
\sigma_{i})p(\theta_{i} | \mu, \tau)p(\mu)p(\tau) =
\mathcal{N}(y_{i}|\theta_{i},\sigma_{i}^{2})\mathcal{N}(\theta_{i}|\mu,\tau^{2})\mathcal{N}(\mu|0,5)\text{Half-Cauchy}(\tau|0,5)$$

And thus the posterior distribution can be formulated as:

$$ p(\theta_{i}, \mu, \tau | y_{i}, \sigma_{i}) \propto p(y_{i}|
\theta_{i}, \sigma_{i})p(\theta_{i} | \mu, \tau)p(\mu)p(\tau) =
\mathcal{N}(y_{i}|\theta_{i},\sigma_{i}^{2})\mathcal{N}(\theta_{i}|\mu,\tau^{2})\mathcal{N}(\mu|0,5)\text{Half-Cauchy}(\tau|0,5)
$$

In the case of standard variational inference, the approximated
posterior will be:

$$ \theta_{i}, \mu, \tau \sim \mathcal{N}(m, \Sigma) $$

The following figure shows the learned distributions for the latent
variable $\theta$ and the parameters $\mu$ and $\tau$

![Eight Schools MF Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_mf_distributions.png)

And the following displays the scatter of samples from q for $log(\tau)$
(x-axis) and $\theta$ (y-axis):

![Eight Schools MF Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_mf_theta_logtau.png)

for non-centered model:

![Eight Schools MF Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_mf_distributions.png)

![Eight Schools MF Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_mf_theta_logtau_noncentered.png)


Using normalizing flows allows a closer fit, especially because it
allows the original gaussian approximating $\tau$ to fit the true
half-cauchy prior very closely. As can be seen in the following:

For $q0$

![Eight Schools PF Distribution q0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_distributions_q0.png)

For $qk$

![Eight Schools PF Distribution qk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_distributions_qk.png)

![Eight Schools PF Theta Logtau](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_theta_logtau.png)

For 64 flows:

![Eight Schools PF Distribution q0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_distributions_q0_nflows64.png)

![Eight Schools PF Distribution qk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_distributions_qk_nflows64.png)

![Eight Schools PF Theta Logtau](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_theta_logtau_nflows64.png)

For 128 flows:

![Eight Schools PF Distribution q0](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_distributions_q0_nflows128.png)

![Eight Schools PF Distribution qk](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_distributions_qk_nflows128.png)

![Eight Schools PF Theta Logtau](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/eight_schools_pf_theta_logtau_nflows128.png)


# Unanswered Questions

* Why does Bishop say that the variance of $q(z)$ is controlled by the
  direction of smalles variance of $p(z)$? (page 467 in the paragraph
  after equation (10.15)) 🆗

>I implemented my own version of that mean field approxmation, and it
>results that the approximated posterior q(z) has variance: $$ Var(q(z))
>= \begin{bmatrix} \Lambda_{11} & 0\\ 0 & \Lambda_{22} \end{bmatrix} $$
>Resulting in the following distributions:

>![Comparison q,p for variance capture](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/pz_qz_variance_capture.png)

> My mistake was coming from the use of the precision matrix. In bishop
> $ \Lambda $ is the precision matrix (i.e $ \Sigma = \Lambda^{-1} $).
> Therefore considering:

> $$ q(z) = q_{1}(z_{1})q_{2}(z_{2}) = \mathcal{N}(z1|m_{1},
> \Lambda_{11}^{-1})\mathcal{N}(z2|m_{2}, \Lambda_{22}^{-1}) $$

> is equivalent to having $ q(z) = \mathcal{N}(z|m,\Sigma) $ with:

> $$ \Sigma = \begin{bmatrix} \frac{1}{\Lambda_{11}} & 0\\ 0 &
> \frac{1}{\Lambda_{22}} \end{bmatrix} $$

> where $ \Lambda_{11} $ is the element at indices 1,1 in the precision
> matrix. It gives the resulting result:

>![Comparison q,p for variance capture, correct](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/pz_qz_variance_capture_true.png)

* How can one draw the contour plot of a mixture of multivariate
  gaussians in mean field approximation?

>The parameters of the different mixture elements have a known
>distribution (see the first part for that), but how can one use the
>following ? $$ p(\mu | \Lambda)p(\Lambda) = p(\mu , \Lambda) =
>\prod_{k=1}^{K}\mathcal{N}(\mu_{k} | m_{0},
>(\beta_{0}\Lambda_{k})^{-1})\mathcal{W}(\Lambda_{k} | W_{0}, \nu_{0})
>$$ In my implementation, I cannot access directly the Wishart component
>$ \Lambda_{k} $.

> __TODO__ add derivation and formulation of pdf as in written notes.

* What dataset will be used for testing the neural network
  implementation ?

>First some test distributions can be used to verify the ability of the
>flow to fit non-trivial distributions. Then, it would be useful to use
>a broad dataset to train a VAE where the encoder is enriched with
>normalizing flows (ex CIFAR10?).

* What non-trivial distributions could be used to test the ability of a
  finite set of flows to fit? 🆗

>[Yes but did it work?]() presents a way to conduct a diagnostic of how
>close an approximated posterior distribtion fits, but I am not sure yet
>on what to use from that. Otherwise I implemented a
>[banana distribution](https://github.com/pierresegonne/VINF/blob/master/Code/funky%20distributions/banana.py)
>and a
>[circle distribution](https://github.com/pierresegonne/VINF/blob/master/Code/funky%20distributions/circle.py).
>![Banana Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/banana_distribution.png)

>![Circle Distribution](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/circle_distribution.png)

* Are there any other types of finite flow we could consider?

>After going through relevant conferences from
>[ICML Normalizing Flow Workshop](https://icml.cc/Conferences/2019/ScheduleMultitrack?event=3521),
>no type of other flow was discussed, only some extension of planar flow
>presented here
>[video](https://slideslive.com/38917902/householder-meets-sylvester-normalizing-flows-for-variational-inference),
>[paper](https://arxiv.org/pdf/1803.05649.pdf)

* Is there a way to prove that the planar flow for example results in a
  normalized distribution? 🆗

>To show that it is the case, one would need to prove $
>\int_{-\inf}^{\inf} q_{1}(z_{1})dz_{1} = 1$

>From the definition, for a single planar flow,

>$$ \int_{-\inf}^{\inf} q_{1}(z_{1})dz_{1} = \int_{-\inf}^{\inf}
>q_{0}(f^{-1}(z_{1}))|1+u^{T}h'(w^{T}f^{-1}(z_{1})+b)w|^{-1}dz_{1} $$

>It would therefore be necessary for that approach to know $ f^{-1} $,
>which is not trivial to find for e.g $h(x) = tanh(x)$.

> So in a specific case it's analytically difficult to prove that. But
> in general, because of how we defined q1 the following holds:

> $$ \int_{-\inf}^{\inf} q_{1}(z_{1})dz_{1} = \int_{-\inf}^{\inf}
> q_{1}(z_{1}) \frac{dz_{1}}{dz_{0}}dz_{0} = \int_{-\inf}^{\inf}
> q_{1}(z_{1})f'(z_{0})dz_{0} = \int_{-\inf}^{\inf} q_{0}(z_{0})dz_{0} =
> 1 $$


* For learning a given distribution, what loss can be used for
  optimization of flow parameters?

> The variational free energy can be used, if one can infer the negative
> log likelihood of the data and provide a prior for the latent
> variables.

* If these flows are used in the setting of a VAE, what kind of loss can
  be used?
> Basic VAE uses ELBO.

* What's wrong with my implementation of the Radial Flows?

> I have tested it on energy 1 thinking that it would be make it easy to
> understand what is happening. Unfortunately, when looking at the log of the energy function, the two kidneys don't appear clearly, it's pretty much a circle.
> Nevertheless that does not prevent the radial flows from learning, even ifµ it gets stuck in a local minimum. __Should try with banana__.
> Should create a little tool to plot the location of the z0 and the direction of influence.

> TODO ask Michael if it could have an influence?

> I have noted that the instability of training decreases with a greater number of flows and a greater number of samples.

# TODO

* Finish mean field mixture study

* Implement Radial Flows

* Refactor to use bijectors
  [to check](https://blog.evjang.com/2018/01/nf1.html)
  [to check](https://github.com/LukasRinder/normalizing-flows)

# MISC

[blog post with pytorch implementation of flows](https://www.ritchievink.com/blog/2019/10/11/sculpting-distributions-with-normalizing-flows/)

[ICML Normalizing Flow Workshop](https://icml.cc/Conferences/2019/ScheduleMultitrack?event=3521)

[Bayesian Modelling Cookbook](https://eigenfoo.xyz/bayesian-modelling-cookbook/)

[Tutorial PyMC3 Eight Schools](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html)

From tf.Bijector doc:

> We take the absolute value of the determinant before log to avoid NaN
> values. Geometrically, a negative determinant corresponds to an
> orientation-reversing transformation. It is ok for us to discard the
> sign of the determinant because we only integrate
> everywhere-nonnegative functions (probability densities) and the
> correct orientation is always the one that produces a nonnegative
> integrand

# References

1. Rezende, Danilo Jimenez, and Shakir Mohamed. "Variational inference
   with normalizing flows." arXiv preprint arXiv:1505.05770 (2015).

```
@article{rezende2015variational,
  title={Variational inference with normalizing flows},
  author={Rezende, Danilo Jimenez and Mohamed, Shakir},
  journal={arXiv preprint arXiv:1505.05770},
  year={2015}
}
```

2. Yao, Yuling, et al. "Yes, but did it work?: Evaluating variational
   inference." arXiv preprint arXiv:1802.02538 (2018).

```
@article{yao2018yes,
  title={Yes, but did it work?: Evaluating variational inference},
  author={Yao, Yuling and Vehtari, Aki and Simpson, Daniel and Gelman, Andrew},
  journal={arXiv preprint arXiv:1802.02538},
  year={2018}
}
```

3. Vehtari, Aki, et al. "Pareto smoothed importance sampling." arXiv
   preprint arXiv:1507.02646 (2015).

```
@article{vehtari2015pareto,
  title={Pareto smoothed importance sampling},
  author={Vehtari, Aki and Simpson, Daniel and Gelman, Andrew and Yao, Yuling and Gabry, Jonah},
  journal={arXiv preprint arXiv:1507.02646},
  year={2015}
}
```

4. Kucukelbir, Alp, et al. "Automatic differentiation variational
   inference." The Journal of Machine Learning Research 18.1 (2017):
   430-474.

```
@article{kucukelbir2017automatic,
  title={Automatic differentiation variational inference},
  author={Kucukelbir, Alp and Tran, Dustin and Ranganath, Rajesh and Gelman, Andrew and Blei, David M},
  journal={The Journal of Machine Learning Research},
  volume={18},
  number={1},
  pages={430--474},
  year={2017},
  publisher={JMLR. org}
}
```

5. Bishop, Christopher M. Pattern recognition and machine learning.
   springer, 2006.

```
@book{bishop2006pattern,
  title={Pattern recognition and machine learning},
  author={Bishop, Christopher M},
  year={2006},
  publisher={springer}
}
```

6. Papamakarios, George, Theo Pavlakou, and Iain Murray. "Masked
   autoregressive flow for density estimation." Advances in Neural
   Information Processing Systems. 2017.

```
@inproceedings{papamakarios2017masked,
  title={Masked autoregressive flow for density estimation},
  author={Papamakarios, George and Pavlakou, Theo and Murray, Iain},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2338--2347},
  year={2017}
}
```

7. Kingma, Durk P., et al. "Improved variational inference with inverse
   autoregressive flow." Advances in neural information processing
   systems. 2016.

```
@inproceedings{kingma2016improved,
  title={Improved variational inference with inverse autoregressive flow},
  author={Kingma, Durk P and Salimans, Tim and Jozefowicz, Rafal and Chen, Xi and Sutskever, Ilya and Welling, Max},
  booktitle={Advances in neural information processing systems},
  pages={4743--4751},
  year={2016}
}
```

8. Oord, Aaron van den, et al. "Parallel wavenet: Fast high-fidelity
   speech synthesis." arXiv preprint arXiv:1711.10433 (2017).

```
@article{oord2017parallel,
  title={Parallel wavenet: Fast high-fidelity speech synthesis},
  author={Oord, Aaron van den and Li, Yazhe and Babuschkin, Igor and Simonyan, Karen and Vinyals, Oriol and Kavukcuoglu, Koray and Driessche, George van den and Lockhart, Edward and Cobo, Luis C and Stimberg, Florian and others},
  journal={arXiv preprint arXiv:1711.10433},
  year={2017}
}
```

