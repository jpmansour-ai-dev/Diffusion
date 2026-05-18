# Diffusion models

Denoising Diffusion Probabilistic Models were first introduced in 2020 in the original paper "[Denoising Diffusion Probabilistic Models - Ho, Jain and Abbeel](https://arxiv.org/abs/2006.11239)" as a powerful class of deep learning generative models, demonstrating impressive results on image, video and audio generation, as well as in one molecule synthesis, and any domain that benefits from synthetic data generation. They have since become the foundation of most generative models we see today, such as DALL-E 3 (OpenAI), Sora (OpenAI) and Imagen 3 (Google).

The aim here is to give a basic demonstration of what diffusion models are capable of, by training a relatively small prediction noise neural network engine, but also to get a solid understanding of how these models operate under the hood, and the brilliancy behind its simplicity. For reference, the official implementation of diffusion models is accessible here: [Official GitHub Repository](https://github.com/hojonathanho/diffusion). In addition, the implementation provided here is based heavily on this [Repository](https://github.com/lucidrains/denoising-diffusion-pytorch)

### Concept 
The idea consists on smoothly perturbating the original image data $x_0$ by iteratively adding Gaussian noise until reaching pure noise $x_T$, then learning to reverse this process to generate new data from pure noise. Here is what noising looks like:

INSERT NOISING IMAGE HERE

**Forward Diffusion:**

Let:
- $x_0$ be an image sampled from a distribution $q$
- $\beta_1, \dots, \beta_T \in (0, 1)$ be a known variance schedule

The forward diffusion process for $t = 1, 2, \dots, T$ is defined as:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t ; \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t \mathbf{I})$$

**Properties:**
- The diffusion process is a Markov chain, meaning the noised image at timestep $x_t$ depends only on the previous step $x_{t-1}$, and not on any earlier steps $x_{t-2}, \dots, x_0$.
- It can be shown that we can sample $x_t$ directly from $x_0$ in closed form:

$$q(x_t \mid x_0) = \mathcal{N}(x_t ; \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t) \mathbf{I})$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 





Self-supervised learning generates its own pseudo-labels or training signals directly from the structure of the data. 


