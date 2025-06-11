# torch-ddpm-sampler

A PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating high-quality images from pure Gaussian noise using reverse diffusion. This repo walks through the complete forward and reverse diffusion process, training loop, and image sampling in an educational and customizable way.

---

## 🌠 What is a DDPM?

DDPMs generate images by **reversing a Markovian diffusion process**. They first add noise to data over many steps (forward process), and then train a neural network to remove that noise (reverse process). Once trained, the model can sample new data from pure noise.

---

## 🔍 Implementation Details

### Forward Diffusion Process (Noise Schedule)

We gradually add noise to an image over `T` steps using a predefined schedule of βₜ values:

\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot I)
\]

We can sample any `x_t` from `x_0` directly using:

\[
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

Where:

- \(\alpha_t = 1 - \beta_t\)
- \(\bar{\alpha}_t = \prod_{s=1}^t \alpha_s\)

---

### Reverse Process (Denoising with UNet)

We train a neural network \( \epsilon_\theta(x_t, t) \) to predict the noise added at each step:

\[
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
\]

Then we estimate the clean image from:

\[
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t) \right)
\]

And use the reverse sampling step:

\[
p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)
\]

---

### 🧠 Model Architecture

The model is a **lightweight UNet**, with 3 convolutional layers:

```python
class UNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, t):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        return self.conv3(h)
