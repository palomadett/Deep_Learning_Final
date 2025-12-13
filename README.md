Prepared by: Paloma Dettloff, University of Washington Applied Physics Laboratory
Contact: palomad1@uw.edu
Project Overview
## Project Overview

This repository focuses on a 1D UNet neural network for denoising acoustic pulse returns. The network is trained to suppress noise and artifacts in Doppler-based acoustic signals, improving the extraction of velocity information from noisy measurements.

The motivation for this work comes from the development of a small, low-power acoustic velocity sensor for Autonomous Underwater Vehicles (AUVs), particularly gliders. In that application, through-water velocity is estimated using Doppler backscatter from water particles rather than reflections from the ocean floor. Accurate signal denoising is critical for making reliable velocity estimates under low SNR conditions, long pulse repetition intervals, and varying vehicle orientation.

This repository contains only the neural network model, training code, and demos.


1. Clone the repository:

```bash
git clone <repo-url>
cd <repo-directory>
```

2. Create a Python virtual environment:
```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Create folders for outputs and checkpoints:
```
mkdir -p checkpoints results demo
```

5. How to Run
```
Run the demo script:
python demo/demo.py
```

The demo will:

Generate or load noisy pulse train data.

Apply the UNet1D denoiser.

Save figures in the results/ folder.

Save model checkpoints in the checkpoints/ folder.\

