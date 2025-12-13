Prepared by: Paloma Dettloff, University of Washington Applied Physics Laboratory
Contact: palomad1@uw.edu
Project Overview
This project implements a small, low-power acoustic velocity sensor designed for Autonomous Underwater Vehicles (AUVs), particularly gliders. The goal is to measure through-water velocity using Doppler-based backscatter from water particles rather than relying on the ocean floor, enabling accurate velocity measurements even in deep water or under extreme pitch/roll conditions.
The core of the project is a 1D UNet neural network denoiser trained to extract velocity information from noisy acoustic pulse returns.
Setup Instructions

1. Clone the repository:

```bash
git clone <repo-url>
cd <repo-directory>

2. Create a Python virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Create folders for outputs and checkpoints:
mkdir -p checkpoints results demo

How to Run

Run the demo script:
python demo/demo.py

he demo will:

Generate or load noisy pulse train data.

Apply the UNet1D denoiser.

Save figures in the results/ folder.

Save model checkpoints in the checkpoints/ folder.\

torch.save(model.state_dict(), "checkpoints/unet_epoch10.pt")
plt.savefig("results/example.png")
