class UNet1D_DenoiseOnly(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, features=32):
        super().__init__()

        # ---------------------------------------------
        # ENCODER: progressively downsamples the signal
        # and extracts higher-level features.
        # ---------------------------------------------

        # First encoder block (no downsampling yet)
        self.enc1 = self._res_block(in_ch, features)
        #Downsample: halves time resolution, doubles receptive field
        self.down1 = nn.Conv1d(features, features, 3, stride=2, padding=1)

        #Second encoder block
        self.enc2 = self._res_block(features, features * 2)
        self.down2 = nn.Conv1d(features * 2, features * 2, 3, stride=2, padding=1)

        #Third encoder block 
        self.enc3 = self._res_block(features * 2, features * 4)
        self.down3 = nn.Conv1d(features * 4, features * 4, 3, stride=2, padding=1)

        # ---------------------------------------------
        # BOTTLENECK: the "compressed understanding"
        # Very abstract features → global structure
        # ---------------------------------------------
        self.bottleneck = nn.Sequential(
            # A deeper residual block for rich internal features
            self._res_block(features * 4, features * 8),
            # Upsample back toward decoder path
            nn.ConvTranspose1d(features * 8, features * 4, 3,
                               stride=2, padding=1, output_padding=1)
        )

        # ---------------------------------------------
        # DECODER: progressively upsamples the signal
        # and blends encoder features (skip connections)
        # to reconstruct a clean waveform.
        # ---------------------------------------------

        # First decoder stage (mirrors enc3)
        self.dec3 = nn.Sequential(
            self._res_block(features * 8, features * 4),  # Because of skip: 4+4 = 8 channels
            nn.ConvTranspose1d(features * 4, features * 2, 3,
                               stride=2, padding=1, output_padding=1),
        )

        # Second decoder stage (mirrors enc2)
        self.dec2 = nn.Sequential(
            self._res_block(features * 4, features * 2),   # 2 (upsampled) + 2 (skip)
            nn.ConvTranspose1d(features * 2, features, 3,
                               stride=2, padding=1, output_padding=1),
        )

        # Final decoder block (mirrors enc1)
        self.dec1 = nn.Sequential(
            self._res_block(features * 2, features),       # 1 (upsampled) + 1 (skip)
        )

        # Final output projection to waveform
        self.out = nn.Conv1d(features, out_ch, 1)

        # Output activation — keeps predictions in [-1, 1]
        self.tanh = nn.Tanh()


    # ---------------------------------------------------------
    # Residual block:
    # - two 1D convolutions
    # - ReLU activations
    # - captures local patterns in waveform
    # ---------------------------------------------------------
    def _res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )


    def forward(self, x):

        # ENCODER---------------------------------------
        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        
        # BOTTLENECK------------------------------------
        x = self.bottleneck(x)

        # DECODER + SKIPS---------------------------------
        # Merge bottleneck with encoder features
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        #Final waveform prediction
        x = self.out(x)

        #Force final output between [-1, 1]
        return self.tanh(x)

