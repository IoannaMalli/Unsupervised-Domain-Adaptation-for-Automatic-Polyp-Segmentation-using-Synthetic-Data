class Segmentor(nn.Module):
    def __init__(self, encoder, daformer_head):
        """
        Initialize the segmentation model.

        Args:
            encoder (nn.Module): The feature extractor (e.g., mit_b5).
            daformer_head (nn.Module): The decoder (e.g., DAFormerHead).
        """
        super(Segmentor, self).__init__()
        self.encoder = encoder
        self.decoder = daformer_head

    def forward(self, x):
        """
        Forward pass for the segmentation model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor with predicted segmentation map.
        """
        # Encoder: extract multi-scale features
        features = self.encoder(x)
        # for i, feature in enumerate(features):
        #   print(f"Feature {i}: {feature.shape}")
        # Decoder: predict segmentation map
        segmentation_map = self.decoder(features)

        return segmentation_map
