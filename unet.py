import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Contracting Path (Encoder)
        self.enc_conv1 = self.conv_block(in_channels, 64)
        self.enc_conv2 = self.conv_block(64, 128)
        self.enc_conv3 = self.conv_block(128, 256)
        self.enc_conv4 = self.conv_block(256, 512)
        
        # Bottom
        self.bottom_conv = self.conv_block(512, 1024)
        
        # Expanding Path (Decoder)
        self.up_conv4 = self.upconv(1024, 512)
        self.dec_conv4 = self.conv_block(1024, 512)  # Concatenation, hence double input channels
        
        self.up_conv3 = self.upconv(512, 256)
        self.dec_conv3 = self.conv_block(512, 256)
        
        self.up_conv2 = self.upconv(256, 128)
        self.dec_conv2 = self.conv_block(256, 128)
        
        self.up_conv1 = self.upconv(128, 64)
        self.dec_conv1 = self.conv_block(128, 64)
        
        # Final Output Layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(F.max_pool2d(enc1, 2))
        enc3 = self.enc_conv3(F.max_pool2d(enc2, 2))
        enc4 = self.enc_conv4(F.max_pool2d(enc3, 2))
        
        # Bottom
        bottom = self.bottom_conv(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.up_conv4(bottom)
        dec4 = torch.cat((enc4, dec4), dim=1)  # Skip connection
        dec4 = self.dec_conv4(dec4)
        
        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)  # Skip connection
        dec3 = self.dec_conv3(dec3)
        
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)  # Skip connection
        dec2 = self.dec_conv2(dec2)
        
        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)  # Skip connection
        dec1 = self.dec_conv1(dec1)
        
        # Final output
        out = self.out_conv(dec1)
        return out
    
    # Function to create a block of 2 convolutional layers
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    # Function to perform upsampling followed by a convolution
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

# Example of using UNet for one of your models (e.g., Model A or Model B)
class FusionModelUNet(nn.Module):
    def __init__(self):
        super(FusionModelUNet, self).__init__()
        self.model_a = UNet(in_channels=3, out_channels=1)  # For photo input (RGB, 3 channels)
        self.model_b = UNet(in_channels=1, out_channels=1)  # For x-ray input (grayscale, 1 channel)
        self.fc1 = nn.Linear(224 * 224 * 2, 64)  # Fusion layer
        self.fc2 = nn.Linear(64, 1)  # Final decision layer

    def forward(self, photo, xray):
        seg_photo = self.model_a(photo)  # Output segmentation mask from photo
        seg_xray = self.model_b(xray)    # Output segmentation mask from x-ray
        combined = torch.cat((seg_photo.view(-1), seg_xray.view(-1)), dim=0)  # Flatten and concatenate
        combined = F.relu(self.fc1(combined))  # Fusion layer
        output = torch.sigmoid(self.fc2(combined))  # Output decision between 0 and 1
        return output

# Example usage
if __name__ == "__main__":
    # Instantiate the model
    model = FusionModelUNet()
    
    # Example input (dummy data)
    photo_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels (RGB), 224x224
    xray_input = torch.randn(1, 1, 224, 224)   # Batch size 1, 1 channel (grayscale), 224x224
    
    # Forward pass
    output = model(photo_input, xray_input)
    
    print("Output (Need for extraction, probability):", output.item())
