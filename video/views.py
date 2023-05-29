from django.shortcuts import render
from .models import Video
from .forms import VideoUploadForm
#from .utils import process_frame
#from .crowd_counting_model import CrowdCountingModel
# Import PyTorch and MathPlot lib
import torch
import numpy as np
import cv2
from torchvision import models, transforms
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
import torch.nn as nn
import torchvision.models as models

# Check PyTorch version
torch.__version__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)



# Channel Attention Module
class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.Softmax(dim=-1)(energy)
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma*out + x
        return out

# Position Attention Module
class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.Softmax(dim=-1)(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        out = torch.bmm(attention, proj_value).permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)
        out = out.view(batch_size, channels, height, width)
        return out

# Revised Spatial Attention
class RevisedSpatialAttention(nn.Module):
    def __init__(self, in_dim):
        super(RevisedSpatialAttention, self).__init__()
        self.cam_module = CAM_Module()
        self.pam_module = PAM_Module(in_dim)

    def forward(self, x):
        cam_out = self.cam_module(x)
        pam_out = self.pam_module(x)
        return cam_out + pam_out

# Front-end
class FrontEnd(nn.Module):
    def __init__(self):
        super(FrontEnd, self).__init__()
        
        # Load pre-trained VGG-16
        vgg = models.vgg16(pretrained=False)
        # FrontEnd
        self.front_end = nn.Sequential(
            # 1st conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2nd conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3rd conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4th conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Copy weights from VGG16
        for i, layer in enumerate(vgg.features[:23]):
            self.front_end[i] = layer

    def forward(self, x):
        x = self.front_end(x)
        return x

# Hierarchical Dense Dilated Deep Pyramid Feature extraction Convolutional Neural Network (HDPF-CNN)
class HDPF(nn.Module):
    def __init__(self, in_channels):
        super(HDPF, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4

# Back-end
class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2)
        self.relu6 = nn.ReLU(inplace=True)
        
        # Add Revised Spatial Attention
        self.revised_spatial_attention = RevisedSpatialAttention(64)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        
        # Apply revised spatial attention after all the convolutions
        x = self.revised_spatial_attention(x) * x
        return x

# Output layer
class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

# Final Model
class CrowdCountingModel(nn.Module):
    def __init__(self):
        super(CrowdCountingModel, self).__init__()
        self.frontend = FrontEnd()
        self.hdpf = HDPF(in_channels=512)
        self.backend = BackEnd()
        self.output_layer = OutputLayer()

    def forward(self, x):
        x = self.frontend(x)
        x1, x2, x3, x4 = self.hdpf(x)
        x = self.backend(x4)  
        x = self.output_layer(x)
        return x
    
# Create an instance of the model
model = CrowdCountingModel()
model.to(device)
print(next(model.parameters()).device)






def process_frame(model, frame):
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Transform the image to what your model expects
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame_transformed = transform(frame_rgb)
    
    # Add an extra dimension for batch size
    frame_transformed = frame_transformed.unsqueeze(0)

    # Send the frame to the device
    frame_transformed = frame_transformed.to(device)
    
    # Forward the frame through the model
    output = model(frame_transformed)

    # The output is a density map. Sum all the values in the density map
    # and return it as the count of people in the frame.
    count = round(output.sum().item()) # round the count
    count = max(0, count) # ensure the count is not less than 0

    # Convert the density map to a numpy array, suitable for display
    density_map = output.squeeze().detach().cpu().numpy()

    # Normalize the density map
    density_map = ((density_map - density_map.min()) * (1/(density_map.max() - density_map.min()) * 255)).astype('uint8')

    # Resize the density map to original frame size
    density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
    
    # Apply color map
    density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    
    return count, density_map


def process_video(model, video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    count = 0
    frame_count = 0
    total_count = 0

    while True:
        # Read the next frame
        ret, frame = video_capture.read()

        # If the frame could not be read, then we have reached the end of the video
        if not ret:
            break

        # Resize the frame to 50% of its original size
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Process the frame
        frame_count, density_map = process_frame(model, frame)
        count += 1

        if count % 10 == 0:
            total_count = frame_count
            frame_count = 0

        # Prepare the text string
        text = f'{total_count} people detected'

        # Display the count on the frame (this creates the outline)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5) # 4 is the thickness

        # Display the count on the frame (this is the main text)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # 2 is the thickness

        # Show the video
        cv2.imshow('Video', frame)

        # Show the density map
        cv2.imshow('Density Map', density_map)

        # Quit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file
    video_capture.release()

    # Destroy the video window
    cv2.destroyAllWindows()

    return count

import os
from django.conf import settings

def handle_uploaded_video(video_file):
    # Generate a unique file name
    file_name = video_file.name
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    # Open the file in write mode
    with open(file_path, 'wb') as destination:
        # Iterate over the uploaded file chunks and save them to the destination
        for chunk in video_file.chunks():
            destination.write(chunk)

    return file_path


from django.shortcuts import render
from .forms import VideoUploadForm

def process_video_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data['video']
            video_path = handle_uploaded_video(video_file)

            # Load the model and process the video
            model = CrowdCountingModel()
            model.to(device)
            #model.load_state_dict(torch.load('video/models_path/VGG16_multiscale_pyramid_Attention_Network_V2_best_6.pth'))
            model.load_state_dict(torch.load('video/models_path/VGG16_multiscale_pyramid_Attention_Network_V2_best_6.pth',map_location=torch.device('cpu')))
            model.eval()
            
            count = process_video(model, video_path)
            
            return render(request, 'result.html', {'count': count})
    else:
        form = VideoUploadForm()
    
    return render(request, 'upload.html', {'form': form})

def process_webcam(model):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    count = 0
    frame_count = 0
    total_count = 0

    while True:
        # Read the next frame from the webcam
        ret, frame = video_capture.read()

        # Resize the frame to 50% of its original size
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Process the frame
        frame_count, density_map = process_frame(model, frame)
        count += 1

        if count % 10 == 0:
            total_count = frame_count
            frame_count = 0

        # Prepare the text string
        text = f'{total_count} people detected'

        # Display the count on the frame (this creates the outline)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5) # 4 is the thickness

        # Display the count on the frame (this is the main text)
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # 2 is the thickness

        # Show the frame
        cv2.imshow('Webcam', frame)

        # Show the density map
        cv2.imshow('Density Map', density_map)

        # Quit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    video_capture.release()

    # Destroy the windows
    cv2.destroyAllWindows()

def process_webcam_view(request):
    # Load the model
    model = CrowdCountingModel()
    model.to(device)
    model.load_state_dict(torch.load('video/models_path/VGG16_multiscale_pyramid_Attention_Network_V2_best_6.pth', map_location=torch.device('cpu')))
    model.eval()

    # Process the webcam frames
    process_webcam(model)
    
    return render(request, 'welcome.html')

def welcome(request):
    return render(request, 'welcome.html')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.load_state_dict(torch.load('VGG16_multiscale_pyramid_Attention_Network_V2_best_6.pth'))
# model.eval()

# video_path = 'video6.mp4'
# process_video(model, video_path)

