from django.urls import path
from . import views

urlpatterns = [
    path('process_video_view/', views.process_video_view, name='process_video_view'),
    path('process_webcam_view/', views.process_webcam_view, name='process_webcam_view'),
    path('', views.welcome, name='welcome'),
    
]