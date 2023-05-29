from django.db import models


class Video(models.Model):
    video = models.FileField(upload_to='videos/')
    count = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
