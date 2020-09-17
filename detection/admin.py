from django.contrib import admin
from .models import Image, TrainedModel, DetectedImage, MaskInfo, SavedImage
# Register your ml here.

admin.site.register(Image)
admin.site.register(TrainedModel)
admin.site.register(DetectedImage)
admin.site.register(MaskInfo)
admin.site.register(SavedImage)
