from django.db import models
from accounts.models import User
# Create your ml here.


# class DetectionModel(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detection_model')
#     triggered_time = models.DateTimeField()
#     trained_time = models.DateTimeField()
#
#     def __str__(self):
#         return self.user.username + "_" + str(self.trained_time)


class Image(models.Model):
    file = models.ImageField(upload_to='ml/users/images/%Y%m%d')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name


class Image2User(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.ForeignKey(Image, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.user.username) + str(self.file)


class TrainedModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trained_model')
    model_name = models.CharField(max_length=30, unique=True)
    class_names = models.CharField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id) + "_" + str(self.user.username) + "_" + str(self.model_name)


class DetectedImage(models.Model):
    trained_model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='detected_image')
    image_url = models.FilePathField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id) + "_" + str(self.trained_model.user.username) + "_" + str(self.trained_model.model_name)


class MaskInfo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mask_info')
    class_name = models.CharField(max_length=15)
    image_url = models.FilePathField(max_length=300)
    mask_path = models.CharField(max_length=1000)
    is_train = models.BooleanField(default=True)

    def __str__(self):
        return str(self.id) + "_" + str(self.user.username) + "_" + str(self.class_name)


class SavedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_image')
    file = models.FilePathField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id) + "_" + str(self.user.username) + "_" + str(self.file)

