from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your ml here.


class User(AbstractUser):

    def __str__(self):
        return self.username

