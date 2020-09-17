from django.urls import path, include
from .views import UserSignUpView

urlpatterns = [
    path('', include('django.contrib.auth.urls')),
]