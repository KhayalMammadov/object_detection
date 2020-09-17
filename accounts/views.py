from django.shortcuts import render, redirect
from django.views.generic import CreateView
from .forms import UserSignUpForm
from .models import User
from django.contrib.auth import login
# Create your views here.


class UserSignUpView(CreateView):
    model = User
    template_name = 'registration/signup.html'
    form_class = UserSignUpForm

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('detection:home')
