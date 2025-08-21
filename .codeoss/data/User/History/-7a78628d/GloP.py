from django.contrib import admin
from django.urls import path
from parser import views  # your app's views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload_resume, name='upload_resume'),
]
