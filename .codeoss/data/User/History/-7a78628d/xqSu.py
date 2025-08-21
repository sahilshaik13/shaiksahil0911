from django.contrib import admin
from django.urls import path, include
from parser import views  # your app's views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include )
]
