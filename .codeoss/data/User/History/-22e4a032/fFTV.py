from parser import views
urlpatterns = [
    path('', views.upload_resume, name='upload_resume'),
]
