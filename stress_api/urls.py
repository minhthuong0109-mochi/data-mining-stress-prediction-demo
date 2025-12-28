from django.urls import path
from .views import demo_page, predict_api

urlpatterns = [
    path("", demo_page, name="demo_page"),
    path("api/predict/", predict_api, name="predict_api"),
]
