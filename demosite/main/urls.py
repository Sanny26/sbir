from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results', views.results, name='results'),
    path('demo', views.inputpad, name='inputpad'),
    path('show_image', views.show_image, name='show_image'),
    #path('image', views.read_image, name='read_image'),
    re_path(r'^image/?', views.read_image, name='read_image'),
    #re_path(r'^image/(?P<datauri>\w+)/$', views.read_image, name='read_image'),
]