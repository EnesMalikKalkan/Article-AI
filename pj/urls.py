from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('giris_kayit/', views.kayit),
    path('usridentification/', views.addUser, name='identificate'),
    path('search/', views.validateUser, name='validate'),
    path('edit_profile/', views.editProfile, name='edit'),
    path('results/', views.listResults, name='results'),
    path('show/', views.showFullPage, name='show'),
    path('update/', views.updateUser, name='update'),
    path('recommendation', views.showFullPageRecommendation, name='rec'),
]