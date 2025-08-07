from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from .views import CustomLoginView

urlpatterns = [
    path("", views.home, name="home"),
    path('submit/', views.submit_article, name='submit_article'),
    path('analyze/', views.analyze_article, name='analyze_article'),
    path('about/', views.about, name='about'),
    path('signup/', views.signup_view, name='register'),
    path('accounts/logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('accounts/profile/', login_required(views.profile_view), name='profile'),
    path('accounts/login/', CustomLoginView.as_view(), name='login'),

]
