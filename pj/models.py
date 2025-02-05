from django.db import models

class User(models.Model):
    username = models.CharField(max_length=8, unique=True)
    mail = models.EmailField(unique=True)
    password = models.CharField(max_length=24)
