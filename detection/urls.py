from django.urls import path, include
from .views import home, draw_mask, class_list_ajax, auto_detection, ImageUploadView, clear_database, train_model, \
    manage_images, delete_image, clear_database_masked, save_trained_image, delete_saved_image, clear_database_saved
app_name = 'detection'

urlpatterns = [
    path('home/', home, name='home'),
    path('upload/', ImageUploadView.as_view(), name='upload'),
    path('draw-mask/', draw_mask, name='draw_mask'),

    path('get-class-list/', class_list_ajax, name='get_class_list'),

    path('auto-detecting/', auto_detection, name='auto_detection'),
    path('train-model/', train_model, name='train_model'),


    path('manage-images/', manage_images, name='manage_images'),


    path('clear/', clear_database, name='clear_database'),
    path('clear-masked/', clear_database_masked, name='clear_database_masked'),
    path('clear-saved/', clear_database_saved, name='clear_database_saved'),

    path('delete_image/<int:pk>', delete_image, name='delete_image'),
    path('delete_saved_image/<int:pk>', delete_saved_image, name='delete_saved_image'),

    path('save_image/', save_trained_image, name='save_image'),
]