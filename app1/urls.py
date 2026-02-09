from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .views import attendance_chart, student_logs
urlpatterns = [
    path('capture_Student/', views.capture_Student, name='capture_Student'),
    path('', views.home, name='home'),
    path('selfie-success/', views.selfie_success, name='selfie_success'),
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('Student/attendance/', views.Student_attendance_list, name='Student_attendance_list'),
    path('Student/', views.Student_list, name='Student-list'),
    path('Student/<int:pk>/', views.Student_detail, name='Student-detail'),
    path('Student/<int:pk>/authorize/', views.Student_authorize, name='Student-authorize'),
    path('Student/<int:pk>/delete/', views.Student_delete, name='Student-delete'),
    path('Students/<int:pk>/', views.Student_edit, name='Student-edit'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),
    path('attendance-chart/', views.attendance_chart, name='attendance_chart'),
    path('User_login/', views.student_login, name='student_login'),
    path('dashboard/', views.student_dashboard, name='student_dashboard'),
    # path('attendance/present/', views.present_students, name='present_students'),
    # path('attendance/absent/', views.absent_students, name='absent_students'),
    path('logout1/', views.logout_view, name='logout1'),
    path('logs/<str:stu_id>/',views.student_logs, name='student_logs'),
    path('check-student-id/', views.check_student_id, name='check_student_id'),
]
    

