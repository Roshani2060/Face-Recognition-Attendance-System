from django.contrib import admin
from .models import Student, Attendance, CameraConfiguration, AttendanceLog

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone_number', 'Student_Id', 'authorized']
    list_filter = ['Student_Id', 'authorized']
    search_fields = ['name', 'email']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'date', 'check_in_time', 'check_out_time', 'extra_time']
    list_filter = ['date']
    search_fields = ['Student__name']

    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing an existing object
            return ['Student', 'date', 'check_in_time', 'check_out_time', 'extra_time']
        else:  # Adding a new object
            return ['date', 'check_in_time', 'check_out_time', 'extra_time']

    def save_model(self, request, obj, form, change):
        if change:  # Editing an existing object
            existing_attendance = Attendance.objects.get(id=obj.id)
            obj.check_in_time = existing_attendance.check_in_time
            obj.check_out_time = existing_attendance.check_out_time
            obj.extra_time = existing_attendance.extra_time  # Preserve extra_time as well
        super().save_model(request, obj, form, change)

@admin.register(CameraConfiguration)
class CameraConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_source', 'threshold']
    search_fields = ['name']


@admin.register(AttendanceLog)
class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = ('attendance', 'action', 'timestamp')
    list_filter = ('action', 'timestamp')