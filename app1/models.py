from django.db import models
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth.models import User
from datetime import datetime, timedelta
from datetime import time

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=255)
    faculty = models.CharField(max_length=100, null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    email = models.EmailField(max_length=255,null=True, blank=True)
    phone_number = models.CharField(max_length=15,null=True, blank=True)
    Student_Id = models.CharField(max_length=100, unique=True)
    image = models.ImageField(upload_to='Student/')
    authorized = models.BooleanField(default=False)

    def __str__(self):
        return self.Student_Id

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE,null=True,blank=True)
    date = models.DateField(default=timezone.now)
    check_in_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=10,
        choices=[
              ('Present', 'Present'),
              ('late','late') ,
              ('Absent', 'Absent')],
        default='Absent'  
    )
    last_action_time = models.DateTimeField(null=True, blank=True)
      # Class-specific fields
    class_start_time = models.DateTimeField(null=True, blank=True)
    grace_period_minutes = 15  # allowed late minutes
    inside_time = models.DurationField(default=timedelta(0))  # Add default
    extra_time = models.DurationField(default=timedelta(0))  # optional, for leaving class early
    # # Track multiple sessions for in/out periods
    # sessions = models.JSONField(default=list)  # [{'in': "timestamp", 'out': "timestamp"}]
    # inside_time = models.DurationField(default=timedelta(0))
    sessions = models.IntegerField(default=1)  # Add a default value

    def __str__(self):
        return f"{self.student.Student_Id} - {self.date} ({self.status})"

    def mark_checked_in(self):
     
        now = timezone.now()
        if not self.check_in_time:
           self.check_in_time = now
           
           self.last_action_time = now
        #------------
        self.update_inside_time()
        self.update_status()
        self.save()

    # Update the status field immediately
        # self.status = self.attendance_status()  # call your method
        # self.save()
        

    def mark_checked_out(self):
        now = timezone.now()
        if self.check_in_time and not self.check_out_time:
          self.check_out_time = now
          self.last_action_time = now
        # Recalculate status just in case
          self.status = self.attendance_status()
          self.save()
        else:
           raise ValueError("Cannot mark check-out without check-in.")


    def calculate_duration(self):
        """Return total duration student was present"""
        if self.check_in_time:
            end_time = self.check_out_time or timezone.now()
            total = end_time - self.check_in_time
            return total if total.total_seconds() > 0 else timedelta(0)
        return timedelta(0)

    def formatted_duration(self):
            
        # Return HH:MM:SS format of duration
            duration = self.calculate_duration()
            total_seconds = int(duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        

       # ---------- ATTENDANCE STATUS (KEY LOGIC) ----------
    def attendance_status(self):
        """
        Before 11:00 AM  -> Present
        After 11:00 AM   -> Late
        """
        if not self.check_in_time:
            return "Absent"

        check_in_time_local = timezone.localtime(self.check_in_time).time()
        present_cutoff = time(11, 0, 0)  # 11:00 AM
        # late_cutoff=time(12,30,0)

        if check_in_time_local < present_cutoff:
            return "Present"
        # elif check_in_time_local<= late_cutoff:
        #     return "Late"
        else:
            return "Late"
        #----------
    def update_status(self):
        self.status = self.attendance_status()
    def calculate_duration(self):
        """Return total duration student was inside"""
        if self.check_in_time:
            end_time = self.check_out_time or timezone.now()
            total = end_time - self.check_in_time + self.extra_time
            return total if total.total_seconds() > 0 else timedelta(0)
        return timedelta(0)

    def update_inside_time(self):
        self.inside_time = self.calculate_duration()

    def formatted_duration(self):
        duration = self.calculate_duration()
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    # ------------------- Override Save -------------------
    def save(self, *args, **kwargs):
        # Set today's date for new records
        if not self.pk:
            self.date = timezone.now().date()
        # Always update status before saving
        self.update_status()
        # Update inside_time in case check-in/check-out exists
        self.update_inside_time()
        super().save(*args, **kwargs)

    # def is_late(self):
    #     return self.status == "Late"
    
    # def attendance_status(self):
    #     """Return attendance status based on class start time"""
    #     if not self.check_in_time:
    #         return "Absent"
    #     if not self.class_start_time:
    #         return "Present"

    #     # Define latest allowed check-in time
    #     grace_limit = self.class_start_time + timedelta(minutes=self.grace_period_minutes)

    #     if self.check_in_time <= grace_limit:
    #         return "Present"
    #     else:
    #         return "Late"

    def is_late(self):
        return self.attendance_status() == "Late"

    def set_class_times(self, start_time_str, end_time_str, time_format="%H:%M"):
        """Set class start and end times from string input"""
        self.class_start_time = datetime.strptime(start_time_str, time_format)
        self.class_end_time = datetime.strptime(end_time_str, time_format)


    
    
    

    # def save(self, *args, **kwargs):
    #     if not self.pk:
    #         self.date = timezone.now().date()
    #     super().save(*args, **kwargs)

class AttendanceLog(models.Model):
    attendance = models.ForeignKey(Attendance, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(default=timezone.now)
    action = models.CharField(max_length=10, choices=[('checkin', 'Check In'), ('checkout', 'Check Out')])

    def __str__(self):
        return f"{self.attendance.student.Student_Id} - {self.action} at {self.timestamp.strftime('%H:%M:%S')}"

class CameraConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Give a name to this camera configuration")
    camera_source = models.CharField(max_length=255, help_text="Camera index (0 for default webcam or RTSP/HTTP URL for IP camera)")
    threshold = models.FloatField(default=0.6, help_text="Face recognition confidence threshold")

    def __str__(self):
        return self.name

   





