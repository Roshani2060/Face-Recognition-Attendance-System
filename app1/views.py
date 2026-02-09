import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Student, Attendance, CameraConfiguration
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .models import Student
from django.shortcuts import render
from django.db.models import Count
from django.shortcuts import render
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from .models import Attendance
from .forms import StudentLoginForm
from .models import AttendanceLog, Attendance
from django.http import JsonResponse
from django.views.decorators.http import require_GET

def attendance_chart(request):
    today = timezone.now().date()
    # Calculate start (Monday) and end (Sunday) of the current week
    week_start = today - timedelta(days=today.weekday())        # Monday
    week_end = week_start + timedelta(days=6)                   # Sunday

    # Get counts
    today_count = Attendance.objects.filter(date=today).count()
    week_count=Attendance.objects.filter(date__range=[week_start, week_end]).count()
    month_count = Attendance.objects.filter(date__month=today.month, date__year=today.year).count()
    year_count = Attendance.objects.filter(date__year=today.year).count()
    total_count = Attendance.objects.all().count()

    context = {
        'today_count': today_count,
        'week_count': week_count,
        'month_count': month_count,
        'year_count': year_count,
        'total_count': total_count,
    }

    return render(request, 'attendance_chart.html', context)


# Initialize MTCNN and InceptionResnetV1
#mtcnn = MTCNN(keep_all=True)
mtcnn = MTCNN(
    keep_all=True,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    device='cpu'
)

resnet = InceptionResnetV1(pretrained='vggface2').eval()



# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32)

        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []




# Function to encode uploaded images
def encode_uploaded_images():
    known_face_encodings = []
    known_face_names = []

    # Fetch only authorized images
    uploaded_images = Student.objects.filter(authorized=True)

    for student in uploaded_images:
        image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        known_image = cv2.imread(image_path)
        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
        encodings = detect_and_encode(known_image_rgb)
        if encodings:
            known_face_encodings.extend(encodings)
            known_face_names.append(student.Student_Id)

    return known_face_encodings, known_face_names





def student_login(request):
    error = None
    if request.method == 'POST':
        form = StudentLoginForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            stu_id = form.cleaned_data['student_id']
            try:
                student = Student.objects.get(name=name, Student_Id=stu_id)
                request.session['student_id'] = stu_id  # Save to session
                return redirect('student_dashboard')
            except Student.DoesNotExist:
                error = "Invalid credentials"
    else:
        form = StudentLoginForm()
    return render(request, 'user_login.html', {'form': form, 'error': error})


def student_dashboard(request):
    stu_id = request.session.get('student_id')
    if not stu_id:
        return redirect('student_login')
    
    try:
        student = Student.objects.get(Student_Id=stu_id)
        attendance_records = Attendance.objects.filter(student=student).order_by('-date')
        return render(request, 'dashboard.html', {'student': student, 'attendance_records': attendance_records})
    except Student.DoesNotExist:
        return redirect('student_login')

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

def check_student_id(request):
    stu_id = request.GET.get('id', '')
    exists = Student.objects.filter(Student_Id=stu_id).exists()  # Correct field name here
    return JsonResponse({'exists': exists})

# View for capturing student information and image
def capture_Student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        faculty=request.POST.get('faculty')
        year=request.POST.get('year')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        Student_Id = request.POST.get('Student_Id')
        image_data = request.POST.get('image_data')

        # Check if Student_Id already exists
        if Student.objects.filter(Student_Id=Student_Id).exists():
            messages.error(request, "Student ID already exists. Please choose a different ID.")
            return render(request, 'capture_Student.html', {
                'name': name,
                'faculty':faculty,
                'year':year,
                'email': email,
                'phone_number': phone_number,
                'Student_Id': Student_Id,
            })

        if image_data:
            try:
                header, encoded = image_data.split(',', 1)
                image_file = ContentFile(base64.b64decode(encoded), name=f"{Student_Id}.jpg")
            except Exception as e:
                messages.error(request, "Invalid image data.")
                return render(request, 'capture_Student.html')

            student = Student(
                name=name,
                faculty=faculty,
                year=year,
                email=email,
                phone_number=phone_number,
                Student_Id=Student_Id,
                image=image_file,
                authorized=False  # Default to False during registration
            )
            student.save()

            return redirect('selfie_success')  # Redirect to a success page

    return render(request, 'capture_Student.html')

def logout_view(request):
    logout(request)
    return redirect('student_login')




# Success view after capturing student information and image
def selfie_success(request):
    return render(request, 'selfie_success.html')

def student_logs(request, stu_id):
    # Get student object by Student_Id or 404 if not found
    student = get_object_or_404(Student, Student_Id=stu_id)
    # Filter logs related to this student (via attendance relation)
    logs = AttendanceLog.objects.filter(attendance__student=student).order_by('-timestamp')
    
    context = {
        'student': student,
        'logs': logs,
    }
    return render(request, 'student_logs.html', context)

def log_attendance_action(attendance, action):
    AttendanceLog.objects.create(attendance=attendance, action=action)
    attendance.last_action_time = timezone.now()
    attendance.save()
# This views for capturing studen faces and recognize
def capture_and_recognize(request):
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        try:
            # Check if the camera source is a number (local webcam) or a string (IP camera URL)
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source))  # Use integer index for webcam
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  # Use string for IP camera URL

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav')  # load sound path

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break  # If frame capture fails, break from the loop
                frame = cv2.flip(frame, 1) 

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #test_face_encodings = detect_and_encode(frame_rgb)  # Function to detect and encode face in frame
                boxes, probs = mtcnn.detect(frame_rgb)
                
                # ---------- NO FACE ----------
                if boxes is None or len(boxes) == 0:
                    cv2.putText(
                        frame,
                        "No face detected. Look at the camera",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    frame = cv2.resize(frame, (960, 720))
                    cv2.imshow(window_name, frame)
                    continue
                # ---------- MULTIPLE FACES ----------
                    
                if len(boxes) > 1:
                    cv2.putText(
                        frame,
                        "Multiple faces detected. Only one student allowed",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    cv2.imshow(window_name, frame)
                    continue
                #test_face_encodings = detect_and_encode(frame_rgb)
                boxes, _ = mtcnn.detect(frame_rgb)
                if boxes is None:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow(window_name, frame)
                    continue
                test_face_encodings = []
                face_boxes = []

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    face = cv2.resize(face, (160, 160))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0

                    face_tensor = torch.tensor(face).unsqueeze(0)
                    with torch.no_grad():
                        encoding = resnet(face_tensor).numpy().flatten()

                    test_face_encodings.append(encoding)
                    face_boxes.append((x1, y1, x2, y2))








                

                

                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()  # Load known face encodings once
                    if known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)
                        # Draw bounding boxes and labels
                    for name, (x1, y1, x2, y2) in zip(names, face_boxes):

                        # Green = recognized, Red = not recognized
                        color = (0, 255, 0) if name != "Not Recognized" else (0, 0, 255)

                        # Draw face box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Show name or message
                        cv2.putText(
                            frame,
                            name,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2
                        )


                        for name, box, prob in zip(names, boxes, probs):
                            if box is not None:
                                (x1, y1, x2, y2) = map(int, box)
                                    # ---------- FACE NOT CLEAR ----------
                                if prob < 0.95:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(
                                        frame,
                                        "Face not clear. Come closer",
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2
                                    )
                                    continue

                                # ---------- VALID FACE ----------
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    name,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2
                                )
                                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                #cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                if name != 'Not Recognized':
                                    student = Student.objects.filter(Student_Id=name)
                                    if student.exists():
                                        student = student.first()







                                        attendance, created = Attendance.objects.get_or_create(student=student,
                                                                                               date=timezone.now().date())
                                        now = timezone.now()

       
                                      

                                        if created:
                                            # First-time check-in for today
                                            attendance.mark_checked_in()
                                            attendance.status = 'Present' 
                                            attendance.save()
                        
                                            log_attendance_action(attendance, 'checkin')
                                            
                                            success_sound.play()
                                            cv2.putText(frame, f"{name}, checked in.", (50, 50),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                        else:
                                            # Already have attendance record for today
                                            if attendance.check_in_time and not attendance.check_out_time:
                                                # Trying to check-out
                                                can_checkout = (now >= attendance.check_in_time + timedelta(seconds=10)) and \
                                                            (attendance.last_action_time is None or (now - attendance.last_action_time) >= timedelta(seconds=10))

                                                if can_checkout:
                                                    attendance.mark_checked_out()
                                                    attendance.status = 'Absent'
                                                    attendance.save()
                                                    log_attendance_action(attendance, 'checkout')
                                                    success_sound.play()

                                                    # Show inside office time (excluding extra)
                                                    formatted_time = attendance.formatted_duration()
                                                    cv2.putText(frame, f"{name}, checked out.", (50, 50),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                                                    cv2.putText(frame, f"Class Time: {formatted_time}", (50, 80),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, wait before checkout.", (50, 50),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                                            elif attendance.check_in_time and attendance.check_out_time:
                                                # Trying to re-check-in
                                                can_recheckin = (attendance.last_action_time is None) or ((now - attendance.last_action_time) >= timedelta(seconds=10))

                                                if can_recheckin:
                                                    if not attendance.extra_time:
                                                        attendance.extra_time = timedelta(0)

                                                    # Add outside time to extra_time
                                                    attendance.extra_time += now - attendance.check_out_time
                                                    attendance.check_out_time = None
                                                    attendance.last_action_time = now
                                                    attendance.status = 'Present'   # ✅ ADD THIS

                                                    attendance.save()

                                                    log_attendance_action(attendance, 'checkin')
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked in again.", (50, 50),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                                                    # Show current inside time
                                                    formatted_time = attendance.formatted_duration()
                                                    cv2.putText(frame, f"Class Time: {formatted_time}", (50, 80),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, wait before re-check-in.", (50, 50),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name)  # Only destroy if window was created

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Ensure all windows are closed in the main thread
        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:  # Check if window exists
                cv2.destroyWindow(window)

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    return redirect('Student_attendance_list')

#this is for showing Attendance list
def Student_attendance_list(request):
    # Get the search query and date filter from the request
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all student
    student = Student.objects.all()

    # Filter student based on the search query
    if search_query:
        student = student.filter(name__icontains=search_query)

    # Prepare the attendance data
    student_attendance_data = []

    for student in student:
        # Get the attendance records for each student, filtering by attendance date if provided
        attendance_records = Attendance.objects.filter(student=student)

        if date_filter:
            # Assuming date_filter is in the format YYYY-MM-DD
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')
        
        student_attendance_data.append({
            'student':student,
            'attendance_records': attendance_records
        })

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query,  # Pass the search query to the template
        'date_filter': date_filter       # Pass the date filter to the template
    }
    return render(request, 'Student_attendance_list.html', context)

# def present_students(request):
#     today = timezone.now().date()

#     present_attendance = Attendance.objects.filter(
#         date=today,
#         check_in_time__isnull=False,
#         check_out_time__isnull=True
#     ).select_related('student')

#     context = {
#         'present_attendance': present_attendance
#     }

#     return render(request, 'present_students.html', context)

# def absent_students(request):
#     return render(request, 'absent_students.html')


def home(request):
    total_student = Student.objects.count()
    total_attendance = Attendance.objects.count()
    total_check_ins = Attendance.objects.filter(check_in_time__isnull=False).count()
    total_check_outs = Attendance.objects.filter(check_out_time__isnull=False).count()
    # total_present = Attendance.objects.filter(status='Present').count()
    # total_absent = Attendance.objects.filter(status='Absent').count()
    total_cameras = CameraConfiguration.objects.count()

    context = {
        'total_student': total_student,
        'total_attendance': total_attendance,
        'total_check_ins': total_check_ins,
        'total_check_outs': total_check_outs,
        # 'total_present' :total_present,
        # 'total_absent' :total_absent,
        'total_cameras': total_cameras,
    }

    return render(request, 'home.html', context)

   


# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def Student_list(request):
    student = Student.objects.all()
    return render(request, 'Student_list.html', {'Student': student})

@login_required
@user_passes_test(is_admin)
def Student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'Student_detail.html', {'Student': student})

@login_required
@user_passes_test(is_admin)
def Student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        return redirect('Student-detail', pk=pk)
    
    return render(request, 'Student_authorize.html', {'Student': student})

# This views is for Deleting student
@login_required
@user_passes_test(is_admin)
def Student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('Student-list')  # Redirect to the student list after deletion
    
    return render(request, 'Student_delete_confirm.html', {'Student': student})

@login_required
@user_passes_test(is_admin)
def Student_edit(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.name = request.POST.get('name', student.name)
        student.faculty = request.POST.get('faculty', student.faculty)
        student.year = request.POST.get('year', student.year)
        student.email = request.POST.get('email', student.email)
        student.phone_number = request.POST.get('phone_number', student.phone_number)
        student.save()
        messages.success(request, 'Student details updated successfully.')
        return redirect('Student-detail', pk=pk)
    
    return render(request, 'Student_edit.html', {'Student': student})




# View function for user login
def user_login(request):
    # Check if the request method is POST, indicating a form submission
    if request.method == 'POST':
        # Retrieve username and password from the submitted form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user using the provided credentials
        user = authenticate(request, username=username, password=password)

        # Check if the user was successfully authenticated
        if user is not None:
            # Log the user in by creating a session
            login(request, user)
            # Redirect the user to the student list page after successful login
            return redirect('home')  # Replace 'student-list' with your desired redirect URL after login
        else:
            # If authentication fails, display an error message
            messages.error(request, 'Invalid username or password.')

    # Render the login template for GET requests or if authentication fails
    return render(request, 'login.html')




# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('login')  # Replace 'login' with your desired redirect URL after logout

# Function to handle the creation of a new camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
#                 name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations
@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        # Save the changes to the database
        config.save()  

        # Redirect to the list page after successful update
        return redirect('camera_config_list')  
    
    # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config})


# DELETE: Function to delete a camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()  
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

#     # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})










