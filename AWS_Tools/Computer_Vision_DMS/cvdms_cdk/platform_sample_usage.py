from cvdms_platform import CvdmsApp

upload_sample_csv_path = r"sample.csv"

app = CvdmsApp(app_name="cvdmsv1",
               profile_name="developers_admin")

successful, upload_attempt_info = app.upload_imagery(upload_sample_csv_path,
                                                     summary="my first test") # summary of the job for the job table

