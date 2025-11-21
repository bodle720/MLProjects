from cvdms_platform import CvdmsApp

app = CvdmsApp(app_name="cvdmsv1",
               profile_name="developers_admin")


ok, upload_attempt_info = app.upload_imagery("sample/sample.csv",
                                                     summary="my first test") # summary of the job for the job table

if ok:
    ok, results, df_res = app.get_logs_by_job_id(upload_attempt_info['job_id'])

