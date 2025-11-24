from cvdms_platform import CvdmsApp
import pandas as pd

app = CvdmsApp(app_name="cvdmsv1",
               profile_name="developers_admin")

ok, upload_attempt_info = app.upload_imagery("sample/sample.csv",
                                                     summary="my first test with the new infrastructure",
                                                     source='FashionMNIST') # summary of the job for the job table

if ok:
    job_id = upload_attempt_info["job_id"]
    log_retrieval_success, log_info, log_df_results = app.get_logs_by_job_id(job_id)

if log_retrieval_success:
    log_df_results.to_csv(f'logs_{job_id}.csv', index=False)